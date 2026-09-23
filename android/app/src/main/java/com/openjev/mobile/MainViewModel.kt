package com.openjev.mobile

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.openjev.mobile.engine.Head
import com.openjev.mobile.engine.ModelStore
import com.openjev.mobile.engine.Native
import com.openjev.mobile.engine.Prediction
import com.openjev.mobile.engine.Predictor
import com.openjev.mobile.jev.Json
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface ModelState {
    data object Missing : ModelState
    data class Downloading(val bytes: Long, val total: Long) : ModelState
    data class Verifying(val fraction: Float) : ModelState
    data object Loading : ModelState
    data object Ready : ModelState
    data class Failed(val message: String) : ModelState
}

data class Example(val file: String, val json: String)

data class UiState(
    val model: ModelState = ModelState.Missing,
    val examples: List<Example> = emptyList(),
    val selected: String? = null,
    val requestText: String = "",
    val running: Pair<Int, Int>? = null,
    val result: Prediction? = null,
    val error: String? = null,
    val threads: Int = 4,
    val contextSize: Int = 2048,
)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val store = ModelStore(app)
    private val prefs = app.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val _state = MutableStateFlow(UiState(
        threads = prefs.getInt("threads", minOf(4, Runtime.getRuntime().availableProcessors())),
        contextSize = prefs.getInt("ctx", 2048),
    ))
    val state: StateFlow<UiState> = _state

    private var handle = 0L
    private var predictor: Predictor? = null

    init {
        loadExamples()
        viewModelScope.launch {
            withContext(Dispatchers.IO) { Native.init(app.applicationInfo.nativeLibraryDir) }
            when {
                store.isReady() -> loadModel()
                store.progress() is ModelStore.Progress.Running -> watchDownload()
                // Files copied in by hand (adb, file manager): check them before use.
                store.hasUnverifiedFiles() -> verifyAndLoad()
                else -> _state.update { it.copy(model = ModelState.Missing) }
            }
        }
    }

    private fun loadExamples() {
        val assets = getApplication<Application>().assets
        val examples = assets.list("").orEmpty().filter { it.endsWith(".json") }.sorted().map { name ->
            Example(name, assets.open(name).bufferedReader().use { it.readText() })
        }
        _state.update { it.copy(examples = examples, selected = examples.firstOrNull()?.file,
                                requestText = examples.firstOrNull()?.json.orEmpty()) }
    }

    private var pendingRun: String? = null

    /** `adb shell am start -n com.openjev.mobile/.MainActivity --es run 03-pt-avaliacao.json` (used for testing). */
    fun runWhenReady(exampleFile: String) {
        pendingRun = exampleFile
        if (_state.value.model == ModelState.Ready) startPendingRun()
    }

    private fun startPendingRun() {
        val name = pendingRun ?: return
        pendingRun = null
        _state.value.examples.firstOrNull { it.file == name }?.let { selectExample(it); run() }
    }

    fun selectExample(example: Example) =
        _state.update { it.copy(selected = example.file, requestText = example.json, result = null, error = null) }

    fun editRequest(text: String) = _state.update { it.copy(requestText = text, selected = null) }

    fun download() {
        store.startDownload()
        watchDownload()
    }

    private fun watchDownload() = viewModelScope.launch {
        while (true) {
            when (val p = withContext(Dispatchers.IO) { store.progress() }) {
                is ModelStore.Progress.Running -> _state.update { it.copy(model = ModelState.Downloading(p.bytes, p.total)) }
                ModelStore.Progress.Done -> { verifyAndLoad(); return@launch }
                is ModelStore.Progress.Failed -> { _state.update { it.copy(model = ModelState.Failed("Download falhou: ${p.reason}")) }; return@launch }
                ModelStore.Progress.Idle -> { _state.update { it.copy(model = ModelState.Missing) }; return@launch }
            }
            delay(700)
        }
    }

    fun import(uris: List<Uri>) = viewModelScope.launch {
        try {
            withContext(Dispatchers.IO) { uris.forEach { store.import(it) } }
            verifyAndLoad()
        } catch (e: Exception) {
            _state.update { it.copy(model = ModelState.Failed("Importação falhou: ${e.message}")) }
        }
    }

    private suspend fun verifyAndLoad() {
        _state.update { it.copy(model = ModelState.Verifying(0f)) }
        val problem = withContext(Dispatchers.IO) {
            store.verify { f -> _state.update { it.copy(model = ModelState.Verifying(f)) } }
        }
        if (problem != null) {
            _state.update { it.copy(model = ModelState.Failed(problem)) }
            return
        }
        loadModel()
    }

    private suspend fun loadModel() {
        _state.update { it.copy(model = ModelState.Loading) }
        try {
            val s = _state.value
            withContext(Dispatchers.IO) {
                if (handle != 0L) Native.free(handle)
                handle = 0L
                predictor = null
                val head = Head.load(store.file(ModelStore.HEAD))
                handle = Native.load(store.file(ModelStore.GGUF).absolutePath, s.contextSize, s.threads)
                predictor = Predictor(handle, head)
            }
            _state.update { it.copy(model = ModelState.Ready) }
            startPendingRun()
        } catch (e: Throwable) {
            _state.update { it.copy(model = ModelState.Failed("Não consegui carregar o modelo: ${e.message}")) }
        }
    }

    fun retry() = viewModelScope.launch {
        if (store.isReady()) loadModel() else _state.update { it.copy(model = ModelState.Missing) }
    }

    fun applySettings(threads: Int, contextSize: Int) {
        if (_state.value.running != null) return // the native context is in use
        prefs.edit().putInt("threads", threads).putInt("ctx", contextSize).apply()
        _state.update { it.copy(threads = threads, contextSize = contextSize) }
        if (store.isReady()) viewModelScope.launch { loadModel() }
    }

    fun run() {
        val p = predictor ?: return
        if (_state.value.running != null) return
        _state.update { it.copy(running = 0 to 0, result = null, error = null) }
        viewModelScope.launch {
            try {
                val request = Json.parse(_state.value.requestText)
                val result = withContext(Dispatchers.Default) {
                    p.predict(request) { done, total -> _state.update { it.copy(running = done to total) } }
                }
                android.util.Log.i("openjev", "prediction ${_state.value.selected} ${"%.1f".format(result.seconds)}s " +
                    result.answers.joinToString { (id, a) -> "$id=" + a.probabilities.joinToString("/") { (k, v) -> "$k:$v" } })
                _state.update { it.copy(result = result, running = null) }
            } catch (e: Throwable) {
                _state.update { it.copy(error = e.message ?: e.toString(), running = null) }
            }
        }
    }

    override fun onCleared() {
        if (handle != 0L) Native.free(handle)
        handle = 0L
    }
}
