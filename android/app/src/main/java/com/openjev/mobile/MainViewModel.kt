package com.openjev.mobile

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.openjev.mobile.engine.Head
import com.openjev.mobile.engine.DeviceInfo
import com.openjev.mobile.engine.ModelStore
import com.openjev.mobile.engine.Optimizer
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

data class OptimizerResult(val cpu: String, val trials: List<Optimizer.Trial>, val best: Int,
                           val contextSize: Int, val ramGb: Double)

data class RunProgress(val done: Int, val total: Int, val step: String, val startedAt: Long)

/** appBytes: resident memory of the app (includes the model pages mapped from the file). */
data class Memory(val appBytes: Long, val availBytes: Long, val totalBytes: Long, val low: Boolean)

data class UiState(
    val model: ModelState = ModelState.Missing,
    val examples: List<Example> = emptyList(),
    val selected: String? = null,
    val requestText: String = "",
    val running: RunProgress? = null,
    val now: Long = 0,
    val memory: Memory? = null,
    val peakAppBytes: Long = 0,
    val result: Prediction? = null,
    val optimizer: OptimizerResult? = null,
    val error: String? = null,
    val threads: Int = 4,
    val contextSize: Int = 2048,
    val prefixCache: Boolean = true,
)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val store = ModelStore(app)
    private val prefs = app.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val _state = MutableStateFlow(UiState(
        threads = prefs.getInt("threads", minOf(4, Runtime.getRuntime().availableProcessors())),
        contextSize = prefs.getInt("ctx", 2048),
        prefixCache = prefs.getBoolean("prefixCache", true),
    ))
    val state: StateFlow<UiState> = _state

    private var handle = 0L
    private var head: Head? = null
    private var predictor: Predictor? = null

    private val activityManager = app.getSystemService(android.app.ActivityManager::class.java)

    private fun readMemory(): Memory {
        // VmRSS from /proc is cheap enough to poll; Debug.getMemoryInfo() takes tens of ms.
        val rssKb = runCatching {
            java.io.File("/proc/self/status").readLines().first { it.startsWith("VmRSS:") }
                .split(Regex("\\s+"))[1].toLong()
        }.getOrDefault(0L)
        val info = android.app.ActivityManager.MemoryInfo().also { activityManager.getMemoryInfo(it) }
        return Memory(rssKb * 1024, info.availMem, info.totalMem, info.lowMemory)
    }

    init {
        loadExamples()
        // Clock and memory for the status panel: every 250 ms while running, every 2 s otherwise.
        viewModelScope.launch(Dispatchers.Default) {
            while (true) {
                val m = readMemory()
                _state.update {
                    it.copy(now = android.os.SystemClock.elapsedRealtime(), memory = m,
                            peakAppBytes = if (it.running != null) maxOf(it.peakAppBytes, m.appBytes) else it.peakAppBytes)
                }
                delay(if (_state.value.running != null) 250 else 2000)
            }
        }
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
                val h = Head.load(store.file(ModelStore.HEAD))
                handle = Native.load(store.file(ModelStore.GGUF).absolutePath, s.contextSize, s.threads)
                head = h
                predictor = Predictor(handle, h)
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

    fun applySettings(threads: Int, contextSize: Int, prefixCache: Boolean) {
        if (_state.value.running != null) return // the native context is in use
        val reload = threads != _state.value.threads || contextSize != _state.value.contextSize
        prefs.edit().putInt("threads", threads).putInt("ctx", contextSize).putBoolean("prefixCache", prefixCache).apply()
        _state.update { it.copy(threads = threads, contextSize = contextSize, prefixCache = prefixCache) }
        if (reload && store.isReady()) viewModelScope.launch { loadModel() }
    }

    /** Test hook: `--ei threads 2 --ez cache false` alongside `--es run <file>`. */
    fun overrideSettings(threads: Int?, prefixCache: Boolean?) {
        val s = _state.value
        applySettings(threads ?: s.threads, s.contextSize, prefixCache ?: s.prefixCache)
    }

    /**
     * Picks the fastest thread count for this CPU on a real prompt, turns the prefix cache on and
     * sizes the context to the RAM (4096 tokens only with 6 GB or more).
     */
    fun optimize() {
        val h = handle
        val hd = head ?: return
        if (h == 0L || _state.value.running != null) return
        val candidates = DeviceInfo.threadCandidates()
        val started = android.os.SystemClock.elapsedRealtime()
        _state.update { it.copy(running = RunProgress(0, candidates.size, "Começando", started), optimizer = null,
                                result = null, error = null, peakAppBytes = it.memory?.appBytes ?: 0) }
        viewModelScope.launch {
            try {
                // A typical prompt: the first option of the support-routing example in Portuguese.
                val example = _state.value.examples.firstOrNull { it.file.startsWith("02-") } ?: _state.value.examples.first()
                val request = Json.parse(example.json) as com.openjev.mobile.jev.JsonObject
                val record = com.openjev.mobile.jev.Api.compileRequest(request["state"]!!, request["questions"]).first()
                val prompt = com.openjev.mobile.jev.Api.candidatePrompts(record).first()
                val trials = withContext(Dispatchers.Default) {
                    val tokens = Native.tokenize(h, hd.promptPrefix + prompt + hd.promptSuffix)
                    Optimizer(h).run(tokens, candidates) { done, text ->
                        _state.update { it.copy(running = RunProgress(done, candidates.size, "$text (${tokens.size} tokens)", started)) }
                    }
                }
                val best = trials.minBy { it.seconds }.threads
                val ramBytes = _state.value.memory?.totalBytes ?: 0L
                val ctx = if (ramBytes >= 6_000_000_000L) 4096 else 2048
                withContext(Dispatchers.Default) { Native.setThreads(h, best) }
                val reload = ctx != _state.value.contextSize
                prefs.edit().putInt("threads", best).putInt("ctx", ctx).putBoolean("prefixCache", true).apply()
                _state.update {
                    it.copy(running = null, threads = best, contextSize = ctx, prefixCache = true,
                            optimizer = OptimizerResult(DeviceInfo.describe(), trials, best, ctx, ramBytes / 1e9))
                }
                android.util.Log.i("openjev", "optimizer ${DeviceInfo.describe()} " +
                    trials.joinToString { "${it.threads}t=${"%.2f".format(it.seconds)}s" } + " best=$best ctx=$ctx")
                if (reload) loadModel()
            } catch (e: Throwable) {
                _state.update { it.copy(error = "Otimização falhou: ${e.message}", running = null) }
            }
        }
    }

    fun run() {
        val p = predictor ?: return
        if (_state.value.running != null) return
        val started = android.os.SystemClock.elapsedRealtime()
        _state.update { it.copy(running = RunProgress(0, 0, "Começando", started), now = started, result = null,
                                error = null, peakAppBytes = it.memory?.appBytes ?: 0) }
        viewModelScope.launch {
            try {
                val request = Json.parse(_state.value.requestText)
                val result = withContext(Dispatchers.Default) {
                    p.predict(request, _state.value.prefixCache) { done, total, step ->
                        _state.update { it.copy(running = RunProgress(done, total, step, started)) }
                    }
                }
                android.util.Log.i("openjev", "prediction ${_state.value.selected} cache=${_state.value.prefixCache} threads=${_state.value.threads} ${"%.1f".format(result.seconds)}s " +
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
