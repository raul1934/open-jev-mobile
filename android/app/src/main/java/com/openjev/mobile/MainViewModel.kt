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
import com.openjev.mobile.engine.SystemMonitor
import com.openjev.mobile.engine.ThreadPolicy
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

data class OptimizerResult(val cpu: String, val trials: List<Optimizer.Trial>, val policy: ThreadPolicy,
                           val contextSize: Int, val ramGb: Double)

data class FormatResult(val name: String, val bytes: Long, val tokens: Int, val seconds: Double,
                        val loadSeconds: Double, val appBytes: Long, val error: String? = null)

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
    val system: SystemMonitor.Snapshot? = null,
    val peakAppBytes: Long = 0,
    val result: Prediction? = null,
    val optimizer: OptimizerResult? = null,
    val formats: List<FormatResult>? = null,
    /** GGUF file used by the app; null = the verified default. */
    val gguf: String? = null,
    /** GGUF files in the model folder, for the model selector. */
    val availableModels: List<String> = emptyList(),
    val error: String? = null,
    val threads: Int = 4,
    val contextSize: Int = 2048,
    val prefixCache: Boolean = true,
    /** Measured by the optimizer; used when [autoThreads] is on. */
    val threadPolicy: ThreadPolicy? = null,
    val autoThreads: Boolean = true,
)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val store = ModelStore(app)
    private val prefs = app.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val _state = MutableStateFlow(UiState(
        threads = prefs.getInt("threads", minOf(4, Runtime.getRuntime().availableProcessors())),
        contextSize = prefs.getInt("ctx", 2048),
        prefixCache = prefs.getBoolean("prefixCache", true),
        threadPolicy = ThreadPolicy.decode(prefs.getString("threadPolicy", null)),
        autoThreads = prefs.getBoolean("autoThreads", true),
        gguf = prefs.getString("gguf", null),
    ))

    private fun modelFile(): java.io.File =
        _state.value.gguf?.let { java.io.File(store.dir, it) }?.takeIf { it.isFile } ?: store.file(ModelStore.GGUF)

    private fun threadsForCurrentSettings(): (Int) -> Int {
        val s = _state.value
        val policy = s.threadPolicy
        return if (s.autoThreads && policy != null) policy::threadsFor else { _ -> s.threads }
    }
    val state: StateFlow<UiState> = _state

    private var handle = 0L
    private var head: Head? = null
    private var predictor: Predictor? = null

    private val activityManager = app.getSystemService(android.app.ActivityManager::class.java)
    private val monitor = SystemMonitor(app)

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
                val sys = monitor.snapshot()
                _state.update {
                    it.copy(now = android.os.SystemClock.elapsedRealtime(), memory = m, system = sys,
                            peakAppBytes = if (it.running != null) maxOf(it.peakAppBytes, m.appBytes) else it.peakAppBytes)
                }
                delay(if (_state.value.running != null) 500 else 2000)
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
                handle = Native.load(modelFile().absolutePath, s.contextSize, s.threads)
                head = h
                predictor = Predictor(handle, h).also { it.threadsFor = threadsForCurrentSettings() }
            }
            _state.update { it.copy(model = ModelState.Ready, availableModels = store.ggufFiles().map { f -> f.name }) }
            startPendingRun()
        } catch (e: Throwable) {
            _state.update { it.copy(model = ModelState.Failed("Não consegui carregar o modelo: ${e.message}")) }
        }
    }

    fun retry() = viewModelScope.launch {
        if (store.isReady()) loadModel() else _state.update { it.copy(model = ModelState.Missing) }
    }

    fun applySettings(threads: Int, contextSize: Int, prefixCache: Boolean, autoThreads: Boolean = _state.value.autoThreads) {
        if (_state.value.running != null) return // the native context is in use
        // The thread count changes per chunk without reloading; only the context size needs a reload.
        val reload = contextSize != _state.value.contextSize
        prefs.edit().putInt("threads", threads).putInt("ctx", contextSize).putBoolean("prefixCache", prefixCache)
            .putBoolean("autoThreads", autoThreads).apply()
        _state.update { it.copy(threads = threads, contextSize = contextSize, prefixCache = prefixCache, autoThreads = autoThreads) }
        predictor?.threadsFor = threadsForCurrentSettings()
        if (reload && store.isReady()) viewModelScope.launch { loadModel() }
    }

    /** Test hook: `--ei threads 2 --ez cache false` alongside `--es run <file>`. */
    fun overrideSettings(threads: Int?, prefixCache: Boolean?) {
        val s = _state.value
        applySettings(threads ?: s.threads, s.contextSize, prefixCache ?: s.prefixCache,
                      autoThreads = if (threads != null) false else s.autoThreads)
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
        _state.update { it.copy(running = RunProgress(0, 0, "Começando", started), optimizer = null,
                                result = null, error = null, peakAppBytes = it.memory?.appBytes ?: 0) }
        viewModelScope.launch {
            try {
                // A typical prompt: the first option of the support-routing example in Portuguese.
                val example = _state.value.examples.firstOrNull { it.file.startsWith("02-") } ?: _state.value.examples.first()
                val request = Json.parse(example.json) as com.openjev.mobile.jev.JsonObject
                val record = com.openjev.mobile.jev.Api.compileRequest(request["state"]!!, request["questions"]).first()
                val prompt = com.openjev.mobile.jev.Api.candidatePrompts(record).first()
                val optimizer = Optimizer(h)
                val trials = withContext(Dispatchers.Default) {
                    val tokens = Native.tokenize(h, hd.promptPrefix + prompt + hd.promptSuffix)
                    optimizer.run(tokens, candidates) { done, total, text ->
                        _state.update { it.copy(running = RunProgress(done, total, text, started)) }
                    }
                }
                predictor?.forgetThreads()
                val policy = optimizer.policy(trials)
                val best = policy.threads.last()  // fastest for the longest chunk; used if automatic is turned off
                val ramBytes = _state.value.memory?.totalBytes ?: 0L
                val ctx = if (ramBytes >= 6_000_000_000L) 4096 else 2048
                val reload = ctx != _state.value.contextSize
                prefs.edit().putInt("threads", best).putInt("ctx", ctx).putBoolean("prefixCache", true)
                    .putString("threadPolicy", policy.encode()).putBoolean("autoThreads", true).apply()
                _state.update {
                    it.copy(running = null, threads = best, contextSize = ctx, prefixCache = true,
                            threadPolicy = policy, autoThreads = true,
                            optimizer = OptimizerResult(DeviceInfo.describe(), trials, policy, ctx, ramBytes / 1e9))
                }
                predictor?.threadsFor = threadsForCurrentSettings()
                android.util.Log.i("openjev", "optimizer ${DeviceInfo.describe()} " +
                    trials.joinToString { "${it.tokens}tok/${it.threads}t=${"%.3f".format(java.util.Locale.ROOT, it.seconds)}s" } +
                    " policy=${policy.encode()} ctx=$ctx")
                if (reload) loadModel()
            } catch (e: Throwable) {
                _state.update { it.copy(error = "Otimização falhou: ${e.message}", running = null) }
            }
        }
    }

    /**
     * Loads each GGUF in the model folder in turn (the current one is freed first, so only one
     * is in memory) and times the same chunk with each, then reloads the model in use.
     */
    fun testFormats() {
        val hd = head ?: return
        if (handle == 0L || _state.value.running != null) return
        val files = store.ggufFiles()
        val started = android.os.SystemClock.elapsedRealtime()
        _state.update { it.copy(running = RunProgress(0, files.size, "Começando", started), formats = null,
                                result = null, optimizer = null, error = null, peakAppBytes = it.memory?.appBytes ?: 0) }
        viewModelScope.launch {
            val example = _state.value.examples.firstOrNull { it.file.startsWith("02-") } ?: _state.value.examples.first()
            val request = Json.parse(example.json) as com.openjev.mobile.jev.JsonObject
            val record = com.openjev.mobile.jev.Api.compileRequest(request["state"]!!, request["questions"]).first()
            val prompt = hd.promptPrefix + com.openjev.mobile.jev.Api.candidatePrompts(record).first() + hd.promptSuffix
            val s = _state.value
            val threads = s.threadPolicy?.takeIf { s.autoThreads }?.threads?.last() ?: s.threads
            val results = withContext(Dispatchers.IO) {
                Native.free(handle); handle = 0L; predictor = null
                files.mapIndexed { i, f ->
                    _state.update { it.copy(running = RunProgress(i, files.size, "Carregando ${f.name}", started)) }
                    try {
                        val t0 = System.nanoTime()
                        val h = Native.load(f.absolutePath, s.contextSize, threads)
                        val load = (System.nanoTime() - t0) / 1e9
                        try {
                            val tokens = Native.tokenize(h, prompt)
                            _state.update { it.copy(running = RunProgress(i, files.size, "Aquecendo ${f.name}", started)) }
                            Native.hiddenState(h, tokens)
                            _state.update { it.copy(running = RunProgress(i, files.size, "Medindo ${f.name} (${tokens.size} tokens)", started)) }
                            val secs = (0 until 2).minOf {
                                val t1 = System.nanoTime(); Native.hiddenState(h, tokens); (System.nanoTime() - t1) / 1e9
                            }
                            FormatResult(f.name, f.length(), tokens.size, secs, load, readMemory().appBytes)
                        } finally {
                            Native.free(h)
                        }
                    } catch (e: Throwable) {
                        FormatResult(f.name, f.length(), 0, 0.0, 0.0, 0, e.message ?: e.toString())
                    }
                }
            }
            results.forEach {
                android.util.Log.i("openjev", "format ${it.name} bytes=${it.bytes} tokens=${it.tokens} " +
                    "seconds=${"%.3f".format(java.util.Locale.ROOT, it.seconds)} load=${"%.1f".format(java.util.Locale.ROOT, it.loadSeconds)} " +
                    "rss=${it.appBytes} threads=$threads error=${it.error}")
            }
            _state.update { it.copy(running = null, formats = results) }
            loadModel()
        }
    }

    fun useFormat(name: String?) {
        if (_state.value.running != null) return
        prefs.edit().putString("gguf", name).apply()
        _state.update { it.copy(gguf = name) }
        viewModelScope.launch { loadModel() }
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
                android.util.Log.i("openjev", "prediction ${_state.value.selected} cache=${_state.value.prefixCache} threads=${
                    _state.value.threadPolicy?.takeIf { _state.value.autoThreads }?.encode() ?: _state.value.threads
                } ${"%.1f".format(java.util.Locale.ROOT, result.seconds)}s " +
                    result.answers.joinToString { (id, a) -> "$id=" + a.probabilities.joinToString("/") { (k, v) -> "$k:$v" } })
                android.util.Log.i("openjev", "profile ${_state.value.selected} total=${"%.2f".format(java.util.Locale.ROOT, result.seconds)}s " +
                    "decoded=${result.decodedTokens}tok of ${result.inputTokens} ${result.profile.describe()}")
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
