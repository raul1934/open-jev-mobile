package com.openjev.mobile

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.openjev.mobile.engine.ModelStore
import com.openjev.mobile.jev.Answer
import com.openjev.mobile.jev.ChoiceAnswer
import com.openjev.mobile.jev.Json
import com.openjev.mobile.jev.JsonString
import com.openjev.mobile.jev.NoulAnswer
import com.openjev.mobile.jev.ScoreAnswer
import java.util.Locale
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels()

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        handleTestIntent(intent)
    }

    private fun handleTestIntent(intent: android.content.Intent) {
        val threads = if (intent.hasExtra("threads")) intent.getIntExtra("threads", 4) else null
        val cache = if (intent.hasExtra("cache")) intent.getBooleanExtra("cache", true) else null
        if (threads != null || cache != null) viewModel.overrideSettings(threads, cache)
        intent.getStringExtra("run")?.let(viewModel::runWhenReady)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        intent?.let(::handleTestIntent)
        setContent {
            MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
                Surface(Modifier.fillMaxSize()) {
                    val state by viewModel.state.collectAsStateWithLifecycle()
                    // Keep the screen on during downloads and long inferences.
                    val busy = state.running != null || state.model is ModelState.Downloading
                    if (busy) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    Screen(state, viewModel)
                }
            }
        }
    }
}

@Composable
private fun Screen(state: UiState, vm: MainViewModel) {
    var showSettings by remember { mutableStateOf(false) }
    val scroll = rememberScrollState()
    // Answers are below the editor: bring them into view when a run (or the optimizer) finishes.
    LaunchedEffect(state.result, state.optimizer, state.error) {
        if (state.result != null || state.optimizer != null || state.error != null) {
            kotlinx.coroutines.delay(150)
            scroll.animateScrollTo(scroll.maxValue)
        }
    }
    Column(
        Modifier.fillMaxSize().safeDrawingPadding().verticalScroll(scroll).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(Modifier.fillMaxWidth()) {
            Column(Modifier.weight(1f)) {
                Text("Open-Jev", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
                Text("Open-Jev-2B no aparelho, sem internet", style = MaterialTheme.typography.bodySmall)
            }
            TextButton(onClick = { showSettings = true }, enabled = state.running == null) { Text("Ajustes") }
        }
        ModelCard(state.model, vm)
        StatusPanel(state)
        if (state.model == ModelState.Ready) {
            RequestEditor(state, vm)
            state.error?.let { Text("Erro: $it", color = MaterialTheme.colorScheme.error) }
            state.optimizer?.let { OptimizerCard(it) }
            state.formats?.let { FormatsCard(it, state.gguf, vm) }
            state.result?.let { result ->
                result.answers.forEach { (id, answer) -> AnswerCard(id, answer) }
            }
        }
    }
    if (showSettings) SettingsDialog(
        state,
        onDismiss = { showSettings = false },
        onOptimize = { showSettings = false; vm.optimize() },
        onTestFormats = { showSettings = false; vm.testFormats() },
        onSelectModel = { name -> showSettings = false; vm.useFormat(name) },
    ) { threads, ctx, cache, auto ->
        showSettings = false
        vm.applySettings(threads, ctx, cache, auto)
    }
}

/** Live timer, current step and memory; after a run, its time, time per option and memory peak. */
@Composable
private fun StatusPanel(state: UiState) {
    val running = state.running
    val result = state.result
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (running != null) {
                val elapsed = (state.now - running.startedAt).coerceAtLeast(0) / 1000.0
                Row {
                    Text("Rodando", Modifier.weight(1f), fontWeight = FontWeight.SemiBold)
                    Text(seconds(elapsed), fontFamily = FontFamily.Monospace, fontWeight = FontWeight.SemiBold,
                         style = MaterialTheme.typography.titleMedium)
                }
                Text(running.step + "…", style = MaterialTheme.typography.bodyMedium)
                LinearProgressIndicator(
                    progress = { if (running.total == 0) 0f else running.done / running.total.toFloat() },
                    Modifier.fillMaxWidth(),
                )
                Text("${running.done} de ${running.total} opções prontas" +
                     if (running.done > 0) " · ${seconds(elapsed / running.done)} por opção" else "",
                     style = MaterialTheme.typography.bodySmall)
            } else if (result != null) {
                Text("Última execução: ${seconds(result.seconds)} · ${result.candidates} opções · " +
                     "${seconds(result.seconds / result.candidates)} por opção · " +
                     "${result.decodedTokens} de ${result.inputTokens} tokens calculados" +
                     if (state.prefixCache) " (contexto reaproveitado)" else "",
                     style = MaterialTheme.typography.bodyMedium)
                // Where the time went, largest first.
                Text(result.profile.entries.entries.sortedByDescending { it.value.seconds }.joinToString(" · ") { (k, e) ->
                    "$k ${String.format(Locale.ROOT, "%.1f", e.seconds)} s" + if (e.tokens > 0) " (${e.tokens} tok)" else ""
                }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            state.system?.let { DeviceLines(it) }
            state.memory?.let { m ->
                Text(
                    "Memória do app: ${gb(m.appBytes)}" +
                        (if (running != null || result != null) " (pico ${gb(state.peakAppBytes)})" else "") +
                        " · livre no aparelho: ${gb(m.availBytes)} de ${gb(m.totalBytes)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (m.low) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (m.low) Text("Pouca memória livre: feche outros apps.", style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error)
            }
        }
    }
}

/** CPU use, clocks, GPU and temperatures; lines are skipped when Android does not expose a value. */
@Composable
private fun DeviceLines(s: com.openjev.mobile.engine.SystemMonitor.Snapshot) {
    val small = MaterialTheme.typography.bodySmall
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    fun c(v: Double?) = v?.let { String.format(Locale.ROOT, "%.0f °C", it) }
    val cpu = buildList {
        s.appCores?.let { add("app usando " + String.format(Locale.ROOT, "%.1f", it) + " de ${s.cores} núcleos") }
        if (s.coreMHz.isNotEmpty()) {
            // Group identical clocks: "2×2208 + 6×1804 MHz".
            add(s.coreMHz.groupingBy { it }.eachCount().entries.sortedByDescending { it.key }
                .joinToString(" + ") { "${it.value}×${it.key}" } + " MHz")
        }
        c(s.cpuTempC)?.let { add(it) }
    }
    if (cpu.isNotEmpty()) Text("CPU: " + cpu.joinToString(" · "), style = small, color = muted)
    val gpu = buildList {
        add("não usada pelo modelo")
        s.gpuBusyPercent?.let { add("uso $it%") }
        s.gpuMHz?.let { add("$it MHz") }
        c(s.gpuTempC)?.let { add(it) }
    }
    Text("GPU: " + gpu.joinToString(" · "), style = small, color = muted)
    val temps = buildList {
        c(s.skinTempC)?.let { add("superfície $it") }
        c(s.batteryTempC)?.let { add("bateria $it") }
    }
    if (temps.isNotEmpty()) Text("Temperatura: " + temps.joinToString(" · "), style = small, color = muted)
    s.thermalStatus?.let { status ->
        val name = when (status) {
            0 -> "normal"; 1 -> "leve"; 2 -> "moderado"; 3 -> "severo"; 4 -> "crítico"; 5 -> "emergência"
            else -> "desligando"
        }
        val headroom = s.thermalHeadroom?.let { " · " + String.format(Locale.ROOT, "%.0f%%", it * 100) + " do limite" } ?: ""
        val throttling = status >= 2 || (s.thermalHeadroom ?: 0f) >= 1f
        Text("Térmico: $name$headroom" + if (throttling) " · o sistema está reduzindo a velocidade" else "",
             style = small, color = if (throttling) MaterialTheme.colorScheme.error else muted)
    }
}

private fun seconds(s: Double) = if (s < 60) String.format(Locale.ROOT, "%.1f s", s)
                                 else String.format(Locale.ROOT, "%d min %02d s", (s / 60).toInt(), (s % 60).toInt())

@Composable
private fun ModelCard(model: ModelState, vm: MainViewModel) {
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) vm.import(uris)
    }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            when (model) {
                ModelState.Missing, is ModelState.Failed -> {
                    if (model is ModelState.Failed) Text(model.message, color = MaterialTheme.colorScheme.error)
                    Text("O modelo (${gb(ModelStore.GGUF.bytes)}) ainda não está no aparelho. " +
                         "Ele é baixado uma vez do GitHub e depois tudo roda offline.")
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = vm::download) { Text("Baixar modelo") }
                        OutlinedButton(onClick = { picker.launch(arrayOf("*/*")) }) { Text("Importar arquivos") }
                    }
                    if (model is ModelState.Failed) TextButton(onClick = { vm.retry() }) { Text("Tentar de novo") }
                    Text("Importar: escolha o .gguf e o head.json juntos, por exemplo da pasta Download.",
                         style = MaterialTheme.typography.bodySmall)
                }
                is ModelState.Downloading -> {
                    Text("Baixando o modelo: ${gb(model.bytes)} de ${gb(model.total)}")
                    LinearProgressIndicator(progress = { model.bytes / model.total.toFloat() }, Modifier.fillMaxWidth())
                    Text("Pode sair do app: o download continua em segundo plano.", style = MaterialTheme.typography.bodySmall)
                }
                is ModelState.Verifying -> {
                    Text("Conferindo o arquivo (SHA-256)…")
                    LinearProgressIndicator(progress = { model.fraction }, Modifier.fillMaxWidth())
                }
                ModelState.Loading -> {
                    Text("Carregando o modelo na memória…")
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
                ModelState.Ready -> Text("Modelo pronto: Open-Jev-2B (Q5_K_M)", fontWeight = FontWeight.Medium)
            }
        }
    }
}

@Composable
private fun RequestEditor(state: UiState, vm: MainViewModel) {
    Text("Prompts de teste", style = MaterialTheme.typography.titleSmall)
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        state.examples.forEach { example ->
            FilterChip(
                selected = example.file == state.selected,
                onClick = { vm.selectExample(example) },
                label = { Text(exampleLabel(example)) },
                enabled = state.running == null,
            )
        }
    }
    OutlinedTextField(
        value = state.requestText,
        onValueChange = vm::editRequest,
        label = { Text("Pedido (JSON com state e questions)") },
        textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace, fontSize = 12.sp),
        modifier = Modifier.fillMaxWidth().heightIn(min = 160.dp, max = 360.dp),
        enabled = state.running == null,
    )
    Button(onClick = vm::run, enabled = state.running == null, modifier = Modifier.fillMaxWidth()) { Text("Rodar") }
}

/** "02-pt-atendimento.json" -> "pt · atendimento". */
private fun exampleLabel(example: Example): String {
    val parts = example.file.removeSuffix(".json").split('-')
    return if (parts.size >= 3) parts[1] + " · " + parts.drop(2).joinToString(" ") else example.file
}

/** Measured on this device; the fidelity numbers were measured on the PC against the original model. */
private val FIDELITY = mapOf(
    "Q4_0" to "pior 24,1 · média 4,9 pts · mudou 1 resposta em 25", "Q4_K_M" to "pior 23,6 · média 4,1 pts", "Q5_K_M" to "pior 7,8 · média 2,0 pts",
    "Q6_K" to "pior 5,9 · média 1,2 pts", "Q8_0" to "pior 4,1 · média 0,7 pts",
)

@Composable
private fun FormatsCard(results: List<FormatResult>, current: String?, vm: MainViewModel) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Formatos do modelo neste aparelho", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            val fastest = results.filter { it.error == null }.minOfOrNull { it.seconds }
            results.forEach { r ->
                val quant = r.name.removePrefix("open-jev-2b-").removeSuffix(".gguf")
                val inUse = (current ?: ModelStore.GGUF.name) == r.name
                Column {
                    Text("$quant · ${gb(r.bytes)}" + if (inUse) " · em uso" else "", fontWeight = FontWeight.Medium)
                    if (r.error != null) {
                        Text("Falhou: ${r.error}", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                    } else {
                        Text("${seconds(r.seconds)} para ${r.tokens} tokens (" +
                             String.format(Locale.ROOT, "%.0f", r.tokens / r.seconds) + " tokens/s)" +
                             (if (r.seconds == fastest) " ✓ mais rápido" else "") +
                             " · memória ${gb(r.appBytes)}", style = MaterialTheme.typography.bodySmall)
                        FIDELITY[quant]?.let {
                            Text("Diferença do modelo original (PC): $it", style = MaterialTheme.typography.bodySmall,
                                 color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        if (!inUse) TextButton(onClick = { vm.useFormat(r.name) }) { Text("Usar este formato") }
                    }
                }
            }
        }
    }
}

@Composable
private fun OptimizerCard(r: OptimizerResult) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Otimização do aparelho", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text("CPU: ${r.cpu} · RAM: ${String.format(Locale.ROOT, "%.1f", r.ramGb)} GB",
                 style = MaterialTheme.typography.bodySmall)
            // One row per chunk size: time with each thread count, the fastest marked.
            r.trials.groupBy { it.tokens }.toSortedMap().forEach { (tokens, list) ->
                val best = list.minBy { it.seconds }
                Text("$tokens tokens: " + list.joinToString("  ") {
                    "${it.threads}t ${String.format(Locale.ROOT, "%.2f", it.seconds)}s" + if (it == best) " ✓" else ""
                }, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
            }
            Text("Aplicado: threads automáticas (${r.policy.describe()}), contexto de ${r.contextSize} tokens, " +
                 "reaproveitar contexto ligado.", style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun AnswerCard(id: String, answer: Answer) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(id, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            when (answer) {
                is NoulAnswer -> {
                    Text("sim/não · probabilidade de SIM: ${pct(answer.noul)}")
                    Bar("sim", answer.noul)
                }
                is ChoiceAnswer -> {
                    Text("escolha: ${answer.choice} · confiança ${pct(answer.confidence)}")
                    answer.probabilities.forEach { (key, p) -> Bar(key, p) }
                }
                is ScoreAnswer -> {
                    Text("nota: " + String.format(Locale.ROOT, "%.2f", answer.score) +
                         " (0 a ${answer.probabilities.size - 1}) · confiança ${pct(answer.confidence)}")
                    answer.probabilities.forEach { (key, p) ->
                        val text = when (val level = answer.legend[key]) {
                            is JsonString -> level.value
                            null -> key
                            else -> Json.dumps(level)
                        }
                        Bar("$key: $text", p)
                    }
                }
            }
        }
    }
}

@Composable
private fun Bar(label: String, p: Double, value: String = pct(p)) {
    Column {
        Row {
            Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall, maxLines = 2,
                 overflow = TextOverflow.Ellipsis)
            Text(value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium)
        }
        LinearProgressIndicator(progress = { p.toFloat() }, Modifier.fillMaxWidth(), drawStopIndicator = {})
    }
}

@Composable
private fun SettingsDialog(state: UiState, onDismiss: () -> Unit, onOptimize: () -> Unit, onTestFormats: () -> Unit,
                           onSelectModel: (String) -> Unit,
                           onApply: (Int, Int, Boolean, Boolean) -> Unit) {
    val cores = Runtime.getRuntime().availableProcessors()
    var threads by remember { mutableFloatStateOf(state.threads.toFloat()) }
    var ctx by remember { mutableStateOf(state.contextSize) }
    var cache by remember { mutableStateOf(state.prefixCache) }
    var auto by remember { mutableStateOf(state.autoThreads) }
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = { TextButton(onClick = { onApply(threads.roundToInt(), ctx, cache, auto) }) { Text("Aplicar") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancelar") } },
        title = { Text("Ajustes") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                if (state.availableModels.size > 1) {
                    Text("Modelo carregado", fontWeight = FontWeight.Medium)
                    val current = state.gguf ?: ModelStore.GGUF.name
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        state.availableModels.forEach { name ->
                            FilterChip(selected = name == current, onClick = { if (name != current) onSelectModel(name) },
                                       label = { Text(name.removePrefix("open-jev-2b-").removeSuffix(".gguf")) })
                        }
                    }
                    Text("Trocar recarrega o modelo (alguns segundos). Q5_K_M é o padrão verificado.",
                         style = MaterialTheme.typography.bodySmall)
                }
                Button(onClick = onOptimize, enabled = state.model == ModelState.Ready,
                       modifier = Modifier.fillMaxWidth()) { Text("Otimizar para este aparelho") }
                OutlinedButton(onClick = onTestFormats, enabled = state.model == ModelState.Ready,
                               modifier = Modifier.fillMaxWidth()) { Text("Testar formatos do modelo") }
                Text("Mede trechos de 16 a 160 tokens com diferentes números de threads (1 a 2 minutos), " +
                     "escolhe o mais rápido para cada tamanho e ajusta o resto para a memória do aparelho.",
                     style = MaterialTheme.typography.bodySmall)
                val policy = state.threadPolicy
                Row {
                    Column(Modifier.weight(1f)) {
                        Text("Threads automáticas por trecho")
                        Text(policy?.describe() ?: "Rode a otimização primeiro.", style = MaterialTheme.typography.bodySmall)
                    }
                    Switch(checked = auto && policy != null, onCheckedChange = { auto = it }, enabled = policy != null)
                }
                Text(if (auto && policy != null) "Threads fixas (usadas com o automático desligado):" else "Threads da CPU:",
                     style = MaterialTheme.typography.bodySmall)
                Text("Threads: ${threads.roundToInt()} (o aparelho tem $cores núcleos)")
                Slider(value = threads, onValueChange = { threads = it }, valueRange = 1f..cores.toFloat(),
                       steps = (cores - 2).coerceAtLeast(0))
                Text("Tamanho máximo do prompt (tokens). 4096 aceita textos maiores, mas usa mais memória.")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(1024, 2048, 4096).forEach { value ->
                        FilterChip(selected = ctx == value, onClick = { ctx = value }, label = { Text("$value") })
                    }
                }
                Row {
                    Column(Modifier.weight(1f)) {
                        Text("Reaproveitar o contexto entre as opções")
                        Text("Bem mais rápido. Os resultados podem variar ligeiramente.",
                             style = MaterialTheme.typography.bodySmall)
                    }
                    Switch(checked = cache, onCheckedChange = { cache = it })
                }
            }
        },
    )
}

private fun pct(p: Double) = String.format(Locale.ROOT, "%.1f%%", p * 100)
private fun gb(bytes: Long) = String.format(Locale.forLanguageTag("pt-BR"), "%.2f GB", bytes / 1e9)
