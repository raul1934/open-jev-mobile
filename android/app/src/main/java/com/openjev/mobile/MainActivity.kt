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
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.runtime.Composable
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
        intent.getStringExtra("run")?.let(viewModel::runWhenReady)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        intent?.getStringExtra("run")?.let(viewModel::runWhenReady)
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
    Column(
        Modifier.fillMaxSize().safeDrawingPadding().verticalScroll(rememberScrollState()).padding(16.dp),
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
        if (state.model == ModelState.Ready) {
            RequestEditor(state, vm)
            state.running?.let { (done, total) ->
                Column {
                    Text("Avaliando candidato $done de $total…", style = MaterialTheme.typography.bodyMedium)
                    Spacer(Modifier.height(6.dp))
                    LinearProgressIndicator(progress = { if (total == 0) 0f else done / total.toFloat() }, Modifier.fillMaxWidth())
                }
            }
            state.error?.let { Text("Erro: $it", color = MaterialTheme.colorScheme.error) }
            state.result?.let { result ->
                result.answers.forEach { (id, answer) -> AnswerCard(id, answer) }
                Text(
                    "${result.candidates} candidatos, ${result.inputTokens} tokens, " +
                        String.format(Locale.ROOT, "%.1f s", result.seconds),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
    if (showSettings) SettingsDialog(state, onDismiss = { showSettings = false }) { threads, ctx ->
        showSettings = false
        vm.applySettings(threads, ctx)
    }
}

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
private fun Bar(label: String, p: Double) {
    Column {
        Row {
            Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall, maxLines = 2,
                 overflow = TextOverflow.Ellipsis)
            Text(pct(p), style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium)
        }
        LinearProgressIndicator(progress = { p.toFloat() }, Modifier.fillMaxWidth(), drawStopIndicator = {})
    }
}

@Composable
private fun SettingsDialog(state: UiState, onDismiss: () -> Unit, onApply: (Int, Int) -> Unit) {
    val cores = Runtime.getRuntime().availableProcessors()
    var threads by remember { mutableFloatStateOf(state.threads.toFloat()) }
    var ctx by remember { mutableStateOf(state.contextSize) }
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = { TextButton(onClick = { onApply(threads.roundToInt(), ctx) }) { Text("Aplicar") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancelar") } },
        title = { Text("Ajustes") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Threads da CPU: ${threads.roundToInt()} (o aparelho tem $cores núcleos)")
                Slider(value = threads, onValueChange = { threads = it }, valueRange = 1f..cores.toFloat(),
                       steps = (cores - 2).coerceAtLeast(0))
                Text("Tamanho máximo do prompt (tokens). 4096 aceita textos maiores, mas usa mais memória.")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(1024, 2048, 4096).forEach { value ->
                        FilterChip(selected = ctx == value, onClick = { ctx = value }, label = { Text("$value") })
                    }
                }
                Box(Modifier.width(1.dp))
            }
        },
    )
}

private fun pct(p: Double) = String.format(Locale.ROOT, "%.1f%%", p * 100)
private fun gb(bytes: Long) = String.format(Locale.forLanguageTag("pt-BR"), "%.2f GB", bytes / 1e9)
