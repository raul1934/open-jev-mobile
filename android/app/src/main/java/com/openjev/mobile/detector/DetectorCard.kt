package com.openjev.mobile.detector

import android.Manifest
import android.content.Context
import android.content.Intent
import android.os.Build
import android.provider.Settings
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.app.NotificationManagerCompat
import com.openjev.mobile.jev.Json
import com.openjev.mobile.jev.JsonArray
import com.openjev.mobile.jev.JsonFloat
import com.openjev.mobile.jev.JsonInt
import com.openjev.mobile.jev.JsonObject
import com.openjev.mobile.jev.JsonString
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private fun pct(p: Double) = String.format(Locale.ROOT, "%.0f%%", p * 100)

private fun listenerEnabled(context: Context) =
    NotificationManagerCompat.getEnabledListenerPackages(context).contains(context.packageName)

/** Scam detector: automatic alerts from notifications, a manual check and the recent history. */
@Composable
fun DetectorCard(busy: Boolean) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val prefs = remember { context.getSharedPreferences(ScamNotificationListener.PREFS, Context.MODE_PRIVATE) }
    var access by remember { mutableStateOf(listenerEnabled(context)) }
    var enabled by remember { mutableStateOf(ScamNotificationListener.enabled(context)) }
    var threshold by remember { mutableFloatStateOf(ScamNotificationListener.threshold(context).toFloat()) }
    var text by remember { mutableStateOf("") }
    var result by remember { mutableStateOf<String?>(null) }
    var working by remember { mutableStateOf(false) }
    var history by remember { mutableStateOf(AlertStore.load(context)) }
    val available = remember { ScamDetector.isAvailable(context) }
    val notifyPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }

    // Permission and history can change while the user is in Settings or messages arrive.
    LaunchedEffect(Unit) {
        while (true) {
            access = listenerEnabled(context)
            history = withContext(Dispatchers.IO) { AlertStore.load(context) }
            delay(2000)
        }
    }

    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Detector de golpes", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            if (!available) {
                Text("Falta o modelo do detector (${ScamDetector.ENCODER_FILE}, 132 MB) na pasta de modelos do app.",
                     color = MaterialTheme.colorScheme.error)
                return@Column
            }

            // Automatic alerts
            Row {
                Column(Modifier.weight(1f)) {
                    Text("Alertas automáticos", fontWeight = FontWeight.Medium)
                    Text(
                        when {
                            !access -> "Desligado: o app ainda não tem acesso às notificações."
                            enabled -> "Ligado: mensagens que chegam são verificadas no aparelho, nada sai dele."
                            else -> "Pausado."
                        },
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                if (access) Switch(checked = enabled, onCheckedChange = {
                    enabled = it
                    prefs.edit().putBoolean(ScamNotificationListener.KEY_ENABLED, it).apply()
                })
            }
            if (!access) {
                Button(onClick = {
                    if (Build.VERSION.SDK_INT >= 33) notifyPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
                    context.startActivity(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
                }, modifier = Modifier.fillMaxWidth()) { Text("Ativar alertas (acesso às notificações)") }
                Text("Na tela que abrir, ative \"Open-Jev\". O app lê o texto das notificações só para " +
                     "verificar golpes; o texto não é enviado para lugar nenhum.", style = MaterialTheme.typography.bodySmall)
            }
            Text("Sensibilidade: alertar a partir de ${pct(threshold.toDouble())}" +
                 if (threshold < 0.45f) " (mais alertas, mais alarmes falsos)" else if (threshold > 0.65f) " (menos alarmes falsos, pode deixar passar golpes)" else "",
                 style = MaterialTheme.typography.bodySmall)
            Slider(value = threshold, valueRange = 0.3f..0.9f, steps = 11, onValueChange = { threshold = it },
                   onValueChangeFinished = { prefs.edit().putFloat(ScamNotificationListener.KEY_THRESHOLD, threshold).apply() })

            // Manual check
            OutlinedTextField(value = text, onValueChange = { text = it }, label = { Text("Cole uma mensagem para verificar") },
                              modifier = Modifier.fillMaxWidth().heightIn(min = 80.dp), enabled = !working && !busy)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(enabled = text.isNotBlank() && !working && !busy, onClick = {
                    working = true
                    scope.launch {
                        val msg = text
                        result = withContext(Dispatchers.Default) {
                            runCatching {
                                val t0 = System.nanoTime()
                                val p = ScamDetector.probability(context, msg)
                                val ms = (System.nanoTime() - t0) / 1_000_000
                                Log.i("openjev", "scam manual chars=${msg.length} p=${"%.3f".format(Locale.ROOT, p)} ms=$ms")
                                (if (p >= threshold) "⚠️ Possível golpe" else "✓ Parece legítima") + " — ${pct(p)} de chance de golpe (${ms} ms)"
                            }.getOrElse { "Erro: ${it.message}" }
                        }
                        working = false
                    }
                }) { Text("Verificar") }
                OutlinedButton(enabled = !working && !busy, onClick = {
                    working = true
                    scope.launch {
                        result = withContext(Dispatchers.Default) { runTestSet(context, threshold.toDouble()) }
                        working = false
                    }
                }) { Text("Testar com 80 mensagens") }
            }
            if (working) LinearProgressIndicator(Modifier.fillMaxWidth())
            result?.let {
                Text(it, fontWeight = FontWeight.Medium,
                     color = if (it.startsWith("⚠️")) MaterialTheme.colorScheme.error else Color.Unspecified)
            }

            // History
            if (history.isNotEmpty()) {
                Row {
                    Text("Últimas mensagens verificadas", Modifier.weight(1f), fontWeight = FontWeight.Medium)
                    TextButton(onClick = { AlertStore.clear(context); history = emptyList() }) { Text("Limpar") }
                }
                val fmt = SimpleDateFormat("dd/MM HH:mm", Locale("pt", "BR"))
                history.take(10).forEach { e ->
                    Text((if (e.alerted) "⚠️ " else "✓ ") + "${pct(e.probability)} · ${e.app} · ${e.sender} · ${fmt.format(Date(e.time))}\n" +
                         e.text.take(140), style = MaterialTheme.typography.bodySmall,
                         color = if (e.alerted) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}

/**
 * Scores the hand-written Brazilian test set bundled in the app and compares each probability
 * with the one computed on the PC by fraud/train_e5.py (same model), so a mismatch in the
 * on-device encoder would show up. Results also go to the log for review.
 */
private fun runTestSet(context: Context, threshold: Double): String = runCatching {
    val items = (Json.parse(context.assets.open("detector/scam-expected.json").bufferedReader().use { it.readText() }) as JsonArray).items
    var tp = 0; var fp = 0; var fn = 0; var tn = 0; var maxDiff = 0.0; var totalMs = 0L
    for (item in items) {
        val o = item as JsonObject
        val text = (o["text"] as JsonString).value
        val label = (o["label"] as JsonInt).value.toInt()
        val expected = when (val v = o["probability"]) { is JsonFloat -> v.value; is JsonInt -> v.value.toDouble(); else -> 0.0 }
        val t0 = System.nanoTime()
        val p = ScamDetector.probability(context, text)
        totalMs += (System.nanoTime() - t0) / 1_000_000
        maxDiff = maxOf(maxDiff, kotlin.math.abs(p - expected))
        val alert = p >= threshold
        when { alert && label == 1 -> tp++; alert -> fp++; label == 1 -> fn++; else -> tn++ }
    }
    val n = items.size
    val summary = "Teste com $n mensagens: pegou $tp de ${tp + fn} golpes, $fp alarmes falsos em ${fp + tn} legítimas · " +
        "${totalMs / n} ms por mensagem · diferença máxima do PC: ${String.format(Locale.ROOT, "%.4f", maxDiff)}"
    Log.i("openjev", "scam test n=$n tp=$tp fp=$fp fn=$fn tn=$tn avg_ms=${totalMs / n} max_diff_vs_pc=${String.format(Locale.ROOT, "%.5f", maxDiff)} threshold=$threshold")
    summary
}.getOrElse { "Erro: ${it.message}" }
