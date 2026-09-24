package com.openjev.mobile.detector

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import com.openjev.mobile.MainActivity
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Reads incoming notifications (WhatsApp, SMS, Telegram, e-mail...), scores their text with
 * [ScamDetector] on this device and posts a "possible scam" alert above the threshold.
 * Message text never leaves the device; the log records only the app, length and probability.
 */
class ScamNotificationListener : NotificationListenerService() {
    companion object {
        const val CHANNEL = "scam-alerts"
        const val PREFS = "detector"
        const val KEY_THRESHOLD = "threshold"
        const val KEY_ENABLED = "enabled"
        private const val MIN_LENGTH = 12

        fun threshold(context: Context) = context.getSharedPreferences(PREFS, MODE_PRIVATE).getFloat(KEY_THRESHOLD, 0.5f).toDouble()
        fun enabled(context: Context) = context.getSharedPreferences(PREFS, MODE_PRIVATE).getBoolean(KEY_ENABLED, true)
    }

    private val worker = Executors.newSingleThreadExecutor()
    private val recent = LinkedHashSet<Int>()  // de-duplicates repeated/updated notifications

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        if (sbn.packageName == packageName || !enabled(this)) return
        val n = sbn.notification
        if (n.flags and (Notification.FLAG_ONGOING_EVENT or Notification.FLAG_GROUP_SUMMARY) != 0) return
        if (n.category in setOf(Notification.CATEGORY_PROGRESS, Notification.CATEGORY_TRANSPORT, Notification.CATEGORY_CALL,
                                Notification.CATEGORY_SERVICE, Notification.CATEGORY_ALARM)) return
        val extras = n.extras
        val sender = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString().orEmpty()
        // Messaging apps put the newest message last in EXTRA_MESSAGES; others use big text / text.
        val messages = extras.getParcelableArray(Notification.EXTRA_MESSAGES)
        val fromMessages = messages?.lastOrNull()?.let { (it as? android.os.Bundle)?.getCharSequence("text")?.toString() }
        val text = (fromMessages ?: extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString()
                    ?: extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()).orEmpty().trim()
        if (text.length < MIN_LENGTH) return
        val key = (sbn.packageName + "\u0000" + text).hashCode()
        synchronized(recent) {
            if (!recent.add(key)) return
            if (recent.size > 200) recent.remove(recent.first())
        }
        val app = runCatching { packageManager.getApplicationLabel(packageManager.getApplicationInfo(sbn.packageName, 0)).toString() }
            .getOrDefault(sbn.packageName)
        worker.execute { check(app, sender, text) }
    }

    private fun check(app: String, sender: String, text: String) {
        if (!ScamDetector.isAvailable(this)) return
        val t0 = System.nanoTime()
        val p = runCatching { ScamDetector.probability(this, text) }.getOrElse {
            Log.e("openjev", "scam check failed: ${it.message}")
            return
        }
        val ms = (System.nanoTime() - t0) / 1_000_000
        val alert = p >= threshold(this)
        Log.i("openjev", "scam app=$app chars=${text.length} p=${"%.3f".format(Locale.ROOT, p)} ms=$ms alert=$alert")
        AlertStore.add(this, AlertStore.Entry(System.currentTimeMillis(), app, sender, text, p, alert, ms))
        if (alert) postAlert(app, sender, text, p)
    }

    private fun postAlert(app: String, sender: String, text: String, p: Double) {
        val manager = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26) {
            manager.createNotificationChannel(NotificationChannel(CHANNEL, "Alertas de golpe", NotificationManager.IMPORTANCE_HIGH).apply {
                description = "Avisa quando uma mensagem recebida parece golpe"
            })
        }
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java).putExtra("detector", true),
                                             PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val pct = String.format(Locale.ROOT, "%.0f%%", p * 100)
        val builder = if (Build.VERSION.SDK_INT >= 26) Notification.Builder(this, CHANNEL) else @Suppress("DEPRECATION") Notification.Builder(this)
        val notification = builder
            .setSmallIcon(android.R.drawable.stat_sys_warning)
            .setContentTitle("⚠️ Possível golpe ($pct)")
            .setContentText("$app · $sender: $text")
            .setStyle(Notification.BigTextStyle().bigText("$app · $sender\n\n$text\n\nNão clique em links, não envie códigos, senhas ou dinheiro sem confirmar por outro canal."))
            .setContentIntent(open)
            .setAutoCancel(true)
            .setCategory(Notification.CATEGORY_MESSAGE)
            .build()
        runCatching { manager.notify((app + text).hashCode(), notification) }
    }

    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
    }
}
