package com.openjev.mobile.engine

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import java.io.File
import java.security.MessageDigest

/** One file of the model package published as a GitHub release asset. */
data class ModelFile(val name: String, val bytes: Long, val sha256: String)

/**
 * Downloads the converted Open-Jev-2B package from the repository's GitHub
 * release (made by convert/build.py) into app storage, and checks its SHA-256.
 * DownloadManager does the transfer, so it survives the app going to the
 * background and resumes after network drops.
 */
class ModelStore(private val context: Context) {
    companion object {
        const val RELEASE_URL = "https://github.com/raul1934/open-jev-mobile/releases/download/model-v1/"
        val HEAD = ModelFile("head.json", 46_476, "8b9282bf0f5fcae0e55f91778c067bbc2e20f1b3bba6dc82dd5179644297a5b9")
        val GGUF = ModelFile("open-jev-2b-Q5_K_M.gguf", 1_411_120_608,
                             "5140cc4c3ef79c4097880694999c60fffeda98289d72ca7660115abf5795aa66")
        val FILES = listOf(HEAD, GGUF)
        private const val PREFS = "model-store"
    }

    val dir: File = File(context.getExternalFilesDir(null) ?: context.filesDir, "models").apply { mkdirs() }
    private val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val downloads = context.getSystemService(DownloadManager::class.java)

    fun file(model: ModelFile) = File(dir, model.name)

    /** True when both files are present and were verified once (hashing 1.4 GB takes a while). */
    fun isReady(): Boolean = FILES.all { f ->
        val file = file(f)
        file.isFile && file.length() == f.bytes && prefs.getString("verified:${f.name}", null) == f.sha256
    }

    fun hasUnverifiedFiles(): Boolean = FILES.all { file(it).isFile && file(it).length() == it.bytes }

    fun startDownload() {
        for (f in FILES) {
            val old = prefs.getLong("download:${f.name}", -1)
            if (old >= 0) downloads.remove(old)
            file(f).delete()
            prefs.edit().remove("verified:${f.name}").apply()
            val request = DownloadManager.Request(Uri.parse(RELEASE_URL + f.name))
                .setTitle("Open-Jev: ${f.name}")
                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                .setAllowedOverMetered(true)
                .setDestinationUri(Uri.fromFile(file(f)))
            prefs.edit().putLong("download:${f.name}", downloads.enqueue(request)).apply()
        }
    }

    sealed interface Progress {
        data object Idle : Progress
        data class Running(val bytes: Long, val total: Long) : Progress
        data object Done : Progress
        data class Failed(val reason: String) : Progress
    }

    /** Aggregated state of the enqueued downloads. */
    fun progress(): Progress {
        var bytes = 0L
        var anyRunning = false
        for (f in FILES) {
            val id = prefs.getLong("download:${f.name}", -1)
            if (id < 0) return if (isReady()) Progress.Done else Progress.Idle
            downloads.query(DownloadManager.Query().setFilterById(id)).use { c ->
                if (!c.moveToFirst()) return Progress.Failed("download canceled")
                val status = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                bytes += c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                when (status) {
                    DownloadManager.STATUS_FAILED -> {
                        val reason = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON))
                        return Progress.Failed(if (reason == 404) "arquivo não encontrado no release (HTTP 404)" else "erro $reason")
                    }
                    DownloadManager.STATUS_SUCCESSFUL -> Unit
                    else -> anyRunning = true
                }
            }
        }
        return if (anyRunning) Progress.Running(bytes, FILES.sumOf { it.bytes }) else Progress.Done
    }

    /** Checks sizes and SHA-256 after a download or import; deletes files that do not match. */
    fun verify(onProgress: (Float) -> Unit = {}): String? {
        val total = FILES.sumOf { it.bytes }.toFloat()
        var done = 0L
        for (f in FILES) {
            val file = file(f)
            if (!file.isFile) return "falta ${f.name}"
            val length = file.length()
            if (length != f.bytes) {
                file.delete()
                return "${f.name} tem $length bytes, esperado ${f.bytes}"
            }
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buffer)
                    if (n < 0) break
                    digest.update(buffer, 0, n)
                    done += n
                    onProgress(done / total)
                }
            }
            val hex = digest.digest().joinToString("") { "%02x".format(it) }
            if (hex != f.sha256) {
                file.delete()
                return "${f.name} está corrompido (SHA-256 diferente); baixe de novo"
            }
            prefs.edit().putString("verified:${f.name}", f.sha256).remove("download:${f.name}").apply()
        }
        return null
    }

    /** Copies a file picked by the user (e.g. from Downloads) into app storage. */
    fun import(uri: Uri): String {
        val name = context.contentResolver.query(uri, null, null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(c.getColumnIndexOrThrow(android.provider.OpenableColumns.DISPLAY_NAME)) else null
        } ?: ""
        val target = when {
            name.endsWith(".gguf") -> GGUF
            name.endsWith(".json") -> HEAD
            else -> throw IllegalArgumentException("escolha o arquivo .gguf ou o head.json")
        }
        context.contentResolver.openInputStream(uri)!!.use { input ->
            file(target).outputStream().use { input.copyTo(it, 1 shl 20) }
        }
        prefs.edit().remove("verified:${target.name}").remove("download:${target.name}").apply()
        return target.name
    }
}
