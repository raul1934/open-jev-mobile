package com.openjev.mobile.detector

import android.content.Context
import com.openjev.mobile.engine.Native
import com.openjev.mobile.jev.Json
import com.openjev.mobile.jev.JsonArray
import com.openjev.mobile.jev.JsonFloat
import com.openjev.mobile.jev.JsonInt
import com.openjev.mobile.jev.JsonObject
import com.openjev.mobile.jev.JsonString
import java.io.File
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Scam-message detector: multilingual-e5-small embedding (llama.cpp) followed by the
 * logistic-regression head trained by fraud/train_e5.py (assets/detector/scam-head.json).
 * One instance per process, shared by the screen and the notification listener.
 */
object ScamDetector {
    const val ENCODER_FILE = "e5-small-Q8_0.gguf"

    class Head(val weight: DoubleArray, val bias: Double, val prefix: String, val defaultThreshold: Double)

    private var head: Head? = null
    private var handle = 0L

    fun encoderFile(context: Context) =
        File(File(context.getExternalFilesDir(null) ?: context.filesDir, "models"), ENCODER_FILE)

    fun isAvailable(context: Context) = encoderFile(context).isFile

    @Synchronized
    private fun ensureLoaded(context: Context) {
        if (handle != 0L) return
        Native.ensureInit(context.applicationInfo.nativeLibraryDir)
        val json = Json.parse(context.assets.open("detector/scam-head.json").bufferedReader().use { it.readText() }) as JsonObject
        fun num(v: Any?) = when (v) { is JsonFloat -> v.value; is JsonInt -> v.value.toDouble(); else -> error("bad number") }
        val encoder = json["encoder"] as JsonObject
        head = Head(
            (json["weight"] as JsonArray).items.map(::num).toDoubleArray(),
            num(json["bias"]),
            (encoder["prefix"] as JsonString).value,
            num(json["threshold"]),
        )
        val file = encoderFile(context)
        require(file.isFile) { "falta o modelo do detector ($ENCODER_FILE)" }
        handle = Native.loadEncoder(file.absolutePath, 4)
    }

    fun defaultThreshold(context: Context): Double { ensureLoaded(context); return head!!.defaultThreshold }

    /** Probability (0..1) that [text] is a scam. Thread-safe; takes a few tens of ms. */
    @Synchronized
    fun probability(context: Context, text: String): Double {
        ensureLoaded(context)
        val h = head!!
        val v = Native.embed(handle, h.prefix + text)
        var norm = 0.0
        for (x in v) norm += x.toDouble() * x
        norm = sqrt(norm).coerceAtLeast(1e-12)
        var z = h.bias
        for (i in v.indices) z += h.weight[i] * (v[i] / norm)
        return 1.0 / (1.0 + exp(-z))
    }
}
