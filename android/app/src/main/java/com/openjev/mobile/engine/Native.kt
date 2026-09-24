package com.openjev.mobile.engine

/** JNI entry points in src/main/cpp/openjev.cpp. Not thread-safe: use one caller at a time. */
object Native {
    init {
        System.loadLibrary("openjev")
    }

    external fun init(nativeLibDir: String)

    private var initialized = false

    /** Loads the CPU backends once per process (the app screen and the notification listener share it). */
    @Synchronized
    fun ensureInit(nativeLibDir: String) {
        if (!initialized) {
            init(nativeLibDir)
            initialized = true
        }
    }
    external fun load(path: String, nCtx: Int, nThreads: Int): Long
    external fun contextSize(handle: Long): Int
    external fun setThreads(handle: Long, threads: Int)
    external fun tokenize(handle: Long, text: String): IntArray
    external fun hiddenState(handle: Long, tokens: IntArray): FloatArray

    // Prefix cache: continue seq 0 at position `start`; save/restore it in slot 0 or 1.
    external fun extend(handle: Long, tokens: IntArray, start: Int, wantHidden: Boolean): FloatArray?
    external fun reset(handle: Long)
    external fun saveState(handle: Long, slot: Int)
    external fun restoreState(handle: Long, slot: Int)
    external fun free(handle: Long)

    // Scam detector encoder (multilingual-e5-small): mean-pooled, not normalized.
    external fun loadEncoder(path: String, nThreads: Int): Long
    external fun embed(handle: Long, text: String): FloatArray
}
