package com.openjev.mobile.engine

/** JNI entry points in src/main/cpp/openjev.cpp. Not thread-safe: use one caller at a time. */
object Native {
    init {
        System.loadLibrary("openjev")
    }

    external fun init(nativeLibDir: String)
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
}
