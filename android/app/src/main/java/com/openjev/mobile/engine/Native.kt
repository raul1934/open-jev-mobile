package com.openjev.mobile.engine

/** JNI entry points in src/main/cpp/openjev.cpp. Not thread-safe: use one caller at a time. */
object Native {
    init {
        System.loadLibrary("openjev")
    }

    external fun init(nativeLibDir: String)
    external fun load(path: String, nCtx: Int, nThreads: Int): Long
    external fun contextSize(handle: Long): Int
    external fun tokenize(handle: Long, text: String): IntArray
    external fun hiddenState(handle: Long, tokens: IntArray): FloatArray
    external fun free(handle: Long)
}
