package com.openjev.mobile.engine

import java.io.File

/** CPU layout read from sysfs: clusters of cores that share a maximum frequency, fastest first. */
object DeviceInfo {
    data class Cluster(val maxKHz: Long, val cores: Int)

    fun clusters(): List<Cluster> {
        val freqs = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { cpu ->
            runCatching {
                File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq").readText().trim().toLong()
            }.getOrNull()
        }
        return freqs.groupingBy { it }.eachCount().map { (khz, n) -> Cluster(khz, n) }.sortedByDescending { it.maxKHz }
    }

    /**
     * Thread counts worth trying: the fast cores alone, each larger cluster boundary, 4 and all cores.
     * Big.LITTLE chips are often fastest with only the big cores, since llama.cpp splits work evenly
     * and waits for the slowest thread.
     */
    fun threadCandidates(): List<Int> {
        val total = Runtime.getRuntime().availableProcessors()
        val bounds = clusters().runningFold(0) { acc, c -> acc + c.cores }.drop(1)
        return (bounds + listOf(2, 4, total)).filter { it in 1..total }.distinct().sorted()
    }

    fun describe(): String = clusters().joinToString(" + ") { "${it.cores}×${"%.1f".format(it.maxKHz / 1e6)} GHz" }
}

/**
 * Best thread count per chunk size, measured on this device. A chunk of n tokens uses the
 * entry for the largest measured size <= n (the smallest size below that).
 */
data class ThreadPolicy(val sizes: List<Int>, val threads: List<Int>) {
    fun threadsFor(tokens: Int): Int {
        var t = threads.first()
        for (i in sizes.indices) if (tokens >= sizes[i]) t = threads[i]
        return t
    }

    fun encode() = sizes.zip(threads).joinToString(",") { "${it.first}:${it.second}" }

    fun describe() = sizes.zip(threads).joinToString(" · ") { "${it.first}+ tokens → ${it.second}" }

    companion object {
        fun decode(text: String?): ThreadPolicy? = runCatching {
            val pairs = text!!.split(",").map { it.split(":").let { (s, t) -> s.toInt() to t.toInt() } }
            ThreadPolicy(pairs.map { it.first }, pairs.map { it.second })
        }.getOrNull()
    }
}

/** Times chunks of several sizes with each thread count; the fastest per size becomes the policy. */
class Optimizer(private val handle: Long) {
    data class Trial(val tokens: Int, val threads: Int, val seconds: Double)

    companion object {
        val SIZES = listOf(16, 32, 64, 160)
    }

    fun run(tokens: IntArray, candidates: List<Int>, step: (done: Int, total: Int, text: String) -> Unit): List<Trial> {
        val sizes = SIZES.map { minOf(it, tokens.size) }.distinct()
        val total = sizes.size * candidates.size
        step(0, total, "Aquecendo o modelo (carregando os pesos na memória)")
        Native.hiddenState(handle, tokens)
        var done = 0
        return sizes.flatMap { size ->
            val chunk = tokens.copyOfRange(0, size)
            candidates.map { n ->
                step(done, total, "Testando $size tokens com $n threads")
                Native.setThreads(handle, n)
                // Best of two, so a background hiccup does not decide the result.
                val seconds = (0 until 2).minOf {
                    val t0 = System.nanoTime()
                    Native.hiddenState(handle, chunk)
                    (System.nanoTime() - t0) / 1e9
                }
                done++
                Trial(size, n, seconds)
            }
        }
    }

    fun policy(trials: List<Trial>): ThreadPolicy {
        val bySize = trials.groupBy { it.tokens }.toSortedMap()
        return ThreadPolicy(bySize.keys.toList(), bySize.values.map { list -> list.minBy { it.seconds }.threads })
    }
}
