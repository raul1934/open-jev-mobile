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

/** Times one fixed prompt with each thread count and keeps the fastest. */
class Optimizer(private val handle: Long) {
    data class Trial(val threads: Int, val seconds: Double)

    fun run(tokens: IntArray, candidates: List<Int>, step: (done: Int, text: String) -> Unit): List<Trial> {
        step(0, "Aquecendo o modelo (carregando os pesos na memória)")
        Native.hiddenState(handle, tokens)
        return candidates.mapIndexed { i, n ->
            step(i, "Testando $n threads")
            Native.setThreads(handle, n)
            // Best of two, so a background hiccup does not decide the result.
            val seconds = (0 until 2).minOf {
                val t0 = System.nanoTime()
                Native.hiddenState(handle, tokens)
                (System.nanoTime() - t0) / 1e9
            }
            Trial(n, seconds)
        }
    }
}
