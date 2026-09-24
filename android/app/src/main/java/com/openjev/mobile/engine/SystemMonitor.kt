package com.openjev.mobile.engine

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.system.Os
import android.system.OsConstants
import java.io.File

/**
 * CPU, GPU and temperature readings for the status panel. Android restricts many of these
 * files for apps (it varies by manufacturer), so every reading is optional: the UI shows
 * what this device allows. The battery temperature and the thermal status always work.
 */
class SystemMonitor(private val context: Context) {
    data class Snapshot(
        val appCores: Double?,          // CPU used by the app, in cores (8.0 = all 8 busy)
        val cores: Int,
        val coreMHz: List<Int>,         // current frequency per core, if readable
        val cpuTempC: Double?,
        val gpuTempC: Double?,
        val skinTempC: Double?,
        val batteryTempC: Double?,
        val gpuBusyPercent: Int?,
        val gpuMHz: Int?,
        val thermalStatus: Int?,        // PowerManager.THERMAL_STATUS_*
        val thermalHeadroom: Float?,    // 1.0 = at the throttling threshold
    )

    private val cores = Runtime.getRuntime().availableProcessors()
    private val ticksPerSecond = runCatching { Os.sysconf(OsConstants._SC_CLK_TCK) }.getOrDefault(100L).toDouble()
    private val power = context.getSystemService(PowerManager::class.java)
    private var lastCpuTicks = -1L
    private var lastWallMs = 0L
    private var lastHeadroomMs = 0L
    private var lastHeadroom: Float? = null
    private val zones: List<Pair<String, File>> = File("/sys/class/thermal").listFiles().orEmpty()
        .filter { it.name.startsWith("thermal_zone") }
        .mapNotNull { z -> runCatching { File(z, "type").readText().trim() to File(z, "temp") }.getOrNull() }

    private fun readLong(path: String) = runCatching { File(path).readText().trim().split(" ")[0].toLong() }.getOrNull()

    private fun appCpuTicks(): Long? = runCatching {
        // Fields after the ")" of the command name: utime is the 12th, stime the 13th.
        val fields = File("/proc/self/stat").readText().substringAfterLast(')').trim().split(' ')
        fields[11].toLong() + fields[12].toLong()
    }.getOrNull()

    private fun zoneMaxC(match: (String) -> Boolean): Double? =
        zones.filter { match(it.first.lowercase()) }
            .mapNotNull { (_, f) -> runCatching { f.readText().trim().toLong() }.getOrNull() }
            .filter { it in 1..150_000 }
            .maxOrNull()?.let { it / 1000.0 }

    fun snapshot(): Snapshot {
        val now = SystemClock.elapsedRealtime()
        val ticks = appCpuTicks()
        var appCores: Double? = null
        if (ticks != null && lastCpuTicks >= 0 && now > lastWallMs) {
            appCores = (ticks - lastCpuTicks) / ticksPerSecond / ((now - lastWallMs) / 1000.0)
        }
        if (ticks != null) { lastCpuTicks = ticks; lastWallMs = now }

        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            ?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            ?.takeIf { it != Int.MIN_VALUE }?.let { it / 10.0 }

        // getThermalHeadroom is rate-limited by the system; ask at most every 2 s.
        if (Build.VERSION.SDK_INT >= 30 && now - lastHeadroomMs >= 2000) {
            lastHeadroom = runCatching { power.getThermalHeadroom(10) }.getOrNull()?.takeIf { !it.isNaN() }
            lastHeadroomMs = now
        }

        return Snapshot(
            appCores = appCores?.coerceIn(0.0, cores.toDouble()),
            cores = cores,
            coreMHz = (0 until cores).mapNotNull { readLong("/sys/devices/system/cpu/cpu$it/cpufreq/scaling_cur_freq") }
                .map { (it / 1000).toInt() },
            cpuTempC = zoneMaxC { it.startsWith("cpu") },
            gpuTempC = zoneMaxC { it.startsWith("gpu") },
            skinTempC = zoneMaxC { "skin" in it && !it.startsWith("modem") },
            batteryTempC = battery,
            gpuBusyPercent = readLong("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage")?.toInt(),
            gpuMHz = readLong("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq")?.let { (it / 1_000_000).toInt() },
            thermalStatus = if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else null,
            thermalHeadroom = lastHeadroom,
        )
    }
}
