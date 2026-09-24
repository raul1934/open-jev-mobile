package com.openjev.mobile.detector

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Last messages checked by the notification listener, kept only on this device (app storage). */
object AlertStore {
    data class Entry(val time: Long, val app: String, val sender: String, val text: String,
                     val probability: Double, val alerted: Boolean, val millis: Long)

    private const val MAX = 50

    private fun file(context: Context) = File(context.filesDir, "detector-history.json")

    @Synchronized
    fun add(context: Context, entry: Entry) {
        val list = load(context).toMutableList()
        list.add(0, entry)
        val array = JSONArray()
        list.take(MAX).forEach {
            array.put(JSONObject().put("time", it.time).put("app", it.app).put("sender", it.sender).put("text", it.text)
                .put("p", it.probability).put("alerted", it.alerted).put("ms", it.millis))
        }
        file(context).writeText(array.toString())
    }

    @Synchronized
    fun load(context: Context): List<Entry> = runCatching {
        val array = JSONArray(file(context).readText())
        (0 until array.length()).map { i ->
            val o = array.getJSONObject(i)
            Entry(o.getLong("time"), o.getString("app"), o.getString("sender"), o.getString("text"),
                  o.getDouble("p"), o.getBoolean("alerted"), o.optLong("ms"))
        }
    }.getOrDefault(emptyList())

    @Synchronized
    fun clear(context: Context) { file(context).delete() }
}
