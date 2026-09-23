package com.openjev.mobile.engine

import com.openjev.mobile.jev.Answer
import com.openjev.mobile.jev.Api
import com.openjev.mobile.jev.Json
import com.openjev.mobile.jev.JsonArray
import com.openjev.mobile.jev.JsonFloat
import com.openjev.mobile.jev.JsonInt
import com.openjev.mobile.jev.JsonObject
import com.openjev.mobile.jev.JsonString
import com.openjev.mobile.jev.JsonValue
import java.io.File

/** The trained decision head and prompt wrapper from convert/build.py's head.json. */
class Head(val weight: DoubleArray, val bias: Double, val temperature: Double, val maxLength: Int,
           val promptPrefix: String, val promptSuffix: String) {
    companion object {
        fun load(file: File): Head {
            val json = Json.parse(file.readText()) as JsonObject
            fun number(key: String) = when (val v = json[key]) {
                is JsonFloat -> v.value
                is JsonInt -> v.value.toDouble()
                else -> throw IllegalArgumentException("head.json: $key is missing")
            }
            fun text(key: String) = (json[key] as? JsonString)?.value ?: throw IllegalArgumentException("head.json: $key is missing")
            val weight = (json["weight"] as JsonArray).items.map {
                when (it) { is JsonFloat -> it.value; is JsonInt -> it.value.toDouble(); else -> Double.NaN }
            }.toDoubleArray()
            require(weight.size == number("hidden_size").toInt() && weight.all { it.isFinite() }) {
                "head.json weight does not match hidden_size or is not finite"
            }
            return Head(weight, number("bias"), number("temperature"), number("max_length").toInt(),
                        text("prompt_prefix"), text("prompt_suffix"))
        }
    }
}

data class Prediction(val answers: List<Pair<String, Answer>>, val candidates: Int, val inputTokens: Int, val seconds: Double)

/**
 * Same pipeline as jev_mobile + jev.serving.Predictor: compile the request,
 * score every candidate prompt independently with head(h_last), softmax with
 * the calibration temperature, format typed answers.
 */
class Predictor(private val handle: Long, private val head: Head) {
    private val maxLength = minOf(head.maxLength, Native.contextSize(handle))

    fun predict(request: JsonValue, onProgress: (done: Int, total: Int) -> Unit = { _, _ -> }): Prediction {
        val start = System.nanoTime()
        require(request is JsonObject && request["state"] != null && request["questions"] != null) {
            "request requires state and questions"
        }
        val records = Api.compileRequest(request["state"]!!, request["questions"])
        val prompts = records.map { Api.candidatePrompts(it) }
        val total = prompts.sumOf { it.size }
        // Tokenize everything first so an over-long prompt fails before any slow work.
        val tokens = prompts.map { group -> group.map { Native.tokenize(handle, head.promptPrefix + it + head.promptSuffix) } }
        val longest = tokens.flatten().maxOf { it.size }
        if (longest > maxLength) {
            throw IllegalArgumentException("Input length $longest exceeds max_length=$maxLength; no silent truncation" +
                if (maxLength < head.maxLength) " (raise the context size in settings, up to ${head.maxLength})" else "")
        }
        var done = 0
        onProgress(0, total)
        val answers = records.mapIndexed { index, record ->
            val scores = tokens[index].map { ids ->
                val hidden = Native.hiddenState(handle, ids)
                var sum = head.bias
                for (i in hidden.indices) sum += head.weight[i] * hidden[i]
                onProgress(++done, total)
                sum
            }
            record.id to Api.answer(record, Api.softmax(Api.logits(record, scores), head.temperature))
        }
        return Prediction(answers, total, tokens.flatten().sumOf { it.size }, (System.nanoTime() - start) / 1e9)
    }
}
