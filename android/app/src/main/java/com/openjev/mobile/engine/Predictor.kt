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

/** Where the time of one prediction went: stage -> (seconds, tokens decoded, calls). */
class Profile {
    data class Entry(var seconds: Double = 0.0, var tokens: Int = 0, var calls: Int = 0)
    val entries = LinkedHashMap<String, Entry>()

    fun <T> time(stage: String, tokens: Int = 0, block: () -> T): T {
        val t0 = System.nanoTime()
        try {
            return block()
        } finally {
            entries.getOrPut(stage) { Entry() }.apply {
                seconds += (System.nanoTime() - t0) / 1e9; this.tokens += tokens; calls++
            }
        }
    }

    fun describe() = entries.entries.joinToString(", ") { (k, e) ->
        "$k=" + String.format(java.util.Locale.ROOT, "%.2f", e.seconds) + "s/${e.tokens}tok/${e.calls}x"
    }
}

data class Prediction(val answers: List<Pair<String, Answer>>, val candidates: Int, val inputTokens: Int, val seconds: Double,
                      val decodedTokens: Int = 0, val profile: Profile = Profile())

/**
 * Same pipeline as jev_mobile + jev.serving.Predictor: compile the request,
 * score every candidate prompt independently with head(h_last), softmax with
 * the calibration temperature, format typed answers.
 */
class Predictor(private val handle: Long, private val head: Head) {
    private val maxLength = minOf(head.maxLength, Native.contextSize(handle))

    /** Thread count for a chunk of n tokens (a measured [ThreadPolicy], or a fixed number). */
    var threadsFor: (Int) -> Int = { 4 }
    private var currentThreads = -1

    private fun useThreadsFor(tokens: Int) {
        val n = threadsFor(tokens)
        if (n != currentThreads) {
            Native.setThreads(handle, n)
            currentThreads = n
        }
    }

    /** Called when something else (the optimizer) changed the native thread count. */
    fun forgetThreads() { currentThreads = -1 }

    private var profile = Profile()

    private fun decodeAll(tokens: IntArray): FloatArray {
        useThreadsFor(tokens.size)
        return profile.time("opções (sem cache)", tokens.size) { Native.hiddenState(handle, tokens) }
    }

    private fun extend(stage: String, tokens: IntArray, start: Int, wantHidden: Boolean): FloatArray? {
        useThreadsFor(tokens.size)
        return profile.time(stage, tokens.size) { Native.extend(handle, tokens, start, wantHidden) }
    }

    private fun save(slot: Int) = profile.time("salvar estado") { Native.saveState(handle, slot) }
    private fun restore(slot: Int) = profile.time("restaurar estado") { Native.restoreState(handle, slot) }
    private fun reset() = profile.time("limpar") { Native.reset(handle) }

    /**
     * With [prefixCache], tokens shared by every prompt of the request (the context) are
     * computed once, tokens shared by one question's candidates once per question, and
     * only each candidate's own tail is computed per candidate. The model state after a
     * shared prefix (attention KV cache and recurrent state) is saved and restored, so
     * every candidate still sees exactly its own full token sequence.
     */
    fun predict(request: JsonValue, prefixCache: Boolean = true,
                onProgress: (done: Int, total: Int, step: String) -> Unit = { _, _, _ -> }): Prediction {
        val start = System.nanoTime()
        profile = Profile()
        require(request is JsonObject && request["state"] != null && request["questions"] != null) {
            "request requires state and questions"
        }
        val records = Api.compileRequest(request["state"]!!, request["questions"])
        val prompts = records.map { Api.candidatePrompts(it) }
        val total = prompts.sumOf { it.size }
        onProgress(0, total, "Preparando os prompts")
        // Tokenize everything first so an over-long prompt fails before any slow work.
        val tokens = profile.time("tokenizar") {
            prompts.map { group -> group.map { Native.tokenize(handle, head.promptPrefix + it + head.promptSuffix) } }
        }
        val longest = tokens.flatten().maxOf { it.size }
        if (longest > maxLength) {
            throw IllegalArgumentException("Input length $longest exceeds max_length=$maxLength; no silent truncation" +
                if (maxLength < head.maxLength) " (raise the context size in settings, up to ${head.maxLength})" else "")
        }
        var done = 0
        val step = { text: String -> onProgress(done, total, text) }
        val tick = { done++; Unit }
        val ids = records.map { it.id }
        val hidden = if (prefixCache) cachedHiddenStates(ids, tokens, step, tick)
                     else tokens.mapIndexed { q, group ->
                         group.mapIndexed { j, t ->
                             step(optionStep(ids[q], j, group.size, t.size))
                             decodeAll(t).also { tick() }
                         }
                     }
        onProgress(done, total, "Calculando as probabilidades")
        val answers = records.mapIndexed { index, record ->
            val scores = hidden[index].map { h ->
                var sum = head.bias
                for (i in h.indices) sum += head.weight[i] * h[i]
                sum
            }
            record.id to Api.answer(record, Api.softmax(Api.logits(record, scores), head.temperature))
        }
        val decoded = profile.entries.values.sumOf { it.tokens }
        return Prediction(answers, total, tokens.flatten().sumOf { it.size }, (System.nanoTime() - start) / 1e9,
                          decoded, profile)
    }

    /** Longest shared token prefix, leaving at least one token per prompt to decode (its output is read). */
    private fun commonPrefix(lists: List<IntArray>): Int {
        var n = lists.minOf { it.size } - 1
        for (ids in lists) {
            var i = 0
            while (i < n && ids[i] == lists[0][i]) i++
            n = i
        }
        return n
    }

    private fun optionStep(id: String, j: Int, count: Int, tokens: Int) =
        if (count == 1) "\"$id\": avaliando ($tokens tokens)" else "\"$id\": opção ${j + 1} de $count ($tokens tokens)"

    private fun cachedHiddenStates(ids: List<String>, tokens: List<List<IntArray>>, step: (String) -> Unit,
                                   tick: () -> Unit): List<List<FloatArray>> {
        val shared = commonPrefix(tokens.flatten())
        reset()
        if (shared > 0) {
            step("Lendo o contexto compartilhado ($shared tokens)")
            extend("contexto", tokens[0][0].copyOfRange(0, shared), 0, false)
            save(0)
        }
        fun restoreShared() = if (shared > 0) restore(0) else reset()
        return tokens.mapIndexed { index, group ->
            val own = maxOf(shared, commonPrefix(group))
            if (index > 0) restoreShared()
            if (own > shared) {
                step("\"${ids[index]}\": lendo a pergunta (${own - shared} tokens)")
                extend("pergunta", group[0].copyOfRange(shared, own), shared, false)
                if (group.size > 1) save(1)
            }
            group.mapIndexed { j, tokens ->
                if (j > 0) if (own > shared) restore(1) else restoreShared()
                step(optionStep(ids[index], j, group.size, tokens.size - own))
                extend("opções", tokens.copyOfRange(own, tokens.size), own, true)!!.also { tick() }
            }
        }
    }
}
