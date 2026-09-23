package com.openjev.mobile.jev

import kotlin.math.abs
import kotlin.math.exp

/**
 * Kotlin port of Open-Jev's `jev/api.py`, `jev/metrics.py` (softmax and
 * confidences) and the probability path of `jev/serving.py`, at commit
 * 3308a15ccd7eea1df7a37d6ddc39b023b801ba16. Error messages match upstream.
 * Parity with Python is enforced by `ParityTest` against fixtures generated
 * from the unmodified upstream modules (`android/tools/make_fixtures.py`).
 */
enum class Kind { CHOICE, SCORE, NOUL }

class Record(
    val id: String,
    val state: JsonValue,
    val kind: Kind,
    val question: String,
    val options: List<String>,
    val answerKeys: List<String>,
    val legend: JsonObject?,
)

sealed interface Answer {
    val probabilities: List<Pair<String, Double>>
}
data class ChoiceAnswer(val choice: String, override val probabilities: List<Pair<String, Double>>, val confidence: Double) : Answer
data class ScoreAnswer(val score: Double, override val probabilities: List<Pair<String, Double>>, val confidence: Double,
                       val legend: JsonObject) : Answer
data class NoulAnswer(val noul: Double, override val probabilities: List<Pair<String, Double>>) : Answer

object Api {
    private fun render(value: JsonValue): String = if (value is JsonString) value.value else Json.dumps(value)

    private fun description(value: JsonValue?, optional: Boolean = false): String? {
        if ((value == null || value == JsonNull) && optional) return null
        if (value !is JsonString && value !is JsonObject && value !is JsonArray) {
            throw IllegalArgumentException("instructions and descriptions must be text, an object, or an array")
        }
        Json.dumps(value) // rejects non-finite numbers, like json.dumps(allow_nan=False)
        return render(value)
    }

    fun compileRequest(state: JsonValue, questions: JsonValue?): List<Record> {
        if (state !is JsonString && state !is JsonObject && state !is JsonArray) {
            throw IllegalArgumentException("state must be text, a JSON object, or an array")
        }
        Json.dumps(state)
        if (questions !is JsonObject || questions.fields.isEmpty()) {
            throw IllegalArgumentException("questions must be a nonempty mapping")
        }
        return questions.fields.map { (questionId, definition) ->
            if (definition !is JsonObject) {
                throw IllegalArgumentException("question IDs must be strings and definitions must be mappings")
            }
            val kind = when ((definition["type"] as? JsonString)?.value) {
                "choice" -> Kind.CHOICE
                "score" -> Kind.SCORE
                "noul" -> Kind.NOUL
                else -> throw IllegalArgumentException("question type must be choice, score, or noul")
            }
            var question = description(definition["instructions"])!!
            val criteria = definition["criteria"]
            when (kind) {
                Kind.CHOICE -> {
                    if (criteria !is JsonObject || criteria.fields.size !in 1..255) {
                        throw IllegalArgumentException("Choice requires between 1 and 255 candidates")
                    }
                    val descriptions = criteria.fields.values.map { description(it, optional = true) }
                    val options = criteria.fields.keys.zip(descriptions).map { (name, text) ->
                        if (text == null) name else "$name: $text"
                    }
                    Record(questionId, state, kind, question, options, criteria.fields.keys.toList(), null)
                }
                Kind.SCORE -> {
                    if (criteria !is JsonArray || criteria.items.size !in 2..10) {
                        throw IllegalArgumentException("Score requires an array of 2 to 10 descriptive levels")
                    }
                    val options = criteria.items.map { description(it)!! }
                    val keys = criteria.items.indices.map { it.toString() }
                    val legend = JsonObject(LinkedHashMap(keys.zip(criteria.items).toMap()))
                    Record(questionId, state, kind, question, options, keys, legend)
                }
                Kind.NOUL -> {
                    if (criteria != null && criteria != JsonNull) {
                        if (criteria !is JsonObject || criteria.fields.keys != setOf("true", "false")) {
                            throw IllegalArgumentException("Noul criteria must contain true and false descriptions")
                        }
                        val yes = description(criteria["true"])
                        val no = description(criteria["false"])
                        question += "\nYes means: $yes\nNo means: $no"
                    }
                    Record(questionId, state, kind, question, listOf("no", "yes"), listOf("false", "true"), null)
                }
            }
        }
    }

    fun candidatePrompts(record: Record): List<String> {
        val prefix = "Context:\n${render(record.state)}\n\nQuestion: ${record.question}\n"
        if (record.kind == Kind.NOUL) return listOf(prefix + "Is the answer to this question yes? Answer Yes or No.")
        return record.options.map {
            prefix + "Proposed answer: $it\nIs this proposed answer correct? Answer Yes or No."
        }
    }

    /** Raw logits for a record from one head score per candidate prompt (jev.model.DecisionModel.forward). */
    fun logits(record: Record, scores: List<Double>): List<Double> =
        if (record.kind == Kind.NOUL) listOf(0.0, scores[0]) else scores

    fun softmax(logits: List<Double>, temperature: Double): List<Double> {
        require(temperature.isFinite() && temperature > 0) { "temperature must be finite and positive" }
        require(logits.isNotEmpty() && logits.none { it.isNaN() || it == Double.POSITIVE_INFINITY }) {
            "logits must be nonempty and contain no NaN or +inf"
        }
        val maximum = logits.max()
        require(maximum != Double.NEGATIVE_INFINITY) { "at least one logit must be finite" }
        val weights = logits.map { exp((it - maximum) / temperature) }
        val total = weights.sum()
        return weights.map { it / total }
    }

    private fun confidenceDistribution(probs: List<Double>): List<Double> {
        val total = probs.sum()
        return if (total != 0.0) probs.map { it / total } else List(probs.size) { 1.0 / probs.size }
    }

    fun choiceConfidence(probs: List<Double>): Double {
        val p = confidenceDistribution(probs)
        if (p.size == 1) return 1.0
        val uniform = 1.0 / p.size
        return (p.max() - uniform) / (1.0 - uniform)
    }

    fun scoreConfidence(probs: List<Double>): Double {
        val p = confidenceDistribution(probs)
        val count = p.size
        if (count == 1) return 1.0
        val mode = p.indices.maxBy { p[it] } // first maximum, like Python's max()
        val distance = p.indices.sumOf { p[it] * abs(it - mode) }
        val center = (count - 1) / 2.0
        val uniformDeviation = (0 until count).sumOf { abs(it - center) } / count
        return maxOf(0.0, 1.0 - distance / uniformDeviation)
    }

    /** jev.api.format_response for one record, given calibrated probabilities. */
    fun answer(record: Record, values: List<Double>): Answer {
        require(values.size == record.answerKeys.size) { "probability count must match the declared answer space" }
        require(values.all { it.isFinite() && it in 0.0..1.0 }) { "probabilities must be finite and in [0, 1]" }
        val total = values.sum()
        require(abs(total - 1.0) <= maxOf(1e-6 * maxOf(abs(total), 1.0), 1e-6)) { "probabilities must sum to one" }
        val probs = values.map { it / total }
        val keyed = record.answerKeys.zip(probs)
        return when (record.kind) {
            Kind.NOUL -> NoulAnswer(probs[1], keyed)
            Kind.CHOICE -> ChoiceAnswer(record.answerKeys[probs.indices.maxBy { probs[it] }], keyed, choiceConfidence(probs))
            Kind.SCORE -> ScoreAnswer(probs.indices.sumOf { it * probs[it] }, keyed, scoreConfidence(probs), record.legend!!)
        }
    }
}
