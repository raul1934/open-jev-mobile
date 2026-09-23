package com.openjev.mobile.jev

import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Test

/**
 * The Kotlin port must match Open-Jev's Python exactly. parity.json is made by
 * android/tools/make_fixtures.py from the unmodified upstream modules: exact
 * candidate prompts, typed answers for deterministic fake logits, and the
 * error message for each invalid request.
 */
class ParityTest {
    private val fixture = Json.parse(javaClass.classLoader!!.getResource("parity.json")!!.readText()) as JsonObject
    private val temperature = (fixture["temperature"] as JsonFloat).value

    /** Same deterministic logit as make_fixtures.fake_logits. */
    private fun fakeLogit(prompt: String): Double {
        val digest = MessageDigest.getInstance("SHA-256").digest(prompt.toByteArray())
        val top = java.math.BigInteger(1, digest.copyOfRange(0, 8))
        return top.toDouble() / Math.pow(2.0, 64.0) * 8 - 4
    }

    private fun number(v: JsonValue?): Double = when (v) {
        is JsonFloat -> v.value
        is JsonInt -> v.value.toDouble()
        else -> error("not a number: $v")
    }

    @Test
    fun promptsAndAnswersMatchPython() {
        val cases = (fixture["cases"] as JsonArray).items.map { it as JsonObject }
        for (case in cases) {
            val name = (case["name"] as JsonString).value
            val request = case["request"] as JsonObject
            val records = Api.compileRequest(request["state"]!!, request["questions"])
            val prompts = records.flatMap { Api.candidatePrompts(it) }
            val expected = (case["prompts"] as JsonArray).items.map { (it as JsonString).value }
            assertEquals("prompt count for $name", expected.size, prompts.size)
            prompts.zip(expected).forEachIndexed { i, (got, want) -> assertEquals("$name prompt $i", want, got) }

            val answers = case["answers"] as JsonObject
            for (record in records) {
                val scores = Api.candidatePrompts(record).map(::fakeLogit)
                val answer = Api.answer(record, Api.softmax(Api.logits(record, scores), temperature))
                val want = answers[record.id] as JsonObject
                val tag = "$name/${record.id}"
                when (answer) {
                    is NoulAnswer -> assertEquals(tag, number(want["noul"]), answer.noul, 1e-12)
                    is ChoiceAnswer -> {
                        assertEquals(tag, (want["choice"] as JsonString).value, answer.choice)
                        assertEquals(tag, number(want["confidence"]), answer.confidence, 1e-12)
                    }
                    is ScoreAnswer -> {
                        assertEquals(tag, number(want["score"]), answer.score, 1e-12)
                        assertEquals(tag, number(want["confidence"]), answer.confidence, 1e-12)
                        assertEquals(tag, Json.dumps(want["legend"]!!), Json.dumps(answer.legend))
                    }
                }
                if (answer !is NoulAnswer) {
                    val probs = want["probabilities"] as JsonObject
                    assertEquals(tag, probs.fields.keys.toList(), answer.probabilities.map { it.first })
                    answer.probabilities.forEach { (k, p) -> assertEquals("$tag/$k", number(probs[k]), p, 1e-12) }
                }
            }
        }
    }

    @Test
    fun invalidRequestsFailLikePython() {
        for (item in (fixture["errors"] as JsonArray).items.map { it as JsonObject }) {
            val request = item["request"] as JsonObject
            val want = (item["error"] as JsonString).value
            try {
                Api.compileRequest(request["state"]!!, request["questions"])
                fail("expected an error: $want")
            } catch (e: IllegalArgumentException) {
                assertEquals(want, e.message)
            }
        }
    }

    @Test
    fun floatReprMatchesPython() {
        val cases = mapOf(1.0 to "1.0", 0.1 to "0.1", 1e-05 to "1e-05", 1e16 to "1e+16", 1.5e-7 to "1.5e-07",
                          123456.789 to "123456.789", 1e15 to "1000000000000000.0", -0.0 to "-0.0",
                          0.0001 to "0.0001", 2.5e300 to "2.5e+300", 1.7976931348623157e308 to "1.7976931348623157e+308")
        cases.forEach { (value, text) -> assertEquals(text, Json.pythonFloatRepr(value)) }
    }

    @Test
    fun strictParsing() {
        for (bad in listOf("""{"a": 1, "a": 2}""", """{"a": NaN}""", """[1,]""", """{"a": Infinity}""")) {
            try { Json.parse(bad); fail("should reject $bad") } catch (_: JsonError) {}
        }
    }
}
