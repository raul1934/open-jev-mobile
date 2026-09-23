package com.openjev.mobile.jev

import java.math.BigDecimal
import java.math.BigInteger

/**
 * Minimal JSON model that behaves like Python's `json` module where Open-Jev
 * depends on it: objects keep insertion order, integers stay exact, floats
 * are doubles, and [dumps] reproduces `json.dumps(value, ensure_ascii=False,
 * sort_keys=True)` byte for byte (the text the model is trained on).
 */
sealed interface JsonValue

data class JsonString(val value: String) : JsonValue
data class JsonInt(val value: BigInteger) : JsonValue
data class JsonFloat(val value: Double) : JsonValue
data class JsonBool(val value: Boolean) : JsonValue
object JsonNull : JsonValue
data class JsonArray(val items: List<JsonValue>) : JsonValue
data class JsonObject(val fields: LinkedHashMap<String, JsonValue>) : JsonValue {
    operator fun get(key: String): JsonValue? = fields[key]
}

class JsonError(message: String) : IllegalArgumentException(message)

object Json {
    /** Strict parse, like `jev.server.strict_json`: duplicate keys and NaN/Infinity are rejected. */
    fun parse(text: String): JsonValue = Parser(text).parseDocument()

    /** `json.dumps(value, ensure_ascii=False, sort_keys=sortKeys, allow_nan=False)`. */
    fun dumps(value: JsonValue, sortKeys: Boolean = true): String =
        StringBuilder().also { write(it, value, sortKeys, null, 0) }.toString()

    /** Pretty output for the UI (Python `indent=2`, insertion order). */
    fun pretty(value: JsonValue): String =
        StringBuilder().also { write(it, value, false, "  ", 0) }.toString()

    private fun write(out: StringBuilder, value: JsonValue, sortKeys: Boolean, indent: String?, depth: Int) {
        when (value) {
            is JsonString -> writeString(out, value.value)
            is JsonInt -> out.append(value.value.toString())
            is JsonFloat -> out.append(pythonFloatRepr(value.value))
            is JsonBool -> out.append(if (value.value) "true" else "false")
            JsonNull -> out.append("null")
            is JsonArray -> {
                if (value.items.isEmpty()) { out.append("[]"); return }
                out.append('[')
                value.items.forEachIndexed { i, item ->
                    if (i > 0) out.append(if (indent == null) ", " else ",")
                    newline(out, indent, depth + 1)
                    write(out, item, sortKeys, indent, depth + 1)
                }
                newline(out, indent, depth)
                out.append(']')
            }
            is JsonObject -> {
                if (value.fields.isEmpty()) { out.append("{}"); return }
                val keys = if (sortKeys) value.fields.keys.sortedWith(::compareCodePoints) else value.fields.keys.toList()
                out.append('{')
                keys.forEachIndexed { i, key ->
                    if (i > 0) out.append(if (indent == null) ", " else ",")
                    newline(out, indent, depth + 1)
                    writeString(out, key)
                    out.append(": ")
                    write(out, value.fields.getValue(key), sortKeys, indent, depth + 1)
                }
                newline(out, indent, depth)
                out.append('}')
            }
        }
    }

    private fun newline(out: StringBuilder, indent: String?, depth: Int) {
        if (indent != null) { out.append('\n'); repeat(depth) { out.append(indent) } }
    }

    /** Python sorts str keys by code point; Kotlin's compareTo uses UTF-16 units. */
    private fun compareCodePoints(a: String, b: String): Int {
        var i = 0
        var j = 0
        while (i < a.length && j < b.length) {
            val ca = a.codePointAt(i)
            val cb = b.codePointAt(j)
            if (ca != cb) return ca.compareTo(cb)
            i += Character.charCount(ca)
            j += Character.charCount(cb)
        }
        return (a.length - i).compareTo(b.length - j)
    }

    /** json.encoder.py_encode_basestring (ensure_ascii=False). */
    private fun writeString(out: StringBuilder, s: String) {
        out.append('"')
        for (c in s) {
            when (c) {
                '\\' -> out.append("\\\\")
                '"' -> out.append("\\\"")
                '\n' -> out.append("\\n")
                '\r' -> out.append("\\r")
                '\t' -> out.append("\\t")
                '\b' -> out.append("\\b")
                '\u000C' -> out.append("\\f")
                else -> if (c < ' ') out.append("\\u%04x".format(c.code)) else out.append(c)
            }
        }
        out.append('"')
    }

    /** Python `repr(float)`: shortest round-trip digits, fixed notation for 1e-4 <= |x| < 1e16. */
    fun pythonFloatRepr(value: Double): String {
        if (value.isNaN() || value.isInfinite()) throw JsonError("Out of range float values are not JSON compliant")
        if (value == 0.0) return if (1.0 / value < 0) "-0.0" else "0.0"
        // Shortest decimal that parses back to the same double.
        var digits = ""
        var exponent = 0
        for (precision in 1..17) {
            val text = "%.${precision - 1}e".format(java.util.Locale.ROOT, value)
            if (text.toDouble() == value) {
                val mantissa = text.substringBefore('e').replace("-", "").replace(".", "")
                digits = mantissa.trimEnd('0').ifEmpty { "0" }
                exponent = text.substringAfter('e').toInt()
                break
            }
        }
        val sign = if (value < 0) "-" else ""
        return if (exponent < -4 || exponent >= 16) {
            val mantissa = if (digits.length == 1) digits else digits[0] + "." + digits.substring(1)
            val expSign = if (exponent < 0) "-" else "+"
            sign + mantissa + "e" + expSign + Math.abs(exponent).toString().padStart(2, '0')
        } else {
            val text = BigDecimal(BigInteger(digits), digits.length - 1 - exponent).toPlainString()
            sign + if (text.contains('.')) text else "$text.0"
        }
    }

    private class Parser(private val s: String) {
        private var i = 0

        fun parseDocument(): JsonValue {
            skipSpace()
            val value = parseValue()
            skipSpace()
            if (i != s.length) fail("Extra data")
            return value
        }

        private fun fail(message: String): Nothing = throw JsonError("$message at character $i")

        private fun skipSpace() {
            while (i < s.length && s[i] in " \t\n\r") i++
        }

        private fun parseValue(): JsonValue {
            if (i >= s.length) fail("Expecting value")
            return when (val c = s[i]) {
                '{' -> parseObject()
                '[' -> parseArray()
                '"' -> JsonString(parseString())
                't' -> literal("true", JsonBool(true))
                'f' -> literal("false", JsonBool(false))
                'n' -> literal("null", JsonNull)
                'N', 'I' -> fail("non-finite JSON number")
                else -> if (c == '-' || c in '0'..'9') parseNumber() else fail("Expecting value")
            }
        }

        private fun literal(word: String, value: JsonValue): JsonValue {
            if (!s.startsWith(word, i)) fail("Expecting value")
            i += word.length
            return value
        }

        private fun parseObject(): JsonObject {
            i++
            val fields = LinkedHashMap<String, JsonValue>()
            skipSpace()
            if (i < s.length && s[i] == '}') { i++; return JsonObject(fields) }
            while (true) {
                skipSpace()
                if (i >= s.length || s[i] != '"') fail("Expecting property name enclosed in double quotes")
                val key = parseString()
                skipSpace()
                if (i >= s.length || s[i] != ':') fail("Expecting ':' delimiter")
                i++
                skipSpace()
                if (fields.containsKey(key)) throw JsonError("duplicate JSON key")
                fields[key] = parseValue()
                skipSpace()
                if (i < s.length && s[i] == ',') { i++; continue }
                if (i < s.length && s[i] == '}') { i++; return JsonObject(fields) }
                fail("Expecting ',' delimiter")
            }
        }

        private fun parseArray(): JsonArray {
            i++
            val items = ArrayList<JsonValue>()
            skipSpace()
            if (i < s.length && s[i] == ']') { i++; return JsonArray(items) }
            while (true) {
                skipSpace()
                items.add(parseValue())
                skipSpace()
                if (i < s.length && s[i] == ',') { i++; continue }
                if (i < s.length && s[i] == ']') { i++; return JsonArray(items) }
                fail("Expecting ',' delimiter")
            }
        }

        private fun parseString(): String {
            i++
            val out = StringBuilder()
            while (true) {
                if (i >= s.length) fail("Unterminated string")
                val c = s[i++]
                when {
                    c == '"' -> return out.toString()
                    c == '\\' -> {
                        if (i >= s.length) fail("Unterminated string")
                        when (val e = s[i++]) {
                            '"' -> out.append('"'); '\\' -> out.append('\\'); '/' -> out.append('/')
                            'b' -> out.append('\b'); 'f' -> out.append('\u000C'); 'n' -> out.append('\n')
                            'r' -> out.append('\r'); 't' -> out.append('\t')
                            'u' -> {
                                if (i + 4 > s.length) fail("Invalid \\uXXXX escape")
                                out.append(s.substring(i, i + 4).toIntOrNull(16)?.toChar() ?: fail("Invalid \\uXXXX escape"))
                                i += 4
                            }
                            else -> fail("Invalid \\escape: $e")
                        }
                    }
                    c < ' ' -> fail("Invalid control character")
                    else -> out.append(c)
                }
            }
        }

        private fun parseNumber(): JsonValue {
            val start = i
            if (s[i] == '-') i++
            if (i >= s.length) fail("Expecting value")
            if (s[i] == '0') i++ else if (s[i] in '1'..'9') { while (i < s.length && s[i].isDigit()) i++ } else fail("Expecting value")
            var isFloat = false
            if (i < s.length && s[i] == '.') {
                isFloat = true; i++
                if (i >= s.length || !s[i].isDigit()) fail("Expecting value")
                while (i < s.length && s[i].isDigit()) i++
            }
            if (i < s.length && (s[i] == 'e' || s[i] == 'E')) {
                isFloat = true; i++
                if (i < s.length && (s[i] == '+' || s[i] == '-')) i++
                if (i >= s.length || !s[i].isDigit()) fail("Expecting value")
                while (i < s.length && s[i].isDigit()) i++
            }
            val text = s.substring(start, i)
            return if (isFloat) JsonFloat(text.toDouble()) else JsonInt(BigInteger(text))
        }
    }
}
