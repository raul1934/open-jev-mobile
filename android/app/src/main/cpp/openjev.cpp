// JNI bridge: tokenize a prompt and return the final hidden state of its last token.
//
// Returns the same vector as `llama-server --embeddings --pooling last` (the
// path jev_mobile/scorer.py uses and convert/verify.py measured): the output of
// the final RMS norm ("result_norm") for the last prompt token, with special
// tokens parsed, no BOS added, memory cleared between prompts, no L2 norm.
//
// It is read with an eval callback instead of embeddings mode because in
// llama.cpp v0.4.1 embeddings mode makes every token an output and reserves a
// logits buffer for all of them (248k vocabulary x n_ctx floats, ~2 GB at
// n_ctx = 2048). Here only the last token is an output, so the model head runs
// for one row and the compute buffer stays around 200 MB.
#include <android/log.h>
#include <jni.h>

#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

#define TAG "openjev"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

struct Engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_embd = 0;
    int n_ctx = 0;
    std::vector<float> hidden;  // filled by capture_hidden during llama_decode
    bool captured = false;
    std::vector<uint8_t> saved[2];  // prefix-cache slots (KV cache + recurrent state of seq 0)
};

// ggml eval callback: ask == true -> "do you want this tensor?"; ask == false -> it was computed.
bool capture_hidden(ggml_tensor * t, bool ask, void * user_data) {
    if (std::strcmp(t->name, "result_norm") != 0) return !ask;
    if (ask) return true;
    auto * e = static_cast<Engine *>(user_data);
    if (t->type != GGML_TYPE_F32 || t->ne[0] != e->n_embd) return true;
    // One row per output token; the last row is the last prompt token.
    const int64_t rows = ggml_nrows(t);
    if (rows == 0) return true;  // a prefix-only decode has no outputs
    e->hidden.resize(e->n_embd);
    ggml_backend_tensor_get(t, e->hidden.data(), (rows - 1) * t->nb[1], e->n_embd * sizeof(float));
    e->captured = true;
    return true;
}

void log_callback(ggml_log_level level, const char * text, void *) {
    __android_log_print(level >= GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, TAG, "%s", text);
}

void throw_java(JNIEnv * env, const std::string & message) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message.c_str());
}

std::string to_string(JNIEnv * env, jstring value) {
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

Engine * engine(jlong handle) { return reinterpret_cast<Engine *>(handle); }

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_openjev_mobile_engine_Native_init(JNIEnv * env, jobject, jstring native_lib_dir) {
    llama_log_set(log_callback, nullptr);
    // Loads libggml-cpu-*.so and picks the best variant for this CPU.
    ggml_backend_load_all_from_path(to_string(env, native_lib_dir).c_str());
    llama_backend_init();
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_openjev_mobile_engine_Native_load(JNIEnv * env, jobject, jstring path, jint n_ctx, jint n_threads) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP;  // weights stay in the file cache and can be evicted under memory pressure
    llama_model * model = llama_model_load_from_file(to_string(env, path).c_str(), mparams);
    if (!model) {
        throw_java(env, "could not load the model file");
        return 0;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;   // a whole prompt must fit in one micro-batch
    cparams.n_ubatch = n_ctx;
    cparams.n_seq_max = 1;
    cparams.n_outputs_max = 1;  // only the last token's hidden state is read
    cparams.n_outputs_max_per_seq = 1;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings = false;
    auto * e = new Engine();
    e->n_embd = llama_model_n_embd_out(model);
    cparams.cb_eval = capture_hidden;
    cparams.cb_eval_user_data = e;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        delete e;
        llama_model_free(model);
        throw_java(env, "could not create the inference context (not enough memory?)");
        return 0;
    }
    e->model = model;
    e->ctx = ctx;
    e->vocab = llama_model_get_vocab(model);
    e->n_ctx = static_cast<int>(llama_n_ctx(ctx));
    LOGI("model loaded: n_embd=%d n_ctx=%d threads=%d", e->n_embd, e->n_ctx, n_threads);
    return reinterpret_cast<jlong>(e);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_openjev_mobile_engine_Native_contextSize(JNIEnv *, jobject, jlong handle) {
    return engine(handle)->n_ctx;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_openjev_mobile_engine_Native_tokenize(JNIEnv * env, jobject, jlong handle, jstring jtext) {
    Engine * e = engine(handle);
    const std::string text = to_string(env, jtext);
    int n = -llama_tokenize(e->vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0,
                            /*add_special=*/false, /*parse_special=*/true);
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(e->vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), n,
                       false, true) != n) {
        throw_java(env, "tokenization failed");
        return nullptr;
    }
    jintArray out = env->NewIntArray(n);
    env->SetIntArrayRegion(out, 0, n, tokens.data());
    return out;
}

namespace {

// Decodes tokens at positions [start, start + n) of seq 0. When want_hidden, the last
// token is the only output and its result_norm row is returned; otherwise nothing is
// output (used to fill the cache with a shared prefix).
jfloatArray decode(JNIEnv * env, Engine * e, jintArray jtokens, int start, bool want_hidden) {
    const jsize n = env->GetArrayLength(jtokens);
    if (n <= 0 || start + n > e->n_ctx) {
        throw_java(env, "prompt has " + std::to_string(start + n) + " tokens; the limit is " + std::to_string(e->n_ctx));
        return nullptr;
    }
    std::vector<llama_token> tokens(n);
    env->GetIntArrayRegion(jtokens, 0, n, tokens.data());
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = start + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = want_hidden && i == n - 1;
    }
    batch.n_tokens = n;
    e->captured = false;
    const int rc = llama_decode(e->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw_java(env, "llama_decode failed with code " + std::to_string(rc));
        return nullptr;
    }
    if (!want_hidden) return nullptr;
    if (!e->captured) {
        throw_java(env, "the model graph produced no result_norm tensor");
        return nullptr;
    }
    jfloatArray out = env->NewFloatArray(e->n_embd);
    env->SetFloatArrayRegion(out, 0, e->n_embd, e->hidden.data());
    return out;
}

}  // namespace

// Whole prompt from an empty cache.
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_openjev_mobile_engine_Native_hiddenState(JNIEnv * env, jobject, jlong handle, jintArray jtokens) {
    Engine * e = engine(handle);
    // Each candidate is independent: drop the previous prompt's KV and recurrent state.
    llama_memory_clear(llama_get_memory(e->ctx), true);
    return decode(env, e, jtokens, 0, true);
}

// Continue the cached sequence with more tokens (prefix cache).
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_openjev_mobile_engine_Native_extend(JNIEnv * env, jobject, jlong handle, jintArray jtokens,
                                             jint start, jboolean want_hidden) {
    return decode(env, engine(handle), jtokens, start, want_hidden);
}

extern "C" JNIEXPORT void JNICALL
Java_com_openjev_mobile_engine_Native_reset(JNIEnv *, jobject, jlong handle) {
    llama_memory_clear(llama_get_memory(engine(handle)->ctx), true);
}

extern "C" JNIEXPORT void JNICALL
Java_com_openjev_mobile_engine_Native_saveState(JNIEnv *, jobject, jlong handle, jint slot) {
    Engine * e = engine(handle);
    auto & buf = e->saved[slot];
    buf.resize(llama_state_seq_get_size(e->ctx, 0));
    buf.resize(llama_state_seq_get_data(e->ctx, buf.data(), buf.size(), 0));
}

extern "C" JNIEXPORT void JNICALL
Java_com_openjev_mobile_engine_Native_restoreState(JNIEnv * env, jobject, jlong handle, jint slot) {
    Engine * e = engine(handle);
    const auto & buf = e->saved[slot];
    llama_memory_clear(llama_get_memory(e->ctx), true);
    if (buf.empty() || llama_state_seq_set_data(e->ctx, buf.data(), buf.size(), 0) != buf.size()) {
        throw_java(env, "could not restore the cached prefix");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_openjev_mobile_engine_Native_free(JNIEnv *, jobject, jlong handle) {
    Engine * e = engine(handle);
    if (!e) return;
    llama_free(e->ctx);
    llama_model_free(e->model);
    delete e;
}
