#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Evaluation {
    ggml_backend_dev_t device;
    bool observed = false;
};

bool observe_cuda_matmul(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & evaluation = *static_cast<Evaluation *>(opaque);
    if (!ask) {
        // The scheduler reports the node after synchronizing its backend.
        evaluation.observed = true;
        return true;
    }
    return !evaluation.observed && tensor->op == GGML_OP_MUL_MAT &&
        tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer) &&
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer)) ==
            evaluation.device;
}
using Clock = std::chrono::steady_clock;

double millis(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct LoadedModel {
    struct Backend {
        Backend() { llama_backend_init(); }
        ~Backend() { llama_backend_free(); }
    } backend;
    ggml_backend_dev_t devices[2] = {};
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
};

std::unique_ptr<LoadedModel> open_model(const char * path, double * timing, bool resident) {
    const auto backend_start = timing ? Clock::now() : Clock::time_point{};
    auto loaded_model = std::make_unique<LoadedModel>();

    auto cuda = ggml_backend_reg_by_name("CUDA");
    if (!cuda || ggml_backend_reg_dev_count(cuda) == 0) {
        throw std::runtime_error("llama.cpp CUDA backend or NVIDIA device is unavailable");
    }
    loaded_model->devices[0] = ggml_backend_reg_dev_get(cuda, 0);
    auto model_params = llama_model_default_params();
    model_params.devices = loaded_model->devices;
    model_params.n_gpu_layers = -1;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_params.main_gpu = 0;
    if (resident) {
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
        model_params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    }
    const auto load_start = timing ? Clock::now() : Clock::time_point{};
    loaded_model->model.reset(llama_model_load_from_file(path, model_params));
    const auto loaded = timing ? Clock::now() : Clock::time_point{};
    if (timing) timing[0] = millis(load_start, loaded);
    if (!loaded_model->model) throw std::runtime_error("llama.cpp could not load model on the CUDA device");

    char architecture[16] = {};
    llama_model_meta_val_str(loaded_model->model.get(), "general.architecture", architecture, sizeof(architecture));
    if (std::strcmp(architecture, "llama") != 0) throw std::runtime_error("expected Llama GGUF");
    if (timing) timing[1] = millis(backend_start, load_start) + millis(loaded, Clock::now());
    return loaded_model;
}

bool valid_request(int32_t prompt_count, int32_t count) {
    return prompt_count >= 1 && prompt_count <= 512 && count >= 1 && count <= 128 &&
        prompt_count + count - 1 <= 512;
}

using TokenCallback = int32_t (*)(void *, int32_t, int32_t);

int32_t run_request(
    LoadedModel & loaded_model, const int32_t * prompt, int32_t prompt_count, int32_t count,
    int32_t * output, char * message, int32_t capacity, double * timing, int32_t cancel_after,
    TokenCallback on_token = nullptr, void * opaque = nullptr) {
    auto fail = [&](const char * reason) {
        std::snprintf(message, capacity, "%s", reason);
        return -1;
    };
    try {
        if (!valid_request(prompt_count, count)) return fail("CUDA prompt or generation exceeds the supported context");
        const auto setup_started = timing ? Clock::now() : Clock::time_point{};
        auto & model = loaded_model.model;
        auto device = loaded_model.devices[0];
        const auto * vocab = llama_model_get_vocab(model.get());
        const auto vocabulary = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> tokens(prompt, prompt + prompt_count);
        if (std::any_of(tokens.begin(), tokens.end(), [&](int32_t id) {
            return id < 0 || id >= vocabulary;
        })) return fail("prompt token ID outside the vocabulary");

        const int context = prompt_count + count - 1;
        if (context > llama_model_n_ctx_train(model.get())) return fail("context exceeds model limit");
        Evaluation evaluation{device};
        auto params = llama_context_default_params();
        params.n_ctx = context;
        params.n_batch = prompt_count;
        params.n_ubatch = prompt_count;
        params.offload_kqv = true;
        params.op_offload = true;
        params.cb_eval = observe_cuda_matmul;
        params.cb_eval_user_data = &evaluation;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), params), llama_free);
        if (!ctx) return fail("llama.cpp could not initialize the CUDA context");

        if (timing) timing[1] += millis(setup_started, Clock::now());
        int32_t generated = 0;
        while (generated < count) {
            const auto decode_start = timing ? Clock::now() : Clock::time_point{};
            evaluation.observed = false;
            if (llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
                return fail("llama.cpp CUDA decode failed");
            }
            auto sample_start = decode_start;
            if (timing) llama_synchronize(ctx.get());
            if (timing && generated == 0) {
                sample_start = Clock::now();
                timing[2] = millis(decode_start, sample_start);
            }
            const auto * logits = llama_get_logits_ith(ctx.get(), -1);
            if (!evaluation.observed) return fail("decode produced no completed CUDA matrix multiply");
            if (!logits || vocabulary <= 0 ||
                !std::all_of(logits, logits + vocabulary, [](float value) { return std::isfinite(value); })) {
                return fail("llama.cpp returned invalid logits");
            }
            const llama_token next = std::max_element(logits, logits + vocabulary) - logits;
            output[generated++] = next;
            if (timing) {
                const auto ended = Clock::now();
                timing[2 + generated] = millis(sample_start, ended);
            }
            if (on_token && on_token(opaque, next, generated - 1) != 0) {
                return fail("token output callback failed");
            }
            if (cancel_after == generated) {
                std::snprintf(message, capacity, "%s", "cancelled");
                return -2;
            }
            if (llama_vocab_is_eog(vocab, next)) break;
            tokens.assign(1, next);
        }
        std::snprintf(message, capacity, "%s (%s); completed CUDA matmul in each of %d decode calls",
                      ggml_backend_dev_name(device), ggml_backend_dev_description(device), generated);
        return generated;
    } catch (...) {
        return fail("llama.cpp CUDA backend raised an exception");
    }
}

int32_t generate(
    const char * path, const int32_t * prompt, int32_t prompt_count, int32_t count,
    int32_t * output, char * message, int32_t capacity, double * timing) {
    if (!valid_request(prompt_count, count)) {
        std::snprintf(message, capacity, "%s", "CUDA prompt or generation exceeds the supported context");
        return -1;
    }
    try {
        auto model = open_model(path, timing, false);
        return run_request(*model, prompt, prompt_count, count, output, message, capacity, timing, 0);
    } catch (const std::exception & error) {
        std::snprintf(message, capacity, "%s", error.what());
        return -1;
    } catch (...) {
        std::snprintf(message, capacity, "%s", "llama.cpp CUDA backend raised an exception");
        return -1;
    }
}

}

// v1 remains available to clients that do not request timing.
extern "C" int32_t leviathan_llama_cuda_generate_v1(
    const char * path, const int32_t * prompt, int32_t prompt_count, int32_t count,
    int32_t * output, char * message, int32_t capacity) {
    return generate(path, prompt, prompt_count, count, output, message, capacity, nullptr);
}

// Output: model load, backend/context, prefill, first token, then one sample per later token.
extern "C" int32_t leviathan_llama_cuda_generate_v2(
    const char * path, const int32_t * prompt, int32_t prompt_count, int32_t count,
    int32_t * output, char * message, int32_t capacity, double * timing, int32_t timing_capacity) {
    if (timing == nullptr || count < 1 || count > 128 || timing_capacity < count + 3) {
        std::snprintf(message, capacity, "%s", "invalid CUDA timing buffer");
        return -1;
    }
    return generate(path, prompt, prompt_count, count, output, message, capacity, timing);
}

// One serial owner per process. Each run allocates and destroys its own context.
extern "C" void * leviathan_llama_cuda_open_v1(const char * path, char * message, int32_t capacity,
                                              double * timing) {
    try {
        return open_model(path, timing, true).release();
    } catch (const std::exception & error) {
        std::snprintf(message, capacity, "%s", error.what());
    } catch (...) {
        std::snprintf(message, capacity, "%s", "llama.cpp CUDA backend raised an exception");
    }
    return nullptr;
}

extern "C" void leviathan_llama_cuda_close_v1(void * session) {
    delete static_cast<LoadedModel *>(session);
}

static int32_t session_request(
    void * session, const int32_t * prompt, int32_t prompt_count, int32_t count,
    int32_t * output, char * message, int32_t capacity, double * timing, int32_t timing_capacity,
    int32_t cancel_after, TokenCallback on_token, void * opaque) {
    if (!session || !valid_request(prompt_count, count) || !timing || timing_capacity < count + 3 ||
        cancel_after < 0 || cancel_after >= count) {
        std::snprintf(message, capacity, "%s", "invalid CUDA session request");
        return -1;
    }
    timing[0] = timing[1] = 0;
    return run_request(*static_cast<LoadedModel *>(session), prompt, prompt_count, count, output,
                       message, capacity, timing, cancel_after, on_token, opaque);
}

extern "C" int32_t leviathan_llama_cuda_request_v1(
    void * session, const int32_t * prompt, int32_t prompt_count, int32_t count,
    int32_t * output, char * message, int32_t capacity, double * timing, int32_t timing_capacity,
    int32_t cancel_after) {
    return session_request(session, prompt, prompt_count, count, output, message, capacity, timing,
                           timing_capacity, cancel_after, nullptr, nullptr);
}

extern "C" int32_t leviathan_llama_cuda_request_stream_v1(
    void * session, const int32_t * prompt, int32_t prompt_count, int32_t count,
    int32_t * output, char * message, int32_t capacity, double * timing, int32_t timing_capacity,
    int32_t cancel_after, TokenCallback on_token, void * opaque) {
    if (!on_token) {
        std::snprintf(message, capacity, "%s", "missing CUDA token callback");
        return -1;
    }
    return session_request(session, prompt, prompt_count, count, output, message, capacity, timing,
                           timing_capacity, cancel_after, on_token, opaque);
}
