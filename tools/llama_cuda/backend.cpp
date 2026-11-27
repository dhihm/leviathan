#include <ggml-backend.h>
#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
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
}

// Returns the generated token count, or -1 with a diagnostic in message.
extern "C" int32_t leviathan_llama_cuda_generate_v1(
    const char * path, const int32_t * prompt, int32_t prompt_count, int32_t count,
    int32_t * output, char * message, int32_t capacity) {
    auto fail = [&](const char * reason) {
        std::snprintf(message, capacity, "%s", reason);
        return -1;
    };
    try {
        if (prompt_count < 1 || prompt_count > 512 || count < 1 || count > 128 ||
            prompt_count + count - 1 > 512) {
            return fail("CUDA prompt or generation exceeds the supported context");
        }
        llama_backend_init();
        struct Shutdown { ~Shutdown() { llama_backend_free(); } } shutdown;

        auto cuda = ggml_backend_reg_by_name("CUDA");
        if (!cuda || ggml_backend_reg_dev_count(cuda) == 0) {
            return fail("llama.cpp CUDA backend or NVIDIA device is unavailable");
        }
        ggml_backend_dev_t device = ggml_backend_reg_dev_get(cuda, 0);
        ggml_backend_dev_t devices[] = {device, nullptr};
        auto model_params = llama_model_default_params();
        model_params.devices = devices;
        model_params.n_gpu_layers = -1;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.main_gpu = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(path, model_params), llama_model_free);
        if (!model) return fail("llama.cpp could not load model on the CUDA device");

        char architecture[16] = {};
        llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture));
        if (std::strcmp(architecture, "llama") != 0) return fail("expected Llama GGUF");
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

        int32_t generated = 0;
        while (generated < count) {
            evaluation.observed = false;
            if (llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
                return fail("llama.cpp CUDA decode failed");
            }
            const auto * logits = llama_get_logits_ith(ctx.get(), -1);
            if (!evaluation.observed) return fail("decode produced no completed CUDA matrix multiply");
            if (!logits || vocabulary <= 0 ||
                !std::all_of(logits, logits + vocabulary, [](float value) { return std::isfinite(value); })) {
                return fail("llama.cpp returned invalid logits");
            }
            const llama_token next = std::max_element(logits, logits + vocabulary) - logits;
            output[generated++] = next;
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
