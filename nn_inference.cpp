#include "nn_inference.h"

#include <torch/torch.h>
#include <stdexcept>

/* ── Constructor ──────────────────────────────────────────────────────── */

NNInference::NNInference(const std::string &model_path, bool use_cuda)
    : device_(torch::kCPU)
{
    if (use_cuda && torch::cuda::is_available())
        device_ = torch::Device(torch::kCUDA);

    try {
        module_ = torch::jit::load(model_path, device_);
    } catch (const c10::Error &e) {
        throw std::runtime_error(
            std::string("NNInference: failed to load '") + model_path
            + "': " + e.what());
    }

    module_.eval();
}

/* ── eval ─────────────────────────────────────────────────────────────── */

void NNInference::eval(const float *planes, int batch_size,
                       float *values, float *policies)
{
    /*
     * Build input tensor from the caller's float buffer without copying
     * (from_blob wraps in-place; .to(device_) copies to GPU when needed).
     *
     * planes layout: batch × 17 × NUM_POSITIONS (flat).
     * NN expects:    batch × 17 × BOARD_SIZE × BOARD_SIZE.
     * Both shapes represent the same contiguous memory.
     */
    torch::Tensor input = torch::from_blob(
        const_cast<float *>(planes),
        {batch_size, 17, BOARD_SIZE, BOARD_SIZE},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).to(device_);

    torch::NoGradGuard no_grad;

    std::vector<torch::jit::IValue> inputs = {input};
    auto out_tuple = module_.forward(inputs).toTuple();

    /* Value head: (batch, 1) — squeeze to (batch,), move to CPU. */
    torch::Tensor val_t = out_tuple->elements()[0].toTensor()
                          .squeeze(1).cpu();

    /* Policy head: (batch, ACTION_SIZE) raw logits — softmax then CPU. */
    torch::Tensor pol_t = torch::softmax(
        out_tuple->elements()[1].toTensor(), /*dim=*/1
    ).cpu();

    /* Copy into the caller's output buffers. */
    const float *val_ptr = val_t.data_ptr<float>();
    const float *pol_ptr = pol_t.data_ptr<float>();

    for (int b = 0; b < batch_size; b++)
        values[b] = val_ptr[b];

    for (int i = 0; i < batch_size * ACTION_SIZE; i++)
        policies[i] = pol_ptr[i];
}
