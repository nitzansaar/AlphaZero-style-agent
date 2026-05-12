#pragma once

/*
 * nn_inference.h — LibTorch TorchScript model inference.
 *
 * Loads a TorchScript model produced by export_model.py and exposes
 * a batched eval() method compatible with the NNEvalFn typedef in mcts.h.
 *
 * Usage in selfplay binary:
 *   NNInference nn("models_9x9/157_ts.pt", /*use_cuda=*‌/true);
 *
 *   static NNInference *g_nn = &nn;
 *   auto wrapper = [](const float *p, int bs, float *v, float *pol){
 *       g_nn->eval(p, bs, v, pol);
 *   };
 *   mcts_simulate(&pool, wrapper, 800, 32, true);
 *
 * Compile flags (from go/ directory):
 *   TORCH=$(python -c "import torch,os; print(os.path.dirname(torch.__file__))")
 *   g++ -O2 -std=c++17 -fno-pie -no-pie -I. \
 *       -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
 *       -L$TORCH/lib -Wl,-rpath,$TORCH/lib \
 *       -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 \
 *       go_engine.c mcts.cpp nn_inference.cpp <your_main.cpp>
 */

#include "mcts.h"          /* NNEvalFn, ACTION_SIZE, NUM_POSITIONS  */

#include <torch/script.h>  /* torch::jit::script::Module            */
#include <string>

class NNInference {
public:
    /*
     * Load a TorchScript model exported by export_model.py.
     *   model_path : path to the .pt TorchScript file
     *   use_cuda   : run on GPU if available; falls back to CPU silently
     */
    explicit NNInference(const std::string &model_path, bool use_cuda = false);

    /*
     * Batched neural-network evaluation.  Signature matches NNEvalFn so
     * a thin static wrapper (see header comment) can serve as the callback.
     *
     *   planes   : batch_size × 17 × NUM_POSITIONS floats in row-major order
     *              (AlphaZero 17-plane form produced by go_board_to_planes_17_with_history)
     *   values   : [out] batch_size scalars in [-1, 1]  (value head, tanh)
     *   policies : [out] batch_size × ACTION_SIZE softmax probabilities
     *              (softmax applied here; NN output is raw logits)
     */
    void eval(const float *planes, int batch_size,
              float *values, float *policies);

    bool on_cuda() const { return device_.is_cuda(); }

private:
    torch::jit::script::Module module_;
    torch::Device              device_;
};
