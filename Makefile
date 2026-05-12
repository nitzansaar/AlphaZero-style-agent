# Makefile — build all C/C++ binaries for the Go AlphaZero project
# Run from the go/ directory.

TORCH := $(shell .venv/bin/python -c "import torch,os; print(os.path.dirname(torch.__file__))")

CC  = gcc
CXX = g++

CFLAGS   = -O2 -Wall -I.
CXXFLAGS = -O2 -std=c++17 -Wall -fno-pie -no-pie -I. \
           -I$(TORCH)/include \
           -I$(TORCH)/include/torch/csrc/api/include
LDFLAGS  = -L$(TORCH)/lib -Wl,-rpath,$(TORCH)/lib \
           -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10

# Shared source files
ENGINE_C = go_engine.c
MCTS_CPP = mcts.cpp
NN_CPP   = nn_inference.cpp
NPY_C    = npy_writer.c

# 19x19 compile-time overrides.
#
# Self-play runs many workers, so keep its per-process MCTS pool compact.
# The GTP evaluator runs one AlphaZero engine and needs a larger pool for
# full-strength 1600-simulation searches on an empty 19x19 board.
DEFINES_19_COMMON = -DBOARD_SIZE=19 -DNUM_POSITIONS=361 -DPASS_ACTION=361 \
                    -DACTION_SIZE=362 "-DKOMI=7.5f"
DEFINES_19 = $(DEFINES_19_COMMON) -DNODE_POOL_SIZE=200000
DEFINES_19_GTP = $(DEFINES_19_COMMON) -DNODE_POOL_SIZE=800000

BINARIES    = test_go_engine test_mcts test_nn selfplay_cpp gatekeeper gtp_engine
BINARIES_19 = selfplay_cpp_19 gatekeeper_19 gtp_engine_19 \
              test_go_engine_19 test_mcts_19 test_nn_19

.PHONY: all all_19 tests tests_19 clean

all: $(BINARIES)

all_19: $(BINARIES_19)

# Phase 1 — Go engine unit tests (pure C)
test_go_engine: $(ENGINE_C) tests/test_go_engine.c
	$(CC) $(CFLAGS) -o $@ $^

# Phase 2 — MCTS unit tests (C++, no LibTorch)
test_mcts: $(ENGINE_C) $(MCTS_CPP) tests/test_mcts.cpp
	$(CXX) -O2 -std=c++17 -Wall -I. -o $@ $^

# Phase 3 — NN inference unit tests (C++ + LibTorch)
test_nn: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) tests/test_nn.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Phase 4 — Self-play data generator (C++ + LibTorch)
selfplay_cpp: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) $(NPY_C) selfplay_cpp.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Model gating binary (C++ + LibTorch)
gatekeeper: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) gatekeeper.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# GTP v2 interface — Sabaki-compatible engine
gtp_engine: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) gtp_engine.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# ── 19×19 binaries ────────────────────────────────────────────────────────

selfplay_cpp_19: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) $(NPY_C) selfplay_cpp.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES_19) $(LDFLAGS) -o $@ $^

gatekeeper_19: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) gatekeeper.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES_19) $(LDFLAGS) -o $@ $^

gtp_engine_19: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) gtp_engine.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES_19_GTP) $(LDFLAGS) -o $@ $^

test_go_engine_19: $(ENGINE_C) tests/test_go_engine.c
	$(CC) $(CFLAGS) $(DEFINES_19) -o $@ $^

test_mcts_19: $(ENGINE_C) $(MCTS_CPP) tests/test_mcts.cpp
	$(CXX) -O2 -std=c++17 -Wall -I. $(DEFINES_19) -o $@ $^

test_nn_19: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) tests/test_nn.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES_19) $(LDFLAGS) -o $@ $^

# ── Test targets ───────────────────────────────────────────────────────────

# Run all test binaries
tests: test_go_engine test_mcts test_nn
	@echo "=== test_go_engine ==="
	./test_go_engine
	@echo "=== test_mcts ==="
	./test_mcts
	@echo "=== test_nn ==="
	./test_nn

# Run 19x19 test binaries (test_nn_19 requires an exported 19x19 TorchScript model)
tests_19: test_go_engine_19 test_mcts_19
	@echo "=== test_go_engine_19 ==="
	./test_go_engine_19
	@echo "=== test_mcts_19 ==="
	./test_mcts_19

clean:
	rm -f $(BINARIES) $(BINARIES_19)
