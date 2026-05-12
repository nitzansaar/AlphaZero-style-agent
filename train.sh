#!/bin/bash
# train.sh — Continuous AlphaZero selfplay → train loop.
# Runs forever until interrupted (Ctrl-C / kill).
# Run from the go/ directory: bash train.sh

set -euo pipefail
cd "$(dirname "$0")"

export BOARD_SIZE=19

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
else
    PYTHON_BIN="python3"
fi

LOGDIR=$("$PYTHON_BIN" -c "from config import Config as cfg; print(cfg.LOGDIR)")

# create a fresh model if none exists
"$PYTHON_BIN" - <<'PY'
import os

import torch

from config import Config as cfg
from model import NeuralNetwork
from checkpoint_metadata import iteration_model_path, latest_checkpoint_path, save_model_metadata

_, latest_model = latest_checkpoint_path(cfg.SAVE_MODEL_PATH)

if latest_model is None:
    os.makedirs(cfg.SAVE_MODEL_PATH, exist_ok=True)
    path = iteration_model_path(cfg.SAVE_MODEL_PATH, 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(NeuralNetwork().state_dict(), tmp_path)
    os.replace(tmp_path, path)
    metadata_path, _ = save_model_metadata(
        path,
        iteration=0,
        global_minibatch_step=0,
        global_step_samples=0,
        total_num_data_rows=0,
        train_steps_per_iteration=cfg.TRAIN_STEPS,
        batch_size=cfg.BATCH_SIZE,
    )
    print(f"[Bootstrap] Saved initial random model: {path}")
    print(f"[Bootstrap] Saved initial model metadata: {metadata_path}")
PY

# train loop
START_TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')
NUM_ITERATIONS=0

trap 'echo ""; echo "=== Stopped after $NUM_ITERATIONS iterations ==="; exit 0' INT TERM

while true; do
    ((++NUM_ITERATIONS))
    echo "=== Iteration $NUM_ITERATIONS at $(date) ==="

    echo "Phase 1/2: Self-play..."
    "$PYTHON_BIN" selfplay_cpp_runner.py || { echo "ERROR: selfplay failed"; exit 1; }

    echo "Phase 2/2: Training..."
    "$PYTHON_BIN" train.py || { echo "ERROR: training failed"; exit 1; }

    NEW_ITER=$(cat "${LOGDIR}/current_iteration.txt")
    echo "=== Iteration $NEW_ITER complete at $(date) ==="
done
