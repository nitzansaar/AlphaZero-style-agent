# AlphaZero-style Go Agent

This project trains an AlphaZero-style agent to play 19×19 Go using MCTS and a
learned value/policy network. The **AZ20** model is a 20-residual-block, 256-channel
network trained from scratch via self-play on 19×19 Go.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Training](#training)
3. [Evaluating against KataGo](#evaluating-against-katago)
   - [Quick start — single 100-game match](#quick-start--single-100-game-match)
4. [Notebook](#notebook)

---

## Prerequisites

| Requirement | Path | Notes |
|---|---|---|
| AZ20 checkpoints | `models_19x19_az20/<iter>/` | Each subfolder has `model.pt` (state dict) and `model_ts.pt` (TorchScript). |
| C++ selfplay binary | `selfplay_cpp_19` | Built with `make selfplay_cpp_19`. Required for training. |
| GTP engine binary | `gtp_engine_19` | Built with `make gtp_engine_19`. Required for evaluation. |
| Pretrained KataGo models | `pretrained_katago_models/` | Available ELOs: **482**, **802**, **1070**. |
| Python env | `.venv/` or system `python3` | Requires PyTorch. Scripts auto-detect `.venv/bin/python`. |

### Building the C++ binaries

All binaries are built with `make` from the `go/` directory. LibTorch is
located automatically via the active Python environment.

```bash
cd go

# Build all 19x19 binaries (selfplay, gatekeeper, GTP engine, unit tests)
make all_19

# Or build individually
make selfplay_cpp_19   # self-play data generator (training)
make gtp_engine_19     # GTP engine (evaluation / Sabaki)
make gatekeeper_19     # model-vs-model gating binary
```

---

## Training

The training loop alternates between two phases each iteration:

1. **Self-play** — `selfplay_cpp_19` runs `NUM_SELFPLAY_WORKERS` parallel C++
   processes. Each worker loads the latest checkpoint, plays `SELFPLAY_GAMES`
   games via MCTS, and writes `(state, policy, value)` tuples as `.npy` files.
2. **Training** — `train.py` loads the accumulated dataset, samples minibatches,
   and updates the network weights with SGD for `TRAIN_STEPS` steps.

### Recommended: `train.sh` (continuous loop)

The simplest way to train is the shell wrapper, which bootstraps a random model
if none exists and then loops forever until interrupted:

```bash
cd go
bash train.sh
```

Stop it at any time with `Ctrl-C`. Training resumes automatically from the
latest checkpoint on the next run. Progress is logged to `logs_19x19_az20/`.
```

### Key hyperparameters (`config.py` — `Config19x19Base`)

| Constant | Value | Meaning |
|---|---|---|
| `NUM_RES_BLOCKS` | 20 | Residual blocks in the network (the "20" in AZ20). |
| `NUM_CHANNELS` | 256 | Convolutional channels per block. |
| `SELFPLAY_GAMES` | 1000 | Games generated per iteration. |
| `NUM_SELFPLAY_WORKERS` | 16 | Parallel C++ self-play processes. |
| `NUM_SIMULATIONS` | 1600 | MCTS simulations per move during self-play. |
| `TRAIN_STEPS` | 1500 | SGD minibatch steps per iteration. |
| `BATCH_SIZE` | 512 | Training minibatch size. |
| `LEARNING_RATE` | 0.01 | Initial SGD learning rate. |
| `LR_DECAY_ITERS` | [30, 60, 200, 400, 700] | Iterations at which LR is multiplied by 0.1. |
| `DATASET_QUEUE_SIZE` | 500,000 | Rolling replay buffer size (positions). |
| `SAVE_MODEL_PATH` | `models_19x19_az20/` | Where checkpoints are saved. |
| `LOGDIR` | `logs_19x19_az20/` | Per-iteration training history CSVs. |

Edit `config.py` (`Config19x19Base`) to change any of these before starting a run.

### Output layout

```
models_19x19_az20/
  <iter>/
    model.pt           <- state dict (PyTorch)
    model_ts.pt        <- TorchScript (used by GTP engine and evaluation)
    training_info.json <- metadata (iteration, steps, dataset size, ...)
    training_state.pt  <- optimizer state (needed to resume training)

logs_19x19_az20/
  <iter>_history.csv   <- per-minibatch loss for that iteration
  current_iteration.txt
```

---

## Evaluating against KataGo

Three options, from simplest to most configurable:

### Quick start — single 100-game match

Run the wrapper script from the `go/` directory:

```bash
cd go

# AZ20 iteration 146 vs KataGo ELO 482, 100 games, KataGo gets 60 visits/move
./run_az20_vs_katago_100.sh
```

Key environment variables (all optional — shown with their defaults):

| Variable | Default | Meaning |
|---|---|---|
| `AZ_ITERATION` | `latest` | AZ20 checkpoint to load (`latest` picks the highest-numbered iteration). |
| `KATAGO_ELO` | `482` | KataGo pretrained model tier. Available: `482`, `802`, `1070`. |
| `GAMES` | `100` | Number of games (colours alternate). |
| `AZ_SIMS` | `160` | MCTS simulations per move for AZ. |
| `AZ_BATCH` | `32` | Neural-net batch size for AZ. |
| `KATAGO_VISITS` | `60` | KataGo search visits per move. |
| `VERBOSE` | `0` | Set to `1` to print every move. |
| `CSV_PATH` | auto | Output CSV path; defaults to `results/az20_iter<N>_vs_katago_elo<E>_v<V>_<N>g_<ts>.csv`. |

Examples:

```bash
# Latest AZ iteration vs KataGo ELO 802, 50 games, faster search
KATAGO_ELO=802 GAMES=50 AZ_SIMS=80 KATAGO_VISITS=30 ./run_az20_vs_katago_100.sh

# Specific iteration with verbose output
AZ_ITERATION=87 KATAGO_ELO=482 VERBOSE=1 GAMES=10 ./run_az20_vs_katago_100.sh

# Hardest available KataGo opponent
KATAGO_ELO=1070 KATAGO_VISITS=120 ./run_az20_vs_katago_100.sh
```

The script prints a summary when it finishes:

```
Summary
=======
AZ iteration 146 vs KataGo ELO 482 visits 60
Games        : 100
AZ wins      : 72
KataGo wins  : 28
Draws        : 0
AZ score     : 72.0/100 = 72.00%  (approx 95% CI 62.8%-81.2%)
AZ as Black  : 40/50
AZ as White  : 32/50
Avg moves    : 203.4
CSV          : results/az20_iter0146_vs_katago_elo482_v60_100g_20260511-120000.csv
```

---

## Notebook

`notebooks/alphazero_vs_katago_19x19.ipynb` plays one full game between an AZ
checkpoint and a pretrained KataGo model and plots every board position. Edit
the configuration cell at the top to choose the AZ iteration and KataGo ELO,
then run all cells.