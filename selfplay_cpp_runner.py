"""
selfplay_cpp_runner.py — C++-backed self-play data generator.

    python selfplay_cpp_runner.py

Requires:
    ./selfplay_cpp   — compiled from selfplay_cpp.cpp
    export_model.py  — included in this repo

How it works:
    1. Find the latest state-dict checkpoint in SAVE_MODEL_PATH
    2. Export it to TorchScript (.pt) so the C++ binary can load it
    3. Launch NUM_WORKERS C++ processes in parallel (1 thread each, own CUDA context)
    4. Load the .npy output from each worker and merge into the pickle dataset
    5. Save the merged dataset back to disk
"""

import os
import sys
import json
import time
import threading
import subprocess
import tempfile
import torch
from tqdm import tqdm
from config import Config as cfg
from dataset import TrainingDataset
from checkpoint_metadata import latest_checkpoint_path, MODEL_FILENAME

# ── Constants ────────────────────────────────────────────────────────────

# Path to the compiled C++ binary (same directory as this script).
_HERE = os.path.dirname(os.path.abspath(__file__))

# Select board-size-specific binary: keep "selfplay_cpp" for 9x9 (backward
# compat with any pre-compiled binary); use "selfplay_cpp_N" for other sizes.
_BINARY_NAME = "selfplay_cpp" if cfg.BOARD_SIZE == 9 else f"selfplay_cpp_{cfg.BOARD_SIZE}"
SELFPLAY_BINARY = os.path.join(_HERE, _BINARY_NAME)

# Cap workers from config (Config19x19Base sets NUM_SELFPLAY_WORKERS=4 because
# each 19x19 worker allocates ~374 MB for the node pool).
_worker_cap = getattr(cfg, 'NUM_SELFPLAY_WORKERS', None)
NUM_WORKERS = min(os.cpu_count() or 1, _worker_cap) if _worker_cap else (os.cpu_count() or 1)

# Whether to pass --cuda to the binary.  Falls back to CPU automatically
# if CUDA is unavailable on the target machine.
USE_CUDA = torch.cuda.is_available()

_RESIGN_METRIC_KEYS = {
    "resignations",
    "resign_disabled_candidates",
    "resign_false_positives",
}

# ── Helper functions ──────────────────────────────────────────────────────

def get_latest_model_path():
    """Return the path of the highest-numbered model checkpoint, or None."""
    latest_num, path = latest_checkpoint_path(cfg.SAVE_MODEL_PATH)
    if path is None:
        return None
    print(f"[Selfplay] Using latest model: iter_{latest_num} ({path})")
    return path


def _source_mtime(paths):
    latest = 0.0
    for path in paths:
        try:
            latest = max(latest, os.path.getmtime(path))
        except OSError:
            pass
    return latest


def verify_binary_fresh():
    """Fail early if the self-play binary predates sources it depends on."""
    binary_mtime = os.path.getmtime(SELFPLAY_BINARY)
    sources = [
        os.path.join(_HERE, "selfplay_cpp.cpp"),
        os.path.join(_HERE, "mcts.cpp"),
        os.path.join(_HERE, "mcts.h"),
        os.path.join(_HERE, "go_engine.c"),
        os.path.join(_HERE, "go_engine.h"),
        os.path.join(_HERE, "nn_inference.cpp"),
        os.path.join(_HERE, "nn_inference.h"),
        os.path.join(_HERE, "npy_writer.c"),
        os.path.join(_HERE, "npy_writer.h"),
    ]
    newest_source = _source_mtime(sources)
    if newest_source > binary_mtime:
        target = "selfplay_cpp" if cfg.BOARD_SIZE == 9 else f"selfplay_cpp_{cfg.BOARD_SIZE}"
        print(f"ERROR: {SELFPLAY_BINARY} is older than its source files.")
        print(f"Rebuild it with:  make {target}")
        sys.exit(1)


def _monitor_progress(worker_dirs, games_per_worker_list, stop_event, bars):
    """Background thread: poll each worker's progress file and update its tqdm bar."""
    last_counts = [0] * len(worker_dirs)

    def _flush():
        for i, (wdir, bar) in enumerate(zip(worker_dirs, bars)):
            try:
                with open(os.path.join(wdir, "progress")) as f:
                    count = int(f.read().strip())
            except (FileNotFoundError, ValueError):
                count = last_counts[i]
            if count > last_counts[i]:
                bar.update(count - last_counts[i])
                last_counts[i] = count

    while not stop_event.wait(0.5):
        _flush()
    _flush()  # final update after all processes finish


def export_torchscript(state_dict_path):
    """Run export_model.py to convert state_dict → TorchScript; return ts path."""
    if os.path.basename(state_dict_path) == MODEL_FILENAME:
        ts_path = os.path.join(os.path.dirname(state_dict_path), "model_ts.pt")
    elif state_dict_path.endswith("_best_model.pt"):
        ts_path = state_dict_path.replace("_best_model.pt", "_ts.pt")
    else:
        root, _ = os.path.splitext(state_dict_path)
        ts_path = f"{root}_ts.pt"
    env = os.environ.copy()
    env["BOARD_SIZE"] = str(cfg.BOARD_SIZE)
    subprocess.run(
        [sys.executable, os.path.join(_HERE, "export_model.py"),
         state_dict_path, ts_path],
        env=env,
        check=True,
    )
    return ts_path


def _clamp(value, low, high):
    return max(low, min(high, value))


def _resign_state_path():
    return os.path.join(cfg.LOGDIR, cfg.RESIGN_STATE_FILE)


def _resign_bounds():
    low = float(cfg.RESIGN_THRESHOLD_MIN)
    high = float(cfg.RESIGN_THRESHOLD_MAX)
    if low > high:
        low, high = high, low
    return low, high


def _resign_auto_enabled():
    return bool(cfg.RESIGN_AUTO_ADJUST) and float(cfg.RESIGN_THRESHOLD) >= 0.0


def load_resign_threshold():
    """Load the persisted auto-adjusted threshold, or fall back to config."""
    threshold = float(cfg.RESIGN_THRESHOLD)
    if threshold < 0.0:
        return threshold, {}

    low, high = _resign_bounds()
    threshold = _clamp(threshold, low, high)
    if not _resign_auto_enabled():
        return threshold, {}

    path = _resign_state_path()
    try:
        with open(path) as f:
            state = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return threshold, {}

    try:
        threshold = _clamp(float(state["threshold"]), low, high)
    except (KeyError, TypeError, ValueError):
        pass
    return threshold, state if isinstance(state, dict) else {}


def _write_json_atomic(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def update_resign_threshold(current_threshold, metrics, state):
    """Adjust the threshold to keep no-resign false positives below target."""
    candidates = int(metrics.get("resign_disabled_candidates", 0) or 0)
    false_positives = int(metrics.get("resign_false_positives", 0) or 0)
    resignations = int(metrics.get("resignations", 0) or 0)

    false_positive_rate = (
        false_positives / candidates if candidates > 0 else None
    )
    update = {
        "auto_adjust": bool(_resign_auto_enabled()),
        "threshold_used": round(current_threshold, 6),
        "threshold_next": round(current_threshold, 6),
        "target_false_positive_rate": float(cfg.RESIGN_TARGET_FALSE_POSITIVE_RATE),
        "disabled_candidates": candidates,
        "false_positives": false_positives,
        "resignations": resignations,
        "false_positive_rate": false_positive_rate,
        "action": "off",
    }

    if current_threshold < 0.0:
        return current_threshold, update
    if not _resign_auto_enabled():
        update["action"] = "fixed"
        return current_threshold, update

    low, high = _resign_bounds()
    target = float(cfg.RESIGN_TARGET_FALSE_POSITIVE_RATE)
    step = float(cfg.RESIGN_ADJUST_STEP)
    min_samples = int(cfg.RESIGN_MIN_ADJUST_SAMPLES)
    next_threshold = current_threshold

    if candidates < min_samples:
        update["action"] = "insufficient_samples"
    elif false_positive_rate > target:
        next_threshold = _clamp(current_threshold + step, low, high)
        update["action"] = "more_conservative" if next_threshold != current_threshold else "at_max"
    elif false_positive_rate < target * 0.5:
        next_threshold = _clamp(current_threshold - step, low, high)
        update["action"] = "more_aggressive" if next_threshold != current_threshold else "at_min"
    else:
        update["action"] = "keep"

    update["threshold_next"] = round(next_threshold, 6)

    history = state.get("history", []) if isinstance(state, dict) else []
    if not isinstance(history, list):
        history = []
    history.append({
        "timestamp": time.time(),
        "threshold_used": round(current_threshold, 6),
        "threshold_next": round(next_threshold, 6),
        "disabled_candidates": candidates,
        "false_positives": false_positives,
        "false_positive_rate": false_positive_rate,
        "resignations": resignations,
        "action": update["action"],
    })

    max_history = int(getattr(cfg, "RESIGN_HISTORY_SIZE", 100))
    _write_json_atomic(_resign_state_path(), {
        "threshold": next_threshold,
        "last_update": update,
        "history": history[-max_history:],
    })
    return next_threshold, update


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if cfg.BOARD_SIZE == 5:
        print("ERROR: 5x5 uses Python selfplay — C++ binary not built for 5x5.")
        print("Use:  python run_training_loop.py   (or selfplay.py directly)")
        sys.exit(1)

    total_start = time.time()

    # Verify binary exists.
    if not os.path.isfile(SELFPLAY_BINARY):
        print(f"ERROR: C++ binary not found: {SELFPLAY_BINARY}")
        if cfg.BOARD_SIZE == 9:
            print("Compile it with:  make selfplay_cpp")
        else:
            print(f"Compile it with:  make selfplay_cpp_{cfg.BOARD_SIZE}")
        sys.exit(1)
    verify_binary_fresh()

    os.makedirs(cfg.SAVE_PICKLES, exist_ok=True)
    os.makedirs(cfg.LOGDIR, exist_ok=True)
    save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)
    load_path = save_path
    if not os.path.exists(load_path) and save_path.endswith(".npz"):
        legacy_path = save_path[:-4] + ".pkl"
        if os.path.exists(legacy_path):
            load_path = legacy_path
    resign_threshold, resign_state = load_resign_threshold()

    # ── 1. Export TorchScript model ──────────────────────────────────────
    model_path = get_latest_model_path()
    if model_path is None:
        print("ERROR: No trained model found in", cfg.SAVE_MODEL_PATH)
        print("Run train.py first to generate an initial model.")
        sys.exit(1)

    print(f"State-dict model : {model_path}")
    model_iter, _ = latest_checkpoint_path(cfg.SAVE_MODEL_PATH)
    seed_base = int(time.time()) ^ (os.getpid() << 16) ^ (int(model_iter or 0) * 1000003)
    export_start = time.time()
    ts_path = export_torchscript(model_path)
    export_time = time.time() - export_start
    print(f"TorchScript model: {ts_path}  ({export_time:.1f}s)")

    # ── 2. Run C++ selfplay processes in parallel ─────────────────────────
    # Never spawn more workers than games (avoids 0-game processes).
    effective_workers = min(NUM_WORKERS, cfg.SELFPLAY_GAMES)
    games_per_worker  = cfg.SELFPLAY_GAMES // effective_workers
    remainder         = cfg.SELFPLAY_GAMES % effective_workers

    print(f"\nRunning {cfg.SELFPLAY_GAMES} games across {effective_workers} workers "
          f"({'GPU' if USE_CUDA else 'CPU'}):")
    if resign_threshold >= 0.0:
        mode = "auto" if _resign_auto_enabled() else "fixed"
        print(f"  resignation threshold: {resign_threshold:.3f} ({mode}; "
              f"min move {cfg.RESIGN_MIN_MOVE}, no-resign {cfg.RESIGN_DISABLE_PROB:.0%})")
    else:
        print("  resignation: off")

    agg_timings = {}
    agg_metrics = {}
    with tempfile.TemporaryDirectory(prefix="selfplay_cpp_") as tmp_dir:
        # Launch all workers in parallel.
        # Redirect each worker's stdout/stderr to a per-worker log file so
        # their output doesn't garble the tqdm progress bar.
        procs       = []
        worker_dirs = []
        log_files   = []
        for i in range(effective_workers):
            games      = games_per_worker + (1 if i < remainder else 0)
            worker_dir = os.path.join(tmp_dir, f"worker_{i}")
            os.makedirs(worker_dir)

            cmd = [
                SELFPLAY_BINARY,
                ts_path,
                "--games",          str(games),
                "--sims",           str(cfg.NUM_SIMULATIONS),
                "--batch",          "64",
                "--threads",        "1",
                "--output",         worker_dir,
                "--temp-moves",     str(cfg.TEMP_THRESHOLD),
                "--max-moves",      str(getattr(cfg, 'MAX_MOVES', 200)),
                "--seed",           str(seed_base + i * 1000),
                "--full-prob",      str(cfg.PLAYOUT_CAP_PROB),
                "--fast-sims",      str(cfg.FAST_SIMS),
                "--resign-threshold", str(resign_threshold),
                "--resign-min-move", str(cfg.RESIGN_MIN_MOVE),
                "--resign-disable-prob", str(cfg.RESIGN_DISABLE_PROB),
                "--c-puct",         str(cfg.MCTS_UCB_C),
                "--dirichlet-alpha", str(cfg.MCTS_DIRICHLET_ALPHA),
                "--dirichlet-frac", str(cfg.MCTS_DIRICHLET_FRACTION),
            ]
            if USE_CUDA:
                cmd.append("--cuda")

            log_f = open(os.path.join(worker_dir, "worker.log"), "w")
            log_files.append(log_f)
            procs.append(subprocess.Popen(cmd, stdout=log_f, stderr=log_f))
            worker_dirs.append(worker_dir)

        print(f"  {games_per_worker}–{games_per_worker + (1 if remainder else 0)} "
              f"games per worker  (logs in each worker dir)\n")

        selfplay_start = time.time()

        # One tqdm bar per worker, just like the Python selfplay version.
        games_list = [games_per_worker + (1 if i < remainder else 0)
                      for i in range(effective_workers)]
        bars = [
            tqdm(total=games_list[i],
                 desc=f"Worker {i:2d}",
                 unit="game",
                 position=i,
                 dynamic_ncols=True,
                 leave=True)
            for i in range(effective_workers)
        ]

        stop_event = threading.Event()
        monitor    = threading.Thread(
            target=_monitor_progress,
            args=(worker_dirs, games_list, stop_event, bars),
            daemon=True,
        )
        monitor.start()

        # Wait for all workers and check for failures.
        failed = []
        for i, proc in enumerate(procs):
            proc.wait()
            if proc.returncode != 0:
                failed.append(i)

        stop_event.set()
        monitor.join()
        for bar in bars:
            bar.close()

        for f in log_files:
            f.close()

        selfplay_time = time.time() - selfplay_start

        if failed:
            # Print the last few lines of each failed worker's log to help debug.
            for i in failed:
                log_path = os.path.join(worker_dirs[i], "worker.log")
                try:
                    with open(log_path) as lf:
                        lines = lf.readlines()
                    print(f"\n--- worker {i} log (last 20 lines) ---")
                    print("".join(lines[-20:]))
                except OSError:
                    pass
            print(f"ERROR: workers {failed} exited with non-zero status.")
            sys.exit(1)

        # ── 3. Merge into the existing pickle dataset ─────────────────────
        training_dataset = TrainingDataset()
        if os.path.exists(load_path):
            training_dataset.load(load_path)
            print(f"Existing dataset : {len(training_dataset.training_dataset)} samples")
        else:
            print("Starting with empty dataset")

        for worker_dir in worker_dirs:
            training_dataset.load_from_npy(worker_dir)

        # Read per-worker timing and resignation metrics before
        # TemporaryDirectory removes them.
        for wdir in worker_dirs:
            timing_path = os.path.join(wdir, "timing.json")
            try:
                with open(timing_path) as f:
                    worker_data = json.load(f)
                for k, v in worker_data.get("timings", {}).items():
                    target = agg_metrics if k in _RESIGN_METRIC_KEYS else agg_timings
                    target[k] = target.get(k, 0.0) + v
                for k, v in worker_data.get("metrics", {}).items():
                    agg_metrics[k] = agg_metrics.get(k, 0.0) + v
            except (FileNotFoundError, KeyError, ValueError):
                pass

    resign_threshold_next, resign_update = update_resign_threshold(
        resign_threshold, agg_metrics, resign_state
    )

    training_dataset.save(save_path)
    total_time = time.time() - total_start

    print(f"\nTotal training samples: {len(training_dataset.training_dataset)}")
    print(f"Self-play time   : {selfplay_time:.1f}s")
    print(f"Total time       : {total_time:.1f}s")
    if resign_threshold >= 0.0:
        fp_rate = resign_update["false_positive_rate"]
        fp_text = "n/a" if fp_rate is None else f"{fp_rate:.2%}"
        print(f"Resign threshold : {resign_threshold:.3f} → {resign_threshold_next:.3f} "
              f"({resign_update['action']}, false positives {fp_text})")

    # ── 4. Save timing data (same format as selfplay.py) ─────────────────
    # Aggregate per-phase timings written by each C++ worker.
    # Round for readability
    agg_timings = {k: round(v, 4) for k, v in agg_timings.items()}
    completed_games = int(agg_metrics.get("completed_games", 0) or 0)
    total_game_moves = agg_metrics.get("total_game_moves", 0) or 0
    if completed_games > 0:
        agg_metrics["avg_selfplay_game_moves"] = total_game_moves / completed_games
    agg_metrics = {k: round(v, 4) for k, v in agg_metrics.items()}

    timing_data = {
        "timings": agg_timings if agg_timings else {
            "selfplay_cpp": round(selfplay_time, 3),
        },
        "metrics": agg_metrics,
        "resignation": resign_update,
    }
    timing_path = os.path.join(cfg.LOGDIR, "selfplay_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)
    print(f"Timing saved to  : {timing_path}")
