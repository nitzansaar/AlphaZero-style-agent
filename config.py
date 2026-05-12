"""
Configuration for the AlphaZero-style Go training loop.

19x19 is the default training target. Smaller-board configs remain available
only for explicit legacy/debug runs via BOARD_SIZE.

Usage:
    from config import Config as cfg, BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE

    # Access settings
    print(cfg.BOARD_SIZE)
    print(cfg.NUM_SIMULATIONS)

To change board size, set environment variable before importing:
    BOARD_SIZE=9 python train.py
"""

import os

# Default to the real training target. Use BOARD_SIZE only for explicit legacy
# or debugging runs.
DEFAULT_BOARD_SIZE = int(os.environ.get('BOARD_SIZE', '19'))


class Config5x5:
    """Configuration for 5x5 Go (proof of concept)"""

    # Board settings
    BOARD_SIZE = 5
    NUM_POSITIONS = 25
    PASS_ACTION = 25
    ACTION_SIZE = 26  # 25 positions + 1 pass

    # Komi (compensation for white)
    KOMI = 2.5

    # Training settings
    BATCH_SIZE = 512
    TRAIN_STEPS = 15
    SELFPLAY_GAMES = 500
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_DECAY_ITERS = [100, 150]
    LR_DECAY_FACTOR = 0.1

    # MCTS settings
    NUM_SIMULATIONS = 1600
    MCTS_UCB_C = 1.414  # sqrt(2)

    # Playout cap randomization
    PLAYOUT_CAP_PROB = 0.25
    FAST_SIMS        = 100

    # Network architecture
    NUM_RES_BLOCKS = 6
    NUM_CHANNELS = 128
    VALUE_HEAD_HIDDEN = 256

    # Dataset
    DATASET_QUEUE_SIZE = 100000

    # Loss weights
    VALUE_LOSS_WEIGHT = 1.0
    POLICY_LOSS_WEIGHT = 1.0

    # Temperature for exploration
    TEMP_THRESHOLD = 6
    INITIAL_TEMP = 1.0

    # Data augmentation
    USE_AUGMENTATION = True

    # Model gating: new model must win >= GATE_WIN_RATE of GATE_GAMES games
    # against the previous best before replacing it for selfplay data generation.
    GATE_WIN_RATE = 0.55
    GATE_GAMES = 20
    GATE_SIMULATIONS = 200
    GATE_TEMPERATURE_MOVES = 4  # opening moves sampled proportionally; rest greedy

    # Paths (board-size specific)
    SAVE_MODEL_PATH = "models_5x5"
    SAVE_PICKLES = "pickles_5x5"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "logs_5x5"
    TEST_OUTPUT_PATH = "test_output_5x5"

    # Evaluation
    EVAL_GAMES = 40
    NUM_GAMES = 100


class Config9x9Base:
    """Base configuration for 9x9 Go (original scaled up settings)"""

    # Board settings
    BOARD_SIZE = 9
    NUM_POSITIONS = 81
    PASS_ACTION = 81
    ACTION_SIZE = 82  # 81 positions + 1 pass

    # Komi 
    KOMI = 6

    # Training settings - increased for larger board
    BATCH_SIZE = 512
    TRAIN_STEPS = 1500  
    SELFPLAY_GAMES = 500  # decreased to 500 to match katago
    LEARNING_RATE = 0.001  # SGD LR
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_DECAY_ITERS = [30, 60, 200, 400, 700]
    LR_DECAY_FACTOR = 0.1

    # MCTS settings
    NUM_SIMULATIONS = 800
    MCTS_UCB_C = 1.414

    # Playout cap randomization
    # A fraction PLAYOUT_CAP_PROB of moves use the full NUM_SIMULATIONS budget
    # and generate training data; the rest use FAST_SIMS to advance the game
    # cheaply without contributing training examples.
    PLAYOUT_CAP_PROB = 0.25
    FAST_SIMS        = 100

    # Network architecture - larger for 9x9
    NUM_RES_BLOCKS = 10
    NUM_CHANNELS = 256
    VALUE_HEAD_HIDDEN = 512

    # Dataset - larger buffer
    DATASET_QUEUE_SIZE = 500000

    # Loss weights — value loss is scaled up to match the policy cross-entropy
    # magnitude (~0.5 raw vs ~5.0 raw), so both heads contribute equally to
    # the shared backbone gradient.  POLICY_LR_MULTIPLIER is kept at 1.0
    # because the loss-weight balance already achieves the desired signal ratio.
    VALUE_LOSS_WEIGHT = 5.0
    POLICY_LOSS_WEIGHT = 1.0
    POLICY_LR_MULTIPLIER = 1.0

    # Temperature for exploration
    TEMP_THRESHOLD = 15  # Opening moves with temp=1; rest greedy (~17% of ~90-move game)
    INITIAL_TEMP = 1.0

    # Data augmentation
    USE_AUGMENTATION = True

    # Model gating: new model must win >= GATE_WIN_RATE of GATE_GAMES games
    # against the previous best before replacing it for selfplay data generation.
    GATE_WIN_RATE = 0.55
    GATE_GAMES = 20
    GATE_SIMULATIONS = 200
    GATE_TEMPERATURE_MOVES = 4  # opening moves sampled proportionally; rest greedy

    # Paths
    SAVE_MODEL_PATH = "models_9x9_base"
    SAVE_PICKLES = "pickles_9x9_base"
    DATASET_PATH = "training_dataset.pkl"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "logs_9x9_base"
    TEST_OUTPUT_PATH = "test_output_9x9_base"

    # Evaluation
    EVAL_GAMES = 40
    NUM_GAMES = 100


class Config19x19Base:
    """Base configuration for 19x19 Go (AlphaZero-style)."""

    # Board settings
    BOARD_SIZE = 19
    NUM_POSITIONS = 361
    PASS_ACTION = 361
    ACTION_SIZE = 362      # 361 positions + 1 pass

    # Komi — standard 19x19 tournament komi
    KOMI = 7.5

    # Training
    BATCH_SIZE = 512
    TRAIN_STEPS = 1500
    SELFPLAY_GAMES = 1000
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_DECAY_ITERS = [30, 60, 200, 400, 700]
    LR_DECAY_STEPS = [400_000, 600_000]
    LR_DECAY_FACTOR = 0.1

    NUM_SIMULATIONS = 1600
    MCTS_UCB_C = 1.414
    PLAYOUT_CAP_PROB = 0.25
    FAST_SIMS = 100

    # Self-play resignation.  RESIGN_THRESHOLD is a positive margin: resign
    # when MCTS root Q for the player to move drops below -RESIGN_THRESHOLD.
    # Disable in a fraction of games so the loop still produces full-game
    # samples for spotting false resignations, as AlphaGo Zero did.
    RESIGN_THRESHOLD = -1.0
    RESIGN_MIN_MOVE = 80
    RESIGN_DISABLE_PROB = 0.10
    RESIGN_AUTO_ADJUST = True
    RESIGN_TARGET_FALSE_POSITIVE_RATE = 0.05
    RESIGN_ADJUST_STEP = 0.02
    RESIGN_THRESHOLD_MIN = 0.80
    RESIGN_THRESHOLD_MAX = 0.99
    RESIGN_MIN_ADJUST_SAMPLES = 20
    RESIGN_STATE_FILE = "resign_threshold_state.json"

    # Network — deeper for 19x19
    NUM_RES_BLOCKS = 20
    NUM_CHANNELS = 256
    VALUE_HEAD_HIDDEN = 256

    DATASET_QUEUE_SIZE = 500_000
    DATA_LOADER_WORKERS = 0

    VALUE_LOSS_WEIGHT = 5.0
    POLICY_LOSS_WEIGHT = 1.0
    POLICY_LR_MULTIPLIER = 1.0

    # Temperature — first 30 moves exploratory
    TEMP_THRESHOLD = 30
    INITIAL_TEMP = 1.0

    # Data augmentation
    USE_AUGMENTATION = True

    # Gating
    GATE_WIN_RATE = 0.55
    GATE_GAMES = 20
    GATE_SIMULATIONS = 200
    GATE_TEMPERATURE_MOVES = 4

    # C++ MCTS knobs. AlphaZero's 19x19 root Dirichlet alpha is per action,
    # not scaled by board size.
    MCTS_DIRICHLET_ALPHA = 0.03
    MCTS_DIRICHLET_FRACTION = 0.25

    # 19x19 workers allocate a large MCTS node pool and, with CUDA enabled,
    # each process owns a CUDA context. This cap fits the current 24-core,
    # 32GB-VRAM training machine while leaving headroom.
    NUM_SELFPLAY_WORKERS = 16

    # Max moves per selfplay game; 19x19x2=722
    MAX_MOVES = 722

    # Paths
    SAVE_MODEL_PATH = "models_19x19_az20"
    SAVE_PICKLES = "pickles_19x19_az20"
    DATASET_PATH = "training_dataset.npz"
    BEST_MODEL = "{}_best_model.pt"
    LOGDIR = "logs_19x19_az20"
    TEST_OUTPUT_PATH = "test_output_19x19_az20"

    # Evaluation
    EVAL_GAMES = 40
    NUM_GAMES = 100


# Select configuration based on board size
_configs = {
    5: Config5x5,
    9: Config9x9Base,
    19: Config19x19Base,
}

if DEFAULT_BOARD_SIZE not in _configs:
    raise ValueError(f"Unsupported board size: {DEFAULT_BOARD_SIZE}. Supported: {list(_configs.keys())}")

Config = _configs[DEFAULT_BOARD_SIZE]

# Provide defaults for attributes introduced by this change so older configs
# (5x5, 9x9) don't need to be updated to define them.
if not hasattr(Config, 'NUM_SELFPLAY_WORKERS'):
    Config.NUM_SELFPLAY_WORKERS = None   # None → use os.cpu_count()
if not hasattr(Config, 'MAX_MOVES'):
    Config.MAX_MOVES = 200               # existing 9x9 default
if not hasattr(Config, 'POLICY_LR_MULTIPLIER'):
    Config.POLICY_LR_MULTIPLIER = 1.0   # 5x5 has no per-head LR boost
if not hasattr(Config, 'RESIGN_THRESHOLD'):
    Config.RESIGN_THRESHOLD = -1.0       # negative disables resignation
if not hasattr(Config, 'RESIGN_MIN_MOVE'):
    Config.RESIGN_MIN_MOVE = 0
if not hasattr(Config, 'RESIGN_DISABLE_PROB'):
    Config.RESIGN_DISABLE_PROB = 0.0
if not hasattr(Config, 'RESIGN_AUTO_ADJUST'):
    Config.RESIGN_AUTO_ADJUST = False
if not hasattr(Config, 'RESIGN_TARGET_FALSE_POSITIVE_RATE'):
    Config.RESIGN_TARGET_FALSE_POSITIVE_RATE = 0.05
if not hasattr(Config, 'RESIGN_ADJUST_STEP'):
    Config.RESIGN_ADJUST_STEP = 0.02
if not hasattr(Config, 'RESIGN_THRESHOLD_MIN'):
    Config.RESIGN_THRESHOLD_MIN = 0.80
if not hasattr(Config, 'RESIGN_THRESHOLD_MAX'):
    Config.RESIGN_THRESHOLD_MAX = 0.99
if not hasattr(Config, 'RESIGN_MIN_ADJUST_SAMPLES'):
    Config.RESIGN_MIN_ADJUST_SAMPLES = 20
if not hasattr(Config, 'RESIGN_STATE_FILE'):
    Config.RESIGN_STATE_FILE = "resign_threshold_state.json"
if not hasattr(Config, 'DATA_LOADER_WORKERS'):
    Config.DATA_LOADER_WORKERS = 8
if not hasattr(Config, 'MCTS_DIRICHLET_ALPHA'):
    Config.MCTS_DIRICHLET_ALPHA = 0.03 * Config.BOARD_SIZE
if not hasattr(Config, 'MCTS_DIRICHLET_FRACTION'):
    Config.MCTS_DIRICHLET_FRACTION = 0.25

# Export commonly used constants at module level for convenience
BOARD_SIZE = Config.BOARD_SIZE
NUM_POSITIONS = Config.NUM_POSITIONS
PASS_ACTION = Config.PASS_ACTION
ACTION_SIZE = Config.ACTION_SIZE
KOMI = Config.KOMI


def get_config(board_size=None):
    """Get configuration for a specific board size."""
    if board_size is None:
        board_size = DEFAULT_BOARD_SIZE
    if board_size not in _configs:
        raise ValueError(f"Unsupported board size: {board_size}. Supported: {list(_configs.keys())}")
    return _configs[board_size]


def print_config():
    """Print current configuration."""
    print(f"{'='*50}")
    print(f"AlphaZero Go Configuration")
    print(f"{'='*50}")
    print(f"Board Size: {Config.BOARD_SIZE}x{Config.BOARD_SIZE}")
    print(f"Action Space: {Config.ACTION_SIZE}")
    print(f"Komi: {Config.KOMI}")
    print(f"")
    print(f"Network:")
    print(f"  Residual Blocks: {Config.NUM_RES_BLOCKS}")
    print(f"  Channels: {Config.NUM_CHANNELS}")
    print(f"  Value Head Hidden: {Config.VALUE_HEAD_HIDDEN}")
    print(f"")
    print(f"Training:")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Train Steps: {Config.TRAIN_STEPS}")
    print(f"  Self-play Games: {Config.SELFPLAY_GAMES}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"")
    print(f"MCTS:")
    print(f"  Simulations: {Config.NUM_SIMULATIONS}")
    print(f"  UCB C: {Config.MCTS_UCB_C}")
    if Config.RESIGN_THRESHOLD >= 0:
        print(f"  Resign: Q < -{Config.RESIGN_THRESHOLD} after move {Config.RESIGN_MIN_MOVE}")
        print(f"  No-resign Games: {Config.RESIGN_DISABLE_PROB:.0%}")
        print(f"  Auto Resign Threshold: {'on' if Config.RESIGN_AUTO_ADJUST else 'off'}")
    else:
        print(f"  Resign: off")
    print(f"")
    print(f"Paths:")
    print(f"  Models: {Config.SAVE_MODEL_PATH}")
    print(f"  Data: {Config.SAVE_PICKLES}")
    print(f"  Logs: {Config.LOGDIR}")
    print(f"  Test Output: {Config.TEST_OUTPUT_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    print_config()
