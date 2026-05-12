import torch
import numpy as np
import pickle
import os
from copy import copy
from config import Config as cfg
import random
from game import board_to_canonical_17
from augmentation import augment_data, augment_data_17plane, get_augmentations

class GoDataset:
    def __init__(self, dataset, use_augmentation=False):
        self.dataset = dataset
        self.use_augmentation = use_augmentation and cfg.USE_AUGMENTATION
        if self.use_augmentation:
            self.augmentations = get_augmentations()

        # Validate shapes without stacking the whole replay buffer in memory.
        expected_state_shape = (17, cfg.BOARD_SIZE, cfg.BOARD_SIZE)
        for datapoint in dataset:
            state = datapoint[0]
            policy = datapoint[1]
            if isinstance(state, np.ndarray) and state.ndim == 3:
                if state.shape != expected_state_shape:
                    raise ValueError(
                        f"Expected C++ self-play state shape {expected_state_shape}, "
                        f"got {state.shape}. Check that Config.SAVE_PICKLES "
                        "matches the current 17-plane model."
                    )
            if np.asarray(policy).shape != (cfg.ACTION_SIZE,):
                raise ValueError(
                    f"Expected policy shape ({cfg.ACTION_SIZE},), "
                    f"got {np.asarray(policy).shape}"
                )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        state_raw, policy_raw, player, value = self.dataset[index]
        policy = np.asarray(policy_raw, dtype=np.float32)
        is_precomputed_3d = isinstance(state_raw, np.ndarray) and state_raw.ndim == 3

        if self.use_augmentation:
            transform_type = random.choice(self.augmentations)
            if is_precomputed_3d:
                # 17-plane C++ selfplay data — augment all planes in one call.
                aug_state, aug_p = augment_data_17plane(
                    state_raw, policy, transform_type
                )
                return (torch.from_numpy(aug_state.astype(np.float32)),
                        torch.tensor(value, dtype=torch.float32),
                        torch.from_numpy(aug_p.astype(np.float32)))
            else:
                # Legacy flat-board Python selfplay data.
                state_flat, aug_p = augment_data(
                    state_raw, policy, transform_type
                )
                state_canonical = board_to_canonical_17(state_flat, player)
                return (torch.from_numpy(state_canonical),
                        torch.tensor(value, dtype=torch.float32),
                        torch.from_numpy(np.array(aug_p, dtype=np.float32)))

        if is_precomputed_3d:
            state = np.asarray(state_raw, dtype=np.float32)
        else:
            state = board_to_canonical_17(state_raw, player)

        return (torch.from_numpy(state),
                torch.tensor(value, dtype=torch.float32),
                torch.from_numpy(policy))

class TrainingDataset:
    def __init__(self):
        self.training_dataset = []

    def calculate_values(self, dataset, winner):
        """Assign value to each position in the dataset based on the winner."""
        for ind, step in enumerate(dataset):
            step_ = copy(step)
            step_player = step_[2]
            if winner == 0:  # draw
                value = 0
            else:
                if winner == step_player:
                    value = 1
                else:
                    value = -1
            step_.append(value)
            dataset[ind] = step_
        return dataset

    def add_game_to_training_dataset(self, dataset, winner):
        """Add the completed game data to the training dataset."""
        data = self.calculate_values(dataset, winner)
        self.training_dataset.extend(data)
        self.training_dataset = self.training_dataset[-1 * cfg.DATASET_QUEUE_SIZE:]

    def load_from_npy(self, npy_dir):
        """Load positions produced by selfplay_cpp from .npy files.

        Reads:
          <npy_dir>/states.npy    (N, 17, BOARD_SIZE, BOARD_SIZE)  float32
          <npy_dir>/policies.npy  (N, ACTION_SIZE)                  float32
          <npy_dir>/values.npy    (N,)                              float32

        States are stored as pre-computed (17, B, B) tensors; GoDataset uses
        them directly without converting flat boards.  Plane 16 encodes
        color-to-play (1.0 = Black, 0.0 = White).

        Any additional files (ownership.npy, scores.npy) are silently ignored.
        """
        import os as _os
        states   = np.load(_os.path.join(npy_dir, 'states.npy'))    # (N, 17, BOARD_SIZE, BOARD_SIZE)
        policies = np.load(_os.path.join(npy_dir, 'policies.npy'))  # (N, ACTION_SIZE)
        values   = np.load(_os.path.join(npy_dir, 'values.npy'))    # (N,)

        expected_state_shape = (17, cfg.BOARD_SIZE, cfg.BOARD_SIZE)
        if states.ndim != 4 or tuple(states.shape[1:]) != expected_state_shape:
            raise ValueError(
                f"Expected states.npy shape (N, {expected_state_shape[0]}, "
                f"{expected_state_shape[1]}, {expected_state_shape[2]}), "
                f"got {states.shape}"
            )
        if policies.ndim != 2 or policies.shape[1] != cfg.ACTION_SIZE:
            raise ValueError(
                f"Expected policies.npy shape (N, {cfg.ACTION_SIZE}), "
                f"got {policies.shape}"
            )
        if len(states) != len(policies) or len(states) != len(values):
            raise ValueError(
                f"Mismatched npy rows: states={len(states)}, "
                f"policies={len(policies)}, values={len(values)}"
            )

        N = len(values)
        print(f"Loaded {N} positions from {npy_dir}")

        for i in range(N):
            # Plane 16 encodes color-to-play: 1.0 = Black (+1), 0.0 = White (-1).
            color = 1 if states[i, 16, 0, 0] > 0.5 else -1
            self.training_dataset.append([
                states[i],           # (17, BOARD_SIZE, BOARD_SIZE) pre-computed planes
                policies[i],         # (ACTION_SIZE,) MCTS visit probabilities
                color,               # current player (+1 or -1)
                float(values[i]),    # game outcome from current player's perspective
            ])

        self.training_dataset = self.training_dataset[-cfg.DATASET_QUEUE_SIZE:]

    def save(self, path):
        """Save the training dataset to a pickle file."""
        tmp_path = f"{path}.tmp.{os.getpid()}"
        try:
            if path.endswith(".npz"):
                states = np.stack([row[0] for row in self.training_dataset]).astype(np.float32)
                policies = np.stack([row[1] for row in self.training_dataset]).astype(np.float32)
                players = np.asarray([row[2] for row in self.training_dataset], dtype=np.int8)
                values = np.asarray([row[3] for row in self.training_dataset], dtype=np.float32)
                np.savez_compressed(
                    tmp_path,
                    states=states,
                    policies=policies,
                    players=players,
                    values=values,
                )
                if not tmp_path.endswith(".npz") and os.path.exists(f"{tmp_path}.npz"):
                    os.replace(f"{tmp_path}.npz", tmp_path)
            else:
                with open(tmp_path, 'wb') as handle:
                    pickle.dump(self.training_dataset, handle)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(f"{tmp_path}.npz"):
                os.remove(f"{tmp_path}.npz")

    def load(self, path):
        """Load the training dataset from a pickle file."""
        if path.endswith(".npz"):
            with np.load(path) as data:
                states = data["states"]
                policies = data["policies"]
                players = data["players"]
                values = data["values"]
                self.training_dataset = [
                    [states[i], policies[i], int(players[i]), float(values[i])]
                    for i in range(len(values))
                ]
        else:
            with open(path, 'rb') as handle:
                self.training_dataset = pickle.load(handle)

    def retreive_test_train_data(self):
        data = self.training_dataset
        num_samples = len(data)
        train_idx = np.random.choice(np.arange(num_samples), int(num_samples), replace=False)
        train_idx_set = set(train_idx)
        val_idx = [t for t in range(num_samples) if t not in train_idx_set]
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        return GoDataset(train_data), GoDataset(val_data)
