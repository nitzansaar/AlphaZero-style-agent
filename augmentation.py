"""
Data augmentation utilities for 5x5 Go.
Implements rotations and reflections to exploit board symmetry.
"""
import numpy as np
from game import BOARD_SIZE, NUM_POSITIONS, PASS_ACTION, ACTION_SIZE

def rotate_90(board_2d):
    """Rotate board 90 degrees clockwise."""
    return np.rot90(board_2d, k=-1)

def rotate_180(board_2d):
    """Rotate board 180 degrees."""
    return np.rot90(board_2d, k=2)

def rotate_270(board_2d):
    """Rotate board 270 degrees clockwise (90 counter-clockwise)."""
    return np.rot90(board_2d, k=1)

def reflect_horizontal(board_2d):
    """Reflect board horizontally."""
    return np.fliplr(board_2d)

def reflect_vertical(board_2d):
    """Reflect board vertically."""
    return np.flipud(board_2d)

def reflect_diagonal(board_2d):
    """Reflect board across main diagonal."""
    return board_2d.T

def reflect_anti_diagonal(board_2d):
    """Reflect board across anti-diagonal."""
    return np.fliplr(np.flipud(board_2d)).T

def transform_action(action_index, transform_type):
    """
    Transform action index according to transformation.

    Args:
        action_index: Original action index (0-24 for board, 25 for pass)
        transform_type: Type of transformation ('rotate_90', 'rotate_180', etc.)

    Returns:
        Transformed action index
    """
    # Pass action doesn't transform
    if action_index == PASS_ACTION:
        return PASS_ACTION

    row = action_index // BOARD_SIZE
    col = action_index % BOARD_SIZE
    max_idx = BOARD_SIZE - 1  # 4 for 5x5 board

    if transform_type == 'rotate_90':
        new_row, new_col = col, max_idx - row
    elif transform_type == 'rotate_180':
        new_row, new_col = max_idx - row, max_idx - col
    elif transform_type == 'rotate_270':
        new_row, new_col = max_idx - col, row
    elif transform_type == 'reflect_horizontal':
        new_row, new_col = row, max_idx - col
    elif transform_type == 'reflect_vertical':
        new_row, new_col = max_idx - row, col
    elif transform_type == 'reflect_diagonal':
        new_row, new_col = col, row
    elif transform_type == 'reflect_anti_diagonal':
        new_row, new_col = max_idx - col, max_idx - row
    else:
        return action_index

    return new_row * BOARD_SIZE + new_col

def transform_policy(policy, transform_type):
    """
    Transform policy distribution according to transformation.

    Args:
        policy: Original policy distribution (26 values: 25 positions + pass)
        transform_type: Type of transformation

    Returns:
        Transformed policy distribution
    """
    transformed_policy = np.zeros_like(policy)
    for i in range(len(policy)):
        if policy[i] > 0:
            new_index = transform_action(i, transform_type)
            transformed_policy[new_index] = policy[i]
    return transformed_policy

def augment_data(state_flat, policy, transform_type):
    """
    Augment a single data point with a transformation.

    Args:
        state_flat: Flat board state (25 values for board, or 27 with ko/passes)
        policy: Policy distribution (26 values: 25 positions + pass)
        transform_type: Type of transformation

    Returns:
        (augmented_state, augmented_policy)
    """
    # Extract just the board portion (first 25 values)
    board_only = state_flat[:NUM_POSITIONS]

    # Reshape to 2D
    board_2d = board_only.reshape(BOARD_SIZE, BOARD_SIZE)

    # Apply transformation to board
    if transform_type == 'rotate_90':
        board_2d = rotate_90(board_2d)
    elif transform_type == 'rotate_180':
        board_2d = rotate_180(board_2d)
    elif transform_type == 'rotate_270':
        board_2d = rotate_270(board_2d)
    elif transform_type == 'reflect_horizontal':
        board_2d = reflect_horizontal(board_2d)
    elif transform_type == 'reflect_vertical':
        board_2d = reflect_vertical(board_2d)
    elif transform_type == 'reflect_diagonal':
        board_2d = reflect_diagonal(board_2d)
    elif transform_type == 'reflect_anti_diagonal':
        board_2d = reflect_anti_diagonal(board_2d)

    # Reconstruct state with transformed board
    augmented_state = state_flat.copy()
    augmented_state[:NUM_POSITIONS] = board_2d.flatten()

    # Transform policy (handles pass action automatically)
    augmented_policy = transform_policy(policy, transform_type)

    return augmented_state, augmented_policy

def augment_data_17plane(state_17, policy, transform_type):
    """
    Augment a 17-plane AlphaZero state and its policy vector.

    Args:
        state_17:      np.ndarray of shape (17, BOARD_SIZE, BOARD_SIZE)
        policy:        np.ndarray of shape (ACTION_SIZE,) — MCTS visit probabilities
        transform_type: one of the strings returned by get_augmentations()

    Returns:
        (augmented_state_17, augmented_policy) with the same shapes
    """
    augmented = np.empty_like(state_17)
    for plane_idx in range(state_17.shape[0]):
        plane = state_17[plane_idx]  # (BOARD_SIZE, BOARD_SIZE)
        if transform_type == 'rotate_90':
            plane = rotate_90(plane)
        elif transform_type == 'rotate_180':
            plane = rotate_180(plane)
        elif transform_type == 'rotate_270':
            plane = rotate_270(plane)
        elif transform_type == 'reflect_horizontal':
            plane = reflect_horizontal(plane)
        elif transform_type == 'reflect_vertical':
            plane = reflect_vertical(plane)
        elif transform_type == 'reflect_diagonal':
            plane = reflect_diagonal(plane)
        elif transform_type == 'reflect_anti_diagonal':
            plane = reflect_anti_diagonal(plane)
        augmented[plane_idx] = plane

    augmented_policy = transform_policy(policy, transform_type)
    return augmented, augmented_policy


def get_augmentations():
    """Get list of all possible augmentation types (all 8 dihedral symmetries)."""
    return [
        'identity',
        'rotate_90',
        'rotate_180',
        'rotate_270',
        'reflect_horizontal',
        'reflect_vertical',
        'reflect_diagonal',
        'reflect_anti_diagonal'
    ]
