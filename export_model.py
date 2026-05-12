#!/usr/bin/env python3
"""
export_model.py — Convert a state_dict .pt file to TorchScript for C++ inference.

Usage:
    BOARD_SIZE=19 python export_model.py <state_dict.pt> <output_ts.pt>

Example:
    BOARD_SIZE=19 python export_model.py models_19x19_az20/0_best_model.pt models_19x19_az20/0_ts.pt
"""

import os
import sys
import torch


def strip_orig_mod(state_dict):
    """Remove the '_orig_mod.' prefix added by torch.compile."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def main():
    if len(sys.argv) != 3:
        print(f"Usage: BOARD_SIZE=19 python {sys.argv[0]} <state_dict.pt> <output_ts.pt>",
              file=sys.stderr)
        sys.exit(1)

    state_path  = sys.argv[1]
    output_path = sys.argv[2]

    os.environ.setdefault("BOARD_SIZE", "19")

    from model import NeuralNetwork  # import after setting BOARD_SIZE

    model = NeuralNetwork()
    raw = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(strip_orig_mod(raw))
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(output_path)

    # Quick sanity check
    with torch.no_grad():
        dummy = torch.zeros(1, 17, model.board_size, model.board_size)
        v, p = scripted(dummy)
        assert v.shape == (1, 1), f"unexpected value shape {v.shape}"
        assert p.shape[0] == 1,   f"unexpected policy shape {p.shape}"

    print(f"Saved TorchScript model to: {output_path}")
    print(f"  Board size : {model.board_size}x{model.board_size}")
    print(f"  Parameters : {model.count_parameters():,}")
    print(f"  Value shape: {list(v.shape)}")
    print(f"  Policy shape: {list(p.shape)}")


if __name__ == "__main__":
    main()
