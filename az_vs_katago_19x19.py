"""
Play many 19x19 games between AlphaZero (gtp_engine_19) and KataGo via GTP.

Both engines are driven through GTP v2 by a small in-process arbiter (mirrors
go/notebooks/alphazero_vs_katago_19x19.ipynb). A lightweight GoState tracks the
board only so we can score the final position; legality/ko is handled by the
engines themselves.

Typical use:
    cd go
    python3 az_vs_katago_19x19.py --games 20 --katago-elo 19 \\
        --az-sims 100 --katago-visits 100 \\
        --csv results/az_vs_katago_19x19.csv

AlphaZero colour alternates between games by default (game 1: Black, game 2:
White, ...). Use --az-first-color black|white to pin it.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants (19x19)
# ---------------------------------------------------------------------------

BOARD_SIZE    = 19
NUM_POSITIONS = BOARD_SIZE * BOARD_SIZE   # 361
PASS_ACTION   = NUM_POSITIONS             # 361
KOMI          = 7.5
GTP_COLS      = 'ABCDEFGHJKLMNOPQRST'     # skip 'I'
MAX_MOVES     = BOARD_SIZE * BOARD_SIZE * 3

SCRIPT_DIR            = os.path.dirname(os.path.abspath(__file__))
KATAGO_BIN            = os.path.join(SCRIPT_DIR, 'KataGo', 'cpp', 'katago')
GTP_CONFIG            = os.path.join(SCRIPT_DIR, 'KataGo', 'cpp', 'configs', 'gtp_example.cfg')
GTP_ENGINE            = os.path.join(SCRIPT_DIR, 'gtp_engine_19')
MODELS_DIR            = os.path.join(SCRIPT_DIR, 'models_19x19_base-pass')
KATAGO_PRETRAINED_DIR = os.path.join(SCRIPT_DIR, 'pretrained_katago_models')


# ---------------------------------------------------------------------------
# Go engine (board tracking for scoring only)
# ---------------------------------------------------------------------------

NEIGHBORS: List[List[int]] = []
for _i in range(NUM_POSITIONS):
    _r, _c = _i // BOARD_SIZE, _i % BOARD_SIZE
    _nb = []
    if _r > 0:             _nb.append((_r-1)*BOARD_SIZE + _c)
    if _r < BOARD_SIZE-1:  _nb.append((_r+1)*BOARD_SIZE + _c)
    if _c > 0:             _nb.append(_r*BOARD_SIZE + (_c-1))
    if _c < BOARD_SIZE-1:  _nb.append(_r*BOARD_SIZE + (_c+1))
    NEIGHBORS.append(_nb)


def find_group(board, idx):
    color = board[idx]
    group, in_stack, liberty_seen = [], [False]*NUM_POSITIONS, [False]*NUM_POSITIONS
    stack = [idx]; in_stack[idx] = True; libs = 0
    while stack:
        cur = stack.pop(); group.append(cur)
        for nb in NEIGHBORS[cur]:
            if board[nb] == 0:
                if not liberty_seen[nb]: liberty_seen[nb] = True; libs += 1
            elif board[nb] == color and not in_stack[nb]:
                in_stack[nb] = True; stack.append(nb)
    return group, libs


def capture_dead_stones(board, player):
    opp = -player; captured = 0; visited = [False]*NUM_POSITIONS
    for idx in range(NUM_POSITIONS):
        if board[idx] != opp or visited[idx]: continue
        group, libs = find_group(board, idx)
        for pos in group: visited[pos] = True
        if libs == 0:
            for pos in group: board[pos] = 0
            captured += len(group)
    return captured


@dataclass
class GoState:
    board:              np.ndarray
    ko_point:           int
    consecutive_passes: int

    @staticmethod
    def initial():
        return GoState(np.zeros(NUM_POSITIONS, dtype=np.int8), -1, 0)

    def copy(self):
        return GoState(self.board.copy(), self.ko_point, self.consecutive_passes)


def go_next_state_canonical(state: GoState, action: int) -> GoState:
    ns = state.copy()
    if action == PASS_ACTION:
        ns.ko_point = -1; ns.consecutive_passes += 1; ns.board = -ns.board; return ns
    ns.consecutive_passes = 0; ns.board[action] = 1
    captured = capture_dead_stones(ns.board, 1)
    ko_point = -1
    if captured == 1:
        group, libs = find_group(ns.board, action)
        if len(group) == 1 and libs == 1:
            for nb in NEIGHBORS[action]:
                if ns.board[nb] == 0: ko_point = nb; break
    ns.ko_point = ko_point; ns.board = -ns.board; return ns


def go_game_ended(state: GoState) -> bool:
    return state.consecutive_passes >= 2


def count_territory(board_abs):
    bs = int(np.sum(board_abs == 1)); ws = int(np.sum(board_abs == -1))
    visited = (board_abs != 0).tolist()
    for start in range(NUM_POSITIONS):
        if visited[start]: continue
        region, bb, wb, stack = [], False, False, [start]
        visited[start] = True
        while stack:
            cur = stack.pop(); region.append(cur)
            for nb in NEIGHBORS[cur]:
                if   board_abs[nb] ==  1: bb = True
                elif board_abs[nb] == -1: wb = True
                elif not visited[nb]:     visited[nb] = True; stack.append(nb)
        if bb and not wb:  bs += len(region)
        elif wb and not bb: ws += len(region)
    return bs, ws


def go_get_winner(state: GoState, player: int) -> int:
    board_abs = state.board.copy()
    if player == -1: board_abs = -board_abs
    bs, ws = count_territory(board_abs)
    wt = ws + KOMI
    return 1 if bs > wt else (-1 if wt > bs else 0)


def score_string(state: GoState, player: int) -> str:
    board_abs = state.board.copy()
    if player == -1: board_abs = -board_abs
    bs, ws = count_territory(board_abs)
    wt = ws + KOMI
    diff = bs - wt
    if   diff > 0: return f'B+{diff:.1f}'
    elif diff < 0: return f'W+{-diff:.1f}'
    else:          return 'Draw'


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def az_to_gtp(action: int) -> str:
    if action == PASS_ACTION:
        return 'pass'
    row = action // BOARD_SIZE
    col = action % BOARD_SIZE
    return f'{GTP_COLS[col]}{BOARD_SIZE - row}'


def gtp_to_az(gtp_move: str) -> Optional[int]:
    gtp_move = gtp_move.strip().upper()
    if gtp_move in ('PASS', ''): return PASS_ACTION
    if gtp_move == 'RESIGN':     return None
    col = GTP_COLS.index(gtp_move[0])
    row = BOARD_SIZE - int(gtp_move[1:])
    return row * BOARD_SIZE + col


# ---------------------------------------------------------------------------
# GTP engine wrapper
# ---------------------------------------------------------------------------

class GTPEngine:
    """Subprocess wrapper for any GTP v2 engine."""

    def __init__(self, cmd: list, board_size: int = BOARD_SIZE, komi: float = KOMI,
                 label: str = ''):
        self.label      = label
        self.board_size = board_size
        self.komi       = komi
        self.proc       = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._cmd(f'boardsize {board_size}')
        self._cmd(f'komi {komi}')
        self._cmd('clear_board')

    def _cmd(self, command: str) -> str:
        self.proc.stdin.write(command + '\n')
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == '':
                stderr_out = self.proc.stderr.read()
                msg = f'{self.label}: process exited during {repr(command)}'
                if stderr_out.strip():
                    msg += f'\n{stderr_out.strip()}'
                raise RuntimeError(msg)
            line = line.rstrip('\n')
            if line.startswith('=') or line.startswith('?'):
                lines.append(line)
                while True:
                    nxt = self.proc.stdout.readline().rstrip('\n')
                    if nxt == '': break
                    lines.append(nxt)
                break
        resp = '\n'.join(lines)
        if resp.startswith('?'):
            raise RuntimeError(f'{self.label} GTP error for {repr(command)}: {resp}')
        return resp.lstrip('=').strip()

    def clear_board(self):
        """Reset board + komi so the engine is ready for a new game."""
        self._cmd(f'komi {self.komi}')
        self._cmd('clear_board')

    def play(self, color: str, gtp_vertex: str):
        self._cmd(f'play {color} {gtp_vertex}')

    def genmove(self, color: str) -> str:
        return self._cmd(f'genmove {color}').upper()

    def close(self):
        try: self._cmd('quit')
        except Exception: pass
        try: self.proc.wait(timeout=5)
        except Exception: self.proc.kill()


def make_az_engine(model_path: str, sims: int, batch: int) -> GTPEngine:
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False
    cmd = [GTP_ENGINE, model_path, '--sims', str(sims), '--batch', str(batch)]
    if use_cuda: cmd.append('--cuda')
    return GTPEngine(cmd, label='AlphaZero')


def make_katago_engine(model_path: str, visits: int) -> GTPEngine:
    override = f'maxVisits={visits},logToStderr=false,logDir='
    cmd = [
        KATAGO_BIN, 'gtp',
        '-model',  model_path,
        '-config', GTP_CONFIG,
        '-override-config', override,
    ]
    return GTPEngine(cmd, label='KataGo')


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def play_game(az: GTPEngine, kg: GTPEngine, az_color: int,
              verbose: bool = False) -> Tuple[list, GoState, int, bool, Optional[int], Optional[str]]:
    """
    Play one full game between the (already-reset) AZ and KG engines.

    az_color: +1 = AlphaZero is Black, -1 = AlphaZero is White.
    Returns (frames, final_state, final_player, resigned, forfeit_by, forfeit_reason).
    """
    state    = GoState.initial()
    player   = 1          # absolute: 1=Black, -1=White
    frames   = []
    resigned = False
    forfeit_by = None
    forfeit_reason = None

    for _ in range(MAX_MOVES):
        if go_game_ended(state):
            break

        color_str  = 'b' if player == 1 else 'w'
        is_az_turn = (player == az_color)
        relay_error = None

        t0 = time.time()
        if is_az_turn:
            gtp_vertex = az.genmove(color_str)
            if gtp_vertex != 'RESIGN':
                try:
                    kg.play(color_str, gtp_vertex)
                except RuntimeError as exc:
                    relay_error = exc
        else:
            gtp_vertex = kg.genmove(color_str)
            if gtp_vertex != 'RESIGN':
                try:
                    az.play(color_str, gtp_vertex)
                except RuntimeError as exc:
                    relay_error = exc
        elapsed = time.time() - t0

        engine_label = 'AZ' if is_az_turn else 'KG'
        if relay_error is not None:
            forfeit_by = player
            forfeit_reason = (
                f'{engine_label} produced illegal move '
                f'{"Black" if player == 1 else "White"} {gtp_vertex}: {relay_error}'
            )
            if verbose:
                print(f'  {forfeit_reason}')
            break

        action = gtp_to_az(gtp_vertex)

        if action is None:  # resign
            if verbose:
                print(f'  {engine_label} resigned at move {len(frames)+1}')
            resigned = True
            break

        frames.append({
            'move_num':   len(frames) + 1,
            'player':     player,
            'is_az':      is_az_turn,
            'gtp_vertex': gtp_vertex,
            'action':     action,
            'elapsed_s':  elapsed,
        })

        if verbose:
            color_name = 'Black' if player == 1 else 'White'
            print(f'  Move {len(frames):3d}: {engine_label} ({color_name}) '
                  f'-> {gtp_vertex}  ({elapsed:.1f}s)')

        state  = go_next_state_canonical(state, action)
        player = -player

    return frames, state, player, resigned, forfeit_by, forfeit_reason


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def resolve_az_model(explicit_path: Optional[str]) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f'AlphaZero model not found: {explicit_path}')
        return explicit_path
    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f'Models dir missing: {MODELS_DIR}')
    ts_files = sorted(
        [f for f in os.listdir(MODELS_DIR)
         if f.endswith('_ts.pt') and not f.startswith('-')],
        key=lambda f: int(f.split('_')[0]),
    )
    if not ts_files:
        raise FileNotFoundError(f'No *_ts.pt models found in {MODELS_DIR}')
    return os.path.join(MODELS_DIR, ts_files[-1])


def resolve_katago_model(elo: int) -> str:
    path = os.path.join(KATAGO_PRETRAINED_DIR, f'katago-elo-{elo}.gz')
    if not os.path.exists(path):
        available = sorted(os.listdir(KATAGO_PRETRAINED_DIR)) \
            if os.path.isdir(KATAGO_PRETRAINED_DIR) else []
        raise FileNotFoundError(
            f'KataGo model not found for elo={elo}: {path}\nAvailable: {available}'
        )
    return path


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    'game_idx', 'az_color', 'result', 'winner',
    'num_moves', 'az_won', 'az_total_s', 'kg_total_s',
    'forfeit_by', 'forfeit_reason',
]


def append_csv(csv_path: Optional[str], row: dict):
    if csv_path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or '.', exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AlphaZero vs KataGo 19x19 multi-game evaluator (GTP arbiter).'
    )
    parser.add_argument('--games', type=int, default=20,
                        help='Number of games to play (default: 20)')
    parser.add_argument('--az-model', default=None,
                        help='Path to AlphaZero torchscript model '
                             '(default: latest *_ts.pt in models_19x19_base/)')
    parser.add_argument('--az-sims', type=int, default=160,
                        help='AlphaZero MCTS simulations per move (default: 100)')
    parser.add_argument('--az-batch', type=int, default=32,
                        help='AlphaZero batch size (default: 32)')
    parser.add_argument('--katago-elo', type=int, default=19,
                        help='KataGo ELO tier in pretrained_katago_models/ (default: 19)')
    parser.add_argument('--katago-visits', type=int, default=60,
                        help='KataGo search visits per move (default: 100)')
    parser.add_argument('--az-first-color', choices=['black', 'white', 'alternate'],
                        default='alternate',
                        help='Colour AlphaZero plays. "alternate" (default) flips each game.')
    parser.add_argument('--csv', default=None,
                        help='Append per-game results to this CSV (header auto-written).')
    parser.add_argument('--verbose', action='store_true',
                        help='Print every move.')
    args = parser.parse_args()

    az_model_path     = resolve_az_model(args.az_model)
    katago_model_path = resolve_katago_model(args.katago_elo)

    for exe in (GTP_ENGINE, KATAGO_BIN):
        if not os.path.exists(exe):
            print(f'ERROR: required binary missing: {exe}', file=sys.stderr)
            sys.exit(1)

    print(f'AlphaZero model : {az_model_path}')
    print(f'  sims={args.az_sims} batch={args.az_batch}')
    print(f'KataGo    model : {katago_model_path}')
    print(f'  visits={args.katago_visits}')
    print(f'Games           : {args.games}   (az_first_color={args.az_first_color})')
    print(flush=True)

    print('Starting AlphaZero engine...', end=' ', flush=True)
    az = make_az_engine(az_model_path, args.az_sims, args.az_batch)
    print('ok')
    print('Starting KataGo engine   ...', end=' ', flush=True)
    kg = make_katago_engine(katago_model_path, args.katago_visits)
    print('ok\n')

    az_wins = kg_wins = draws = 0
    az_black_games = az_black_wins = 0
    az_white_games = az_white_wins = 0
    total_moves = 0
    t_start = time.time()

    try:
        for game_idx in range(args.games):
            if args.az_first_color == 'alternate':
                az_color = 1 if game_idx % 2 == 0 else -1
            else:
                az_color = 1 if args.az_first_color == 'black' else -1
            az_color_name = 'Black' if az_color == 1 else 'White'
            kg_color_name = 'White' if az_color == 1 else 'Black'

            az.clear_board()
            kg.clear_board()

            if args.verbose:
                print(f'=== Game {game_idx+1}/{args.games} | '
                      f'AZ={az_color_name} KG={kg_color_name} ===')

            t_game = time.time()
            frames, final_state, final_player, resigned, forfeit_by, forfeit_reason = play_game(
                az, kg, az_color, verbose=args.verbose,
            )
            game_elapsed = time.time() - t_game

            if forfeit_by is not None:
                winner_abs = -forfeit_by
                result_str = f'{"B" if winner_abs == 1 else "W"}+F'
            elif resigned:
                winner_abs = -final_player  # whoever's turn it was when opponent resigned
                result_str = f'{"B" if winner_abs == 1 else "W"}+R'
            else:
                winner_abs = go_get_winner(final_state, final_player)
                result_str = score_string(final_state, final_player)
            winner_name = 'Black' if winner_abs == 1 else ('White' if winner_abs == -1 else 'Draw')
            az_won = (winner_abs == az_color)

            if winner_abs == 0:
                draws += 1
            elif az_won:
                az_wins += 1
            else:
                kg_wins += 1

            if az_color == 1:
                az_black_games += 1
                if az_won: az_black_wins += 1
            else:
                az_white_games += 1
                if az_won: az_white_wins += 1

            total_moves += len(frames)

            az_total_s = sum(f['elapsed_s'] for f in frames if f['is_az'])
            kg_total_s = sum(f['elapsed_s'] for f in frames if not f['is_az'])

            append_csv(args.csv, {
                'game_idx':   game_idx + 1,
                'az_color':   az_color_name,
                'result':     result_str,
                'winner':     winner_name,
                'num_moves':  len(frames),
                'az_won':     int(az_won),
                'az_total_s': round(az_total_s, 3),
                'kg_total_s': round(kg_total_s, 3),
                'forfeit_by': 'AZ' if forfeit_by == az_color else ('KG' if forfeit_by == -az_color else ''),
                'forfeit_reason': forfeit_reason or '',
            })

            tag = 'AZ' if az_won else ('KG' if winner_abs != 0 else 'Draw')
            print(f'Game {game_idx+1:3d}/{args.games}  AZ={az_color_name}  '
                  f'-> {result_str:>8s}  ({tag:<4s} wins, '
                  f'{len(frames):3d} moves, {game_elapsed:5.1f}s)', flush=True)
            if args.verbose:
                if forfeit_reason:
                    print(f'  Forfeit: {forfeit_reason}')
                print(f'  AZ time: {az_total_s:.1f}s  KG time: {kg_total_s:.1f}s')
                print('=' * 60)
    finally:
        try: az.close()
        except Exception: pass
        try: kg.close()
        except Exception: pass

    total = args.games
    elapsed = time.time() - t_start
    katago_label = os.path.basename(katago_model_path).removesuffix('.gz')

    def pct(n, d):
        return f'{n/d*100:.1f}%' if d > 0 else '-'

    print()
    print('=' * 60)
    print(f'Results over {total} games  '
          f'(AZ sims={args.az_sims}, KG {katago_label} visits={args.katago_visits})')
    print(f'  AlphaZero wins : {az_wins:3d}  ({pct(az_wins, total)})')
    print(f'  KataGo    wins : {kg_wins:3d}  ({pct(kg_wins, total)})')
    print(f'  Draws          : {draws:3d}')
    print(f'  AZ as Black    : {az_black_wins}/{az_black_games}  '
          f'({pct(az_black_wins, az_black_games)})')
    print(f'  AZ as White    : {az_white_wins}/{az_white_games}  '
          f'({pct(az_white_wins, az_white_games)})')
    print(f'  Avg moves/game : {total_moves / total:.1f}')
    print(f'  Total elapsed  : {elapsed:.1f}s')
    print('=' * 60)
    if args.csv:
        print(f'Per-game CSV appended to: {args.csv}')


if __name__ == '__main__':
    main()
