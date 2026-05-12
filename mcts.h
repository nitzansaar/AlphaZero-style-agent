#pragma once

/*
 * mcts.h — AlphaZero-style batched MCTS tree in C++.
 *
 * Mirrors mcts.py (run_simulation_batched, backup, select_move) but uses a
 * flat node pool with zero heap allocation in the hot simulation loop.
 *
 * Usage:
 *   go_init();
 *   NodePool *pool = new NodePool;
 *   GoState s = go_initial_state();
 *   mcts_init_root(pool, &s, 1);          // player 1 = Black
 *   mcts_simulate(pool, my_nn, 800, 32, true, nullptr, 0);
 *   float probs[ACTION_SIZE];
 *   int action = mcts_select_move(pool, 1.0f, probs);
 */

#include "go_engine.h"

/* ── Compile-time limits ──────────────────────────────────────────────── */

/* Maximum nodes in the pool. */
#ifndef NODE_POOL_SIZE
#  define NODE_POOL_SIZE  100000
#endif

/* Maximum batch_size argument to mcts_simulate. */
#define MAX_BATCH_SIZE  64

/* Maximum depth of a single simulation path (19x19 game length bound). */
#define MAX_PATH_DEPTH  800

/* ── MCTS hyper-parameters ────────────────────────────────────────────── */

extern float g_mcts_c_puct;            /* exploration constant */
constexpr float DIR_ALPHA   = 0.03f * BOARD_SIZE;  /* Dirichlet concentration, scaled by board size (KataGo: 0.03*size) */
constexpr float DIR_FRAC    = 0.25f;   /* noise mix-in fraction           */

void mcts_set_c_puct(float c_puct);

/* ── Node ─────────────────────────────────────────────────────────────── */

struct Node {
    /* Search statistics */
    float prior;         /* P(s,a) prior probability from NN            */
    int   player;        /* absolute player to move: +1 Black / -1 White */
    int   parent_idx;    /* pool index of parent; -1 for root           */
    int   action_idx;    /* action that led to this node; -1 for root   */
    int   visits;        /* N(s,a) visit count                          */
    float total_value;   /* W(s,a) accumulated value                    */
    int   virtual_loss;  /* temporary penalty for parallel traversals   */

    /* Children indexed by action (0..ACTION_SIZE-1).
     * children[a] = pool index of child reached by action a, or -1.
     * Iterating all 82 slots is cheap and avoids pointer chasing. */
    int children[ACTION_SIZE];
    int num_children;

    /* Game state at this node in canonical form (current player = +1). */
    GoState state;
    int     state_set;   /* 1 once state has been assigned              */
};

/* ── Node pool (bump allocator) ───────────────────────────────────────── */

/*
 * All nodes for one MCTS search live here.  Reset by calling
 * mcts_init_root() before each move.  ~44 MB — allocate on heap or as a
 * static/global, never on the stack.
 */
struct NodePool {
    Node nodes[NODE_POOL_SIZE];
    int  next_free;
};

/* ── Neural-network evaluation callback ──────────────────────────────────
 *
 * planes:     batch_size × 17 × NUM_POSITIONS floats (AlphaZero 17-plane form
 *             produced by go_board_to_planes_17_with_history)
 * batch_size: number of boards in the batch
 * values:     [out] batch_size floats  (scalar value in [-1, 1])
 * policies:   [out] batch_size × ACTION_SIZE floats (softmax probabilities)
 */
typedef void (*NNEvalFn)(const float *planes, int batch_size,
                          float *values, float *policies);

/* ── Public API ───────────────────────────────────────────────────────── */

/*
 * Reset the pool and initialise root node 0.
 * Must be called before mcts_simulate for each new position.
 * Returns 0 (always the root index).
 */
int mcts_init_root(NodePool *pool, const GoState *state, int player);

/*
 * Run batched MCTS simulations from root (nodes[0]).
 * Mirrors mcts.py::run_simulation_batched exactly.
 *
 *   nn_fn           NN callback used to evaluate leaf nodes
 *   num_simulations total number of simulations to run
 *   batch_size      leaves to collect per NN call (must be ≤ MAX_BATCH_SIZE)
 *   add_noise       add Dirichlet noise to root child priors
 *   game_hist       canonical GoState history: [0]=root, [1]=1 move ago, ...
 *                   Pass nullptr/0 to use only tree-depth history.
 *   game_hist_len   number of valid entries in game_hist (0..8)
 *   noise_alpha     Dirichlet concentration when add_noise is true
 *   noise_frac      Fraction of each prior replaced by sampled noise
 */
void mcts_simulate(NodePool *pool, NNEvalFn nn_fn,
                   int num_simulations, int batch_size, bool add_noise,
                   const GoState *game_hist = nullptr, int game_hist_len = 0,
                   float noise_alpha = DIR_ALPHA,
                   float noise_frac = DIR_FRAC);

/*
 * Select a move from root using temperature-weighted visit counts.
 * Mirrors mcts.py::select_move (explore mode).
 *
 *   temperature      1.0  → proportional to visits  (early game)
 *                    0.0  → argmax visits            (late game / exploit)
 *   action_probs_out [out] ACTION_SIZE normalised visit counts (training target)
 *
 * Returns the selected action index.
 */
int mcts_select_move(const NodePool *pool,
                     float temperature,
                     float *action_probs_out);

/*
 * Re-seed the thread-local RNG used for Dirichlet noise and move sampling.
 * Call once per worker thread before starting self-play (avoids all threads
 * producing identical game sequences).
 */
void mcts_seed_rng(uint32_t seed);
