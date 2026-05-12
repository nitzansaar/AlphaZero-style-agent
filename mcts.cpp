#include "mcts.h"

#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <random>

/* ── RNG (thread-local so each worker thread gets an independent stream) ─ */

static thread_local std::mt19937 s_rng(42);
float g_mcts_c_puct = 1.414f;

void mcts_seed_rng(uint32_t seed) { s_rng.seed(seed); }
void mcts_set_c_puct(float c_puct) { if (c_puct > 0.0f) g_mcts_c_puct = c_puct; }

/* ── Pool helpers ─────────────────────────────────────────────────────── */

static int pool_alloc(NodePool *pool)
{
    assert(pool->next_free < NODE_POOL_SIZE &&
           "node pool exhausted — increase NODE_POOL_SIZE");
    return pool->next_free++;
}

static void node_init(Node *n, float prior, int player,
                      int parent_idx, int action_idx)
{
    n->prior        = prior;
    n->player       = player;
    n->parent_idx   = parent_idx;
    n->action_idx   = action_idx;
    n->visits       = 0;
    n->total_value  = 0.0f;
    n->virtual_loss = 0;
    n->num_children = 0;
    n->state_set    = 0;
    for (int a = 0; a < ACTION_SIZE; a++)
        n->children[a] = -1;
}

/* ── Public: init root ────────────────────────────────────────────────── */

int mcts_init_root(NodePool *pool, const GoState *state, int player)
{
    pool->next_free = 0;
    int root_idx    = pool_alloc(pool);   /* always 0 */
    node_init(&pool->nodes[root_idx], 0.0f, player, -1, -1);
    pool->nodes[root_idx].state     = *state;
    pool->nodes[root_idx].state_set = 1;
    return root_idx;
}

/* ── UCB child selection ──────────────────────────────────────────────── */

/*
 * Return the action index of the child with the highest UCB score.
 *
 *   U(s,a) = -Q_adj + C * P(s,a) * sqrt(N_parent) / (1 + N_child)
 *
 * where Q_adj = (W - virtual_loss) / (N + virtual_loss)
 * penalises nodes that are currently being explored in other branches of
 * the batch, discouraging duplicate paths.
 */
static int select_best_child(const NodePool *pool, int node_idx)
{
    const Node *parent = &pool->nodes[node_idx];
    int   Ns     = parent->visits + parent->virtual_loss;
    float sqrtNs = sqrtf((float)Ns);

    float best_score  = -FLT_MAX;
    int   best_action = -1;

    for (int a = 0; a < ACTION_SIZE; a++) {
        int child_idx = parent->children[a];
        if (child_idx < 0) continue;

        const Node *child = &pool->nodes[child_idx];
        int   Nsa = child->visits + child->virtual_loss;
        float Q;
        if (Nsa > 0) {
            Q = (child->total_value - (float)child->virtual_loss) / (float)Nsa;
        } else {
            Q = 0.0f;
        }

        float U = -Q + g_mcts_c_puct * child->prior * sqrtNs / (1.0f + (float)Nsa);

        if (U > best_score) {
            best_score  = U;
            best_action = a;
        }
    }

    return best_action;
}

/* ── Node expansion ───────────────────────────────────────────────────── */

/*
 * Create child nodes for every action whose masked policy probability > 0.
 * Mirrors Node::expand in mcts.py.
 *
 * policy:      ACTION_SIZE softmax probabilities from the NN
 * next_player: absolute player whose turn it is at the new children
 */
static void expand_node(NodePool *pool, int node_idx,
                        const float *policy, int next_player)
{
    Node *node = &pool->nodes[node_idx];

    /* Mask with legal moves; state is always in canonical form. */
    float valid[ACTION_SIZE];
    go_get_valid_moves(&node->state, 1, valid);

    /* Renormalise: sum policy mass over valid moves so priors sum to 1.
     * Without this, the exploration term C*P*sqrt(N)/(1+n) is deflated by
     * however much probability the network assigned to illegal moves. */
    float prior_sum = 0.0f;
    for (int a = 0; a < ACTION_SIZE; a++)
        prior_sum += policy[a] * valid[a];
    float scale = (prior_sum > 0.0f) ? 1.0f / prior_sum : 1.0f;

    for (int a = 0; a < ACTION_SIZE; a++) {
        float prob = policy[a] * valid[a];
        if (prob <= 0.0f) continue;
        int child_idx = pool_alloc(pool);
        node_init(&pool->nodes[child_idx], prob * scale, next_player, node_idx, a);
        node->children[a] = child_idx;
        node->num_children++;
    }
}

/* ── Dirichlet noise ──────────────────────────────────────────────────── */

/*
 * Mix Dirichlet noise into root child priors.
 * Mirrors the noise block in run_simulation_batched.
 *
 * new_prior[a] = ((1 - noise_frac) * prior[a] + noise_frac * noise[a]) * valid[a]
 * then renormalised so the priors of valid children sum to 1.
 */
static void add_dirichlet_noise(NodePool *pool, int root_idx,
                                float noise_alpha, float noise_frac)
{
    Node *root = &pool->nodes[root_idx];
    if (root->num_children == 0) return;
    if (noise_alpha <= 0.0f || noise_frac <= 0.0f) return;
    if (noise_frac > 1.0f) noise_frac = 1.0f;

    std::gamma_distribution<float> gamma(noise_alpha, 1.0f);

    float noise[ACTION_SIZE] = {};
    float noise_sum = 0.0f;
    for (int a = 0; a < ACTION_SIZE; a++) {
        if (root->children[a] < 0) continue;
        float g    = gamma(s_rng);
        noise[a]   = g;
        noise_sum += g;
    }
    if (noise_sum <= 0.0f) return;

    /* Children only exist for valid moves, so no re-masking needed. */
    float new_sum = 0.0f;
    for (int a = 0; a < ACTION_SIZE; a++) {
        int child_idx = root->children[a];
        if (child_idx < 0) continue;
        Node *child  = &pool->nodes[child_idx];
        float n      = noise[a] / noise_sum;
        child->prior = (1.0f - noise_frac) * child->prior + noise_frac * n;
        new_sum     += child->prior;
    }

    /* Renormalise so the prior distribution remains valid. */
    if (new_sum > 0.0f)
        for (int a = 0; a < ACTION_SIZE; a++)
            if (root->children[a] >= 0)
                pool->nodes[root->children[a]].prior /= new_sum;
}

/* ── Backup ───────────────────────────────────────────────────────────── */

/*
 * Propagate the result from leaf to root along path[].
 * Mirrors MonteCarloTreeSearch::backup in mcts.py.
 *
 * terminal:        true  → use actual game outcome
 * terminal_winner: 1 / -1 / 0  (only valid when terminal == true)
 * leaf_player:     absolute player at the leaf (NN value perspective)
 * nn_value:        NN scalar value from leaf_player's perspective
 */
static void backup(NodePool  *pool,
                   const int *path,  int path_len,
                   bool       terminal, int terminal_winner,
                   int        leaf_player, float nn_value)
{
    for (int i = path_len - 1; i >= 0; i--) {
        Node *n = &pool->nodes[path[i]];
        n->visits++;

        float v;
        if (terminal) {
            if (terminal_winner == 0)
                v = 0.0f;
            else
                v = (terminal_winner == n->player) ? 1.0f : -1.0f;
        } else {
            /* NN value is from leaf_player's perspective.  Flip sign if
             * this node's player is the opponent of the leaf. */
            v = (leaf_player == n->player) ? nn_value : -nn_value;
        }

        n->total_value += v;
    }
}

/* ── Leaf collection ──────────────────────────────────────────────────── */

/*
 * Record the path and leaf node index for a single simulation traversal.
 * The path includes every node from root to leaf (inclusive).
 */
struct LeafInfo {
    int leaf_idx;
    int path[MAX_PATH_DEPTH];
    int path_len;
};

/*
 * Walk the tree `batch_size` times (with virtual loss applied at each
 * visited node) and record the leaf reached in each traversal.
 */
static void collect_leaves(NodePool *pool, int root_idx, int batch_size,
                            LeafInfo *out, int *count_out)
{
    int count = 0;
    for (int b = 0; b < batch_size; b++) {
        int path[MAX_PATH_DEPTH];
        int path_len = 0;
        int cur      = root_idx;

        /* Apply virtual loss to root before descending. */
        pool->nodes[cur].virtual_loss++;
        path[path_len++] = cur;

        while (pool->nodes[cur].num_children > 0) {
            int best_action = select_best_child(pool, cur);
            if (best_action < 0) break;

            cur = pool->nodes[cur].children[best_action];
            pool->nodes[cur].virtual_loss++;
            path[path_len++] = cur;

            if (path_len >= MAX_PATH_DEPTH) break;
        }

        LeafInfo *li = &out[count++];
        li->leaf_idx = cur;
        memcpy(li->path, path, path_len * sizeof(int));
        li->path_len = path_len;
    }
    *count_out = count;
}

/* ── History collection for NN input ─────────────────────────────────── */

/*
 * Collect up to 8 canonical GoStates for a leaf node by walking parent_idx
 * links.  hist_out[0] = leaf, hist_out[1] = leaf's parent, ...
 * When the root is reached (parent_idx < 0) and game_hist is provided, the
 * caller's pre-root history (game_hist[1..]) is appended (game_hist[0] is the
 * root state, already included via the tree chain).
 * Returns the number of states written to hist_out.
 */
static int collect_leaf_history(const NodePool *pool, int leaf_idx,
                                 const GoState *game_hist, int game_hist_len,
                                 GoState *hist_out /* [8] */)
{
    int  n       = 0;
    int  cur     = leaf_idx;
    bool at_root = false;
    while (cur >= 0 && n < 8) {
        const Node *nd = &pool->nodes[cur];
        if (!nd->state_set) break;
        hist_out[n++] = nd->state;
        if (nd->parent_idx < 0) { at_root = true; break; }
        cur = nd->parent_idx;
    }
    if (at_root && game_hist != nullptr) {
        /* game_hist[0] == root state (already in hist_out); extend with older. */
        for (int g = 1; g < game_hist_len && n < 8; g++)
            hist_out[n++] = game_hist[g];
    }
    return n;
}

/* ── Main simulation loop ─────────────────────────────────────────────── */

/*
 * Batched MCTS simulation.
 *
 *  1. Expand root with a single NN call.
 *  2. Loop until num_simulations complete:
 *     a. Collect batch_size leaf nodes (virtual loss applied).
 *     b. Deduplicate leaves (same leaf reached by multiple paths).
 *     c. Compute game states for newly reached leaves.
 *     d. Batch NN evaluation of unique leaves.
 *     e. Remove virtual loss from all paths.
 *     f. Expand non-terminal leaves; back up every path.
 */
void mcts_simulate(NodePool *pool, NNEvalFn nn_fn,
                   int num_simulations, int batch_size, bool add_noise,
                   const GoState *game_hist, int game_hist_len,
                   float noise_alpha, float noise_frac)
{
    assert(batch_size <= MAX_BATCH_SIZE);

    Node *root = &pool->nodes[0];

    /* ── Step 1: expand root ──────────────────────────────────────────── */
    if (!go_game_ended(&root->state)) {
        float planes[17 * NUM_POSITIONS];
        {
            GoState rh[8];
            int rhl = collect_leaf_history(pool, 0, game_hist, game_hist_len, rh);
            go_board_to_planes_17_with_history(rh, rhl, root->player, planes);
        }

        float root_value;
        float root_policy[ACTION_SIZE];
        nn_fn(planes, 1, &root_value, root_policy);

        expand_node(pool, 0, root_policy, -root->player);

        if (add_noise)
            add_dirichlet_noise(pool, 0, noise_alpha, noise_frac);
    }

    /* ── Step 2: simulation loop ──────────────────────────────────────── */
    int sims_done = 0;
    while (sims_done < num_simulations) {
        int cur_batch = batch_size;
        if (sims_done + cur_batch > num_simulations)
            cur_batch = num_simulations - sims_done;

        /* 2a — collect leaves with virtual loss applied */
        LeafInfo raw[MAX_BATCH_SIZE];
        int      raw_count = 0;
        collect_leaves(pool, 0, cur_batch, raw, &raw_count);

        /* 2b — deduplicate by pool index (linear scan; batch ≤ 64) */
        int unique_idx[MAX_BATCH_SIZE];
        int unique_count      = 0;
        int raw_to_unique[MAX_BATCH_SIZE];

        for (int i = 0; i < raw_count; i++) {
            int li = raw[i].leaf_idx;
            int j  = 0;
            for (; j < unique_count; j++)
                if (unique_idx[j] == li) break;
            if (j == unique_count)
                unique_idx[unique_count++] = li;
            raw_to_unique[i] = j;
        }

        /* 2c — compute missing leaf states from parent + action */
        for (int j = 0; j < unique_count; j++) {
            Node *leaf = &pool->nodes[unique_idx[j]];
            if (leaf->state_set)            continue;
            if (leaf->parent_idx < 0)       continue;
            if (leaf->action_idx < 0)       continue;

            Node *parent = &pool->nodes[leaf->parent_idx];
            if (!parent->state_set)         continue;

            leaf->state     = go_next_state_canonical(&parent->state,
                                                       leaf->action_idx);
            leaf->state_set = 1;
        }

        /* 2d — build NN batch (one entry per unique leaf with a state) */
        float batch_planes  [MAX_BATCH_SIZE * 17 * NUM_POSITIONS];
        float batch_values  [MAX_BATCH_SIZE];
        float batch_policies[MAX_BATCH_SIZE * ACTION_SIZE];

        int unique_eval_idx[MAX_BATCH_SIZE];
        int unique_has_eval[MAX_BATCH_SIZE];
        int nn_batch = 0;

        for (int j = 0; j < unique_count; j++) {
            unique_has_eval[j] = 0;
            Node *leaf = &pool->nodes[unique_idx[j]];
            if (!leaf->state_set) continue;

            {
                GoState lh[8];
                int lhl = collect_leaf_history(pool, unique_idx[j],
                                               game_hist, game_hist_len, lh);
                go_board_to_planes_17_with_history(lh, lhl, leaf->player,
                                  batch_planes + nn_batch * 17 * NUM_POSITIONS);
            }
            unique_eval_idx[j] = nn_batch;
            unique_has_eval[j] = 1;
            nn_batch++;
        }

        if (nn_batch > 0)
            nn_fn(batch_planes, nn_batch, batch_values, batch_policies);

        /* 2e — remove virtual loss from all collected paths */
        for (int i = 0; i < raw_count; i++)
            for (int k = 0; k < raw[i].path_len; k++)
                pool->nodes[raw[i].path[k]].virtual_loss--;

        /* 2f — expand unique leaves and back up every path */
        for (int j = 0; j < unique_count; j++) {
            int   li   = unique_idx[j];
            Node *leaf = &pool->nodes[li];

            if (!leaf->state_set || !unique_has_eval[j]) {
                /* No state → can't evaluate.  Back up zero value. */
                for (int i = 0; i < raw_count; i++) {
                    if (raw_to_unique[i] != j) continue;
                    backup(pool, raw[i].path, raw[i].path_len,
                           false, 0, leaf->player, 0.0f);
                }
                continue;
            }

            int   ei    = unique_eval_idx[j];
            float value = batch_values[ei];
            float *pol  = batch_policies + ei * ACTION_SIZE;

            bool terminal = go_game_ended(&leaf->state);
            int  winner   = terminal
                          ? go_get_winner(&leaf->state, leaf->player)
                          : 0;

            /* Expand only if not terminal and not already expanded
             * (another path in this batch may have expanded it first). */
            if (!terminal && leaf->num_children == 0)
                expand_node(pool, li, pol, -leaf->player);

            /* Back up every path that ended at this leaf. */
            for (int i = 0; i < raw_count; i++) {
                if (raw_to_unique[i] != j) continue;
                backup(pool, raw[i].path, raw[i].path_len,
                       terminal, winner, leaf->player, value);
            }
        }

        sims_done += cur_batch;
    }
}

/* ── Move selection ───────────────────────────────────────────────────── */

int mcts_select_move(const NodePool *pool,
                     float temperature,
                     float *action_probs_out)
{
    const Node *root = &pool->nodes[0];

    /* Compute and return normalised visit counts (training target). */
    float total = 0.0f;
    for (int a = 0; a < ACTION_SIZE; a++) {
        int   ci = root->children[a];
        float v  = (ci >= 0) ? (float)pool->nodes[ci].visits : 0.0f;
        action_probs_out[a] = v;
        total              += v;
    }
    if (total > 0.0f)
        for (int a = 0; a < ACTION_SIZE; a++)
            action_probs_out[a] /= total;

    /* Exploit (temperature ≈ 0): argmax visits. */
    if (temperature < 1e-6f) {
        int best_action = PASS_ACTION;
        int best_visits = -1;
        for (int a = 0; a < ACTION_SIZE; a++) {
            int ci = root->children[a];
            if (ci < 0) continue;
            int v = pool->nodes[ci].visits;
            if (v > best_visits) { best_visits = v; best_action = a; }
        }
        return best_action;
    }


    /* Explore: sample proportional to visits^(1/temperature) */
    float weights[ACTION_SIZE] = {};
    float wsum = 0.0f;
    for (int a = 0; a < ACTION_SIZE; a++) {
        int ci = root->children[a];
        if (ci < 0) continue;
        float w    = powf((float)pool->nodes[ci].visits, 1.0f / temperature);
        weights[a] = w;
        wsum       += w;
    }
    if (wsum <= 0.0f) return PASS_ACTION;

    std::uniform_real_distribution<float> uniform(0.0f, wsum);
    float r      = uniform(s_rng);
    float cumsum = 0.0f;
    for (int a = 0; a < ACTION_SIZE; a++) {
        cumsum += weights[a];
        if (r <= cumsum) return a;
    }
    return PASS_ACTION;
}
