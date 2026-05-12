/*
 * selfplay_cpp.cpp — AlphaZero-style self-play data generator.
 *
 * Produces three .npy files in an output directory:
 *   states.npy    float32 (N, 17, BOARD_SIZE, BOARD_SIZE)  AlphaZero 17-plane repr
 *   policies.npy  float32 (N, ACTION_SIZE)                  MCTS visit-count probs
 *   values.npy    float32 (N,)                              game outcome per position
 *
 * Usage:
 *   selfplay_cpp <model_ts.pt> [options]
 *   selfplay_cpp models_9x9/157_ts.pt --games 100 --sims 200 --threads 4
 *
 * Compile (from go/ directory):
 *   TORCH=$(.venv/bin/python -c "import torch,os; print(os.path.dirname(torch.__file__))")
 *   g++ -O2 -std=c++17 -fno-pie -no-pie -I. \
 *       -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
 *       -L$TORCH/lib -Wl,-rpath,$TORCH/lib \
 *       -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 \
 *       -o selfplay_cpp \
 *       go_engine.c mcts.cpp nn_inference.cpp npy_writer.c selfplay_cpp.cpp
 */

#include "nn_inference.h"
#include "mcts.h"
#include "npy_writer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <deque>

/* ── Timing ───────────────────────────────────────────────────────────── */

using Clock = std::chrono::steady_clock;
using Dsec  = std::chrono::duration<double>;

struct Timings {
    double mcts_simulation = 0.0;  /* mcts_init_root + mcts_simulate     */
    double move_selection  = 0.0;  /* mcts_select_move                   */
    double state_copy      = 0.0;  /* board_to_planes + next_state       */
    double winner_check    = 0.0;  /* go_get_winner                      */
    long long total_game_moves = 0; /* total moves across completed games */
    int    completed_games = 0;     /* games counted for move statistics  */
    int    resignations    = 0;    /* self-play games ended by resign    */
    int    resign_disabled_candidates = 0; /* no-resign games that crossed threshold */
    int    resign_false_positives     = 0; /* candidate loser eventually won          */

    void operator+=(const Timings &o) {
        mcts_simulation += o.mcts_simulation;
        move_selection  += o.move_selection;
        state_copy      += o.state_copy;
        winner_check    += o.winner_check;
        total_game_moves += o.total_game_moves;
        completed_games += o.completed_games;
        resignations    += o.resignations;
        resign_disabled_candidates += o.resign_disabled_candidates;
        resign_false_positives     += o.resign_false_positives;
    }
};

/* ── Thread-local NN pointer ─────────────────────────────────────────── */
/*
 * Each worker thread sets tl_nn to point at its own NNInference object
 * before starting games.  nn_callback is then a safe NNEvalFn for that thread.
 */
static thread_local NNInference *tl_nn = nullptr;

static void nn_callback(const float *planes, int batch_size,
                        float *values, float *policies)
{
    tl_nn->eval(planes, batch_size, values, policies);
}

/* ── Configuration ────────────────────────────────────────────────────── */

struct Config {
    std::string model_path;
    int      num_games      = 100;
    int      num_sims       = 800;
    int      batch_size     = 32;
    int      num_threads    = 1;
    bool     use_cuda       = false;
    std::string output_dir  = ".";
    int      temp_moves     = 30;   /* high-temp (explore) moves per game */
    int      max_moves      = 350;  /* force-end if game exceeds this     */
    uint32_t seed           = 42;
    /* Playout cap randomization */
    float    full_prob      = 0.25f; /* fraction of turns that use full search */
    int      fast_sims      = 100;  /* simulation budget for non-training (fast) turns       */
    /* Resign when current player's root Q < -resign_threshold; <0 disables. */
    float    resign_threshold    = -1.0f;
    int      resign_min_move     = 0;
    float    resign_disable_prob = 0.0f;
    float    c_puct              = 1.414f;
    float    dirichlet_alpha     = DIR_ALPHA;
    float    dirichlet_frac      = DIR_FRAC;
};

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s <model_ts.pt> [options]\n"
            "\n"
            "Options:\n"
            "  --games N        total games to play       (default: 100)\n"
            "  --sims  N        MCTS sims per move        (default: 800)\n"
            "  --batch N        MCTS leaf batch size      (default: 32)\n"
            "  --threads N      parallel worker threads   (default: 1)\n"
            "  --cuda           use GPU if available\n"
            "  --output DIR     output directory          (default: .)\n"
            "  --temp-moves N   high-temp moves per game  (default: 15)\n"
            "  --max-moves N    max moves before force end(default: 100)\n"
            "  --seed N         base RNG seed             (default: 42)\n"
            "  --full-prob F    fraction of turns w/ full search (default: 1.0 = disabled)\n"
            "  --fast-sims N    sims for non-training turns       (default: 100)\n"
            "  --resign-threshold F  resign when root Q < -F       (default: off)\n"
            "  --resign-min-move N   earliest move to resign       (default: 0)\n"
            "  --resign-disable-prob F fraction of games with no resignation (default: 0)\n"
            "  --c-puct F            MCTS exploration constant     (default: 1.414)\n"
            "  --dirichlet-alpha F   root noise alpha per action   (default: compiled)\n"
            "  --dirichlet-frac F    root noise mix fraction       (default: 0.25)\n"
            "\n"
            "Output files in DIR:\n"
            "  states.npy    (N, 17, BOARD_SIZE, BOARD_SIZE)  AlphaZero 17-plane repr\n"
            "  policies.npy  (N, ACTION_SIZE)                  MCTS visit probabilities\n"
            "  values.npy    (N,)                              game outcome per position\n",
            prog);
}

static bool parse_args(int argc, char *argv[], Config &cfg)
{
    if (argc < 2) return false;
    cfg.model_path = argv[1];

    for (int i = 2; i < argc; i++) {
        if      (strcmp(argv[i], "--games")     == 0 && i+1 < argc) cfg.num_games   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sims")      == 0 && i+1 < argc) cfg.num_sims    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch")     == 0 && i+1 < argc) cfg.batch_size  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads")   == 0 && i+1 < argc) cfg.num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cuda")      == 0)                cfg.use_cuda    = true;
        else if (strcmp(argv[i], "--output")    == 0 && i+1 < argc) cfg.output_dir  = argv[++i];
        else if (strcmp(argv[i], "--temp-moves")   == 0 && i+1 < argc) cfg.temp_moves = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-moves")    == 0 && i+1 < argc) cfg.max_moves  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")         == 0 && i+1 < argc) cfg.seed       = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--full-prob")    == 0 && i+1 < argc) cfg.full_prob   = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--fast-sims")    == 0 && i+1 < argc) cfg.fast_sims   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--resign-threshold") == 0 && i+1 < argc) cfg.resign_threshold = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--resign-min-move")  == 0 && i+1 < argc) cfg.resign_min_move = atoi(argv[++i]);
        else if (strcmp(argv[i], "--resign-disable-prob") == 0 && i+1 < argc) cfg.resign_disable_prob = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--c-puct") == 0 && i+1 < argc) cfg.c_puct = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--dirichlet-alpha") == 0 && i+1 < argc) cfg.dirichlet_alpha = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--dirichlet-frac") == 0 && i+1 < argc) cfg.dirichlet_frac = (float)atof(argv[++i]);
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    if (cfg.resign_threshold > 1.0f) {
        fprintf(stderr, "--resign-threshold must be <= 1.0, or negative to disable.\n");
        return false;
    }
    if (cfg.resign_min_move < 0) {
        fprintf(stderr, "--resign-min-move must be >= 0.\n");
        return false;
    }
    if (cfg.resign_disable_prob < 0.0f || cfg.resign_disable_prob > 1.0f) {
        fprintf(stderr, "--resign-disable-prob must be in [0, 1].\n");
        return false;
    }
    if (cfg.c_puct <= 0.0f) {
        fprintf(stderr, "--c-puct must be > 0.\n");
        return false;
    }
    if (cfg.dirichlet_alpha < 0.0f) {
        fprintf(stderr, "--dirichlet-alpha must be >= 0.\n");
        return false;
    }
    if (cfg.dirichlet_frac < 0.0f || cfg.dirichlet_frac > 1.0f) {
        fprintf(stderr, "--dirichlet-frac must be in [0, 1].\n");
        return false;
    }
    return true;
}

/* ── Game data structures ─────────────────────────────────────────────── */
// step is one training example
struct Step {
    float planes[17 * NUM_POSITIONS]; /* AlphaZero 17-plane representation */
    float probs[ACTION_SIZE];          /* MCTS visit-count probabilities    */
    int   absolute_player;             /* +1 Black / -1 White               */
};

/* ── Single-game self-play ────────────────────────────────────────────── */

static void play_one_game(NodePool           *pool,
                          const Config       &cfg,
                          std::vector<float> &out_states,
                          std::vector<float> &out_policies,
                          std::vector<float> &out_values,
                          std::mt19937       &coin_rng,
                          Timings            &t)
{
    std::vector<Step> steps;
    steps.reserve((size_t)cfg.max_moves);

    GoState state       = go_initial_state();
    int absolute_player = 1;   /* Black (absolute) moves first */
    int move_count      = 0;
    int resigned_winner = 0;    /* +1 Black / -1 White; 0 means no resign */
    int would_resign_player = 0; /* tracked only in no-resign games */
    bool resignation_configured = (cfg.resign_threshold >= 0.0f);
    bool resignation_enabled = resignation_configured;
    if (resignation_configured && cfg.resign_disable_prob > 0.0f) {
        std::bernoulli_distribution no_resign_dist(cfg.resign_disable_prob);
        resignation_enabled = !no_resign_dist(coin_rng);
    }

    /* History ring: [0]=current, [1]=1 move ago, ..., capped at 8 entries. */
    std::deque<GoState> hist;

    while (!go_game_ended(&state) && move_count < cfg.max_moves) {

        Step step;
        step.absolute_player = absolute_player;

        /* Playout cap randomization: decide full vs fast search for this turn.
         * full_prob=1.0 means every turn is full — feature is disabled. */
        std::bernoulli_distribution full_dist(cfg.full_prob);
        bool is_full_search = full_dist(coin_rng);
        int  sims = is_full_search ? cfg.num_sims : cfg.fast_sims;

        /* 17-plane input with full move history. */
        auto tp = Clock::now();
        hist.push_front(state);
        if ((int)hist.size() > 8) hist.pop_back();
        GoState hist_arr[8];
        int nhist = (int)hist.size();
        for (int h = 0; h < nhist; h++) hist_arr[h] = hist[h];
        go_board_to_planes_17_with_history(hist_arr, nhist, absolute_player, step.planes);
        t.state_copy += Dsec(Clock::now() - tp).count();

        /* MCTS from this position.
         * Dirichlet noise is only added on full-search turns; fast turns run
         * greedier search purely to advance the game state cheaply. */
        auto tm = Clock::now();
        mcts_init_root(pool, &state, absolute_player);
        mcts_set_c_puct(cfg.c_puct);
        mcts_simulate(pool, nn_callback,
                      sims, cfg.batch_size, /*add_noise=*/is_full_search,
                      hist_arr, nhist, cfg.dirichlet_alpha, cfg.dirichlet_frac);
        t.mcts_simulation += Dsec(Clock::now() - tm).count();

        /* Optional AlphaZero-style resignation.
         * Use the root average Q from MCTS, from the current player's
         * perspective.  Only check full-search turns because shallow fast
         * turns are intentionally not trusted as training policy targets. */
        if (resignation_configured && is_full_search && move_count >= cfg.resign_min_move) {
            const Node *root = &pool->nodes[0];
            if (root->visits > 0) {
                float root_q = root->total_value / (float)root->visits;
                if (root_q < -cfg.resign_threshold) {
                    if (resignation_enabled) {
                        resigned_winner = -absolute_player;
                        t.resignations++;
                        break;
                    }
                    if (would_resign_player == 0) {
                        would_resign_player = absolute_player;
                    }
                }
            }
        }

        /* Temperature schedule mirrors selfplay.py (unchanged for both turn types). */
        auto ts = Clock::now();
        float temp = (move_count < cfg.temp_moves) ? 1.0f : 0.0f;
        int action = mcts_select_move(pool, temp, step.probs);
        t.move_selection += Dsec(Clock::now() - ts).count();

        /* Only record full-search turns as training data.
         * Fast-search visit distributions are too shallow to be reliable
         * policy targets, so excluding them improves training signal quality. */
        if (is_full_search) {
            steps.push_back(step);
        }

        /* Advance canonical state; player flips. */
        auto ta = Clock::now();
        state           = go_next_state_canonical(&state, action);
        t.state_copy += Dsec(Clock::now() - ta).count();

        absolute_player = -absolute_player;
        move_count++;
    }

    /*
     * go_get_winner perspective:
     *   absolute_player is who would move next (just flipped at end of loop).
     *   If that player is +1 (Black), the canonical state has Black=+1 which
     *   IS absolute form → perspective=1.
     *   If that player is -1 (White), canonical has White=+1; negate to get
     *   absolute (Black=+1) → perspective=-1.
     * So passing absolute_player directly works for both cases.
     */
    bool force_ended = !go_game_ended(&state) && resigned_winner == 0;
    int winner = resigned_winner;
    if (winner == 0 && !force_ended) {
        auto tw = Clock::now();
        winner = go_get_winner(&state, absolute_player);
        t.winner_check += Dsec(Clock::now() - tw).count();
    }
    if (would_resign_player != 0) {
        t.resign_disabled_candidates++;
        if (winner == would_resign_player) {
            t.resign_false_positives++;
        }
    }
    t.completed_games++;
    t.total_game_moves += move_count;

    /* Force-ended games do not have a reliable terminal result. */
    if (force_ended)
        return;

    /* Append each step to the caller's buffers with winner-derived value. */
    for (const Step &s : steps) {
        float value = (winner == 0) ? 0.0f
                    : (winner == s.absolute_player) ? 1.0f : -1.0f;

        out_states.insert(out_states.end(),
                          s.planes, s.planes + 17 * NUM_POSITIONS);
        out_policies.insert(out_policies.end(),
                            s.probs, s.probs + ACTION_SIZE);
        out_values.push_back(value);
    }
}

/* ── Worker thread ────────────────────────────────────────────────────── */

static void worker(int                  thread_id,
                   int                  num_games,
                   const Config        &cfg,
                   std::vector<float>  &shared_states,
                   std::vector<float>  &shared_policies,
                   std::vector<float>  &shared_values,
                   std::mutex          &out_mutex,
                   Timings             &shared_timings)
{
    /* Each thread loads its own model instance (shared mmap on Linux). */
    NNInference nn(cfg.model_path, cfg.use_cuda);
    tl_nn = &nn;

    /* Distinct RNG stream per thread. */
    mcts_seed_rng(cfg.seed + (uint32_t)thread_id * 1000u);

    /* Separate RNG for the per-move full/fast coin flip — offset from the MCTS
     * Dirichlet seed so the two streams are independent. */
    std::mt19937 coin_rng(cfg.seed + (uint32_t)thread_id * 1000u + 7919u);

    /* Per-thread pool (~44 MB): always on heap, never stack. */
    NodePool *pool = new NodePool;

    std::vector<float> local_states, local_policies, local_values;
    local_states.reserve((size_t)(num_games * 80 * 17 * NUM_POSITIONS));
    local_policies.reserve((size_t)(num_games * 80 * ACTION_SIZE));
    local_values.reserve((size_t)(num_games * 80));

    Timings local_t;

    for (int g = 0; g < num_games; g++) {
        play_one_game(pool, cfg, local_states, local_policies, local_values,
                      coin_rng, local_t);

        /* Overwrite progress file: "games_done\n"
         * Polled by the Python runner to drive per-worker progress bars. */
        {
            std::string pp = cfg.output_dir + "/progress";
            if (FILE *fp = fopen(pp.c_str(), "w")) {
                fprintf(fp, "%d\n", g + 1);
                fclose(fp);
            }
        }

        if ((g + 1) % 10 == 0 || g + 1 == num_games)
            fprintf(stderr, "[thread %d] %d/%d games done\n",
                    thread_id, g + 1, num_games);
    }

    delete pool;

    /* Merge into shared output under lock (one shot, low contention). */
    std::lock_guard<std::mutex> lock(out_mutex);
    shared_states.insert(shared_states.end(),
                         local_states.begin(),   local_states.end());
    shared_policies.insert(shared_policies.end(),
                           local_policies.begin(), local_policies.end());
    shared_values.insert(shared_values.end(),
                         local_values.begin(),   local_values.end());
    shared_timings += local_t;
}

/* ── main ─────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    go_init();

    fprintf(stderr, "=== selfplay_cpp ===\n");
    fprintf(stderr, "Model     : %s\n", cfg.model_path.c_str());
    fprintf(stderr, "Games     : %d\n", cfg.num_games);
    fprintf(stderr, "Sims      : %d\n", cfg.num_sims);
    fprintf(stderr, "Batch     : %d\n", cfg.batch_size);
    fprintf(stderr, "Threads   : %d\n", cfg.num_threads);
    fprintf(stderr, "CUDA      : %s\n", cfg.use_cuda ? "yes" : "no");
    fprintf(stderr, "Output    : %s\n", cfg.output_dir.c_str());
    fprintf(stderr, "TempMoves : %d\n", cfg.temp_moves);
    fprintf(stderr, "MaxMoves  : %d\n", cfg.max_moves);
    fprintf(stderr, "FullProb  : %.3f\n", cfg.full_prob);
    fprintf(stderr, "FastSims  : %d\n",  cfg.fast_sims);
    fprintf(stderr, "C_PUCT    : %.3f\n", cfg.c_puct);
    fprintf(stderr, "DirNoise  : alpha=%.4f frac=%.3f\n",
            cfg.dirichlet_alpha, cfg.dirichlet_frac);
    if (cfg.resign_threshold >= 0.0f) {
        fprintf(stderr, "Resign    : root Q < -%.3f after move %d (disabled in %.1f%% games)\n",
                cfg.resign_threshold, cfg.resign_min_move,
                100.0f * cfg.resign_disable_prob);
    } else {
        fprintf(stderr, "Resign    : off\n");
    }
    fprintf(stderr, "\n");

    std::vector<float> all_states, all_policies, all_values;
    std::mutex out_mutex;
    Timings    all_timings;

    if (cfg.num_threads == 1) {
        /* Single-threaded path avoids std::thread overhead. */
        worker(0, cfg.num_games, cfg,
               all_states, all_policies, all_values, out_mutex, all_timings);
    } else {
        int base      = cfg.num_games / cfg.num_threads;
        int remainder = cfg.num_games % cfg.num_threads;

        std::vector<std::thread> threads;
        threads.reserve((size_t)cfg.num_threads);

        for (int t = 0; t < cfg.num_threads; t++) {
            int n = base + (t < remainder ? 1 : 0);
            threads.emplace_back(worker, t, n, std::cref(cfg),
                                 std::ref(all_states),
                                 std::ref(all_policies),
                                 std::ref(all_values),
                                 std::ref(out_mutex),
                                 std::ref(all_timings));
        }
        for (auto &th : threads)
            th.join();
    }

    /* ── Write output .npy files ─────────────────────────────────────── */

    int N = (int)all_values.size();
    fprintf(stderr, "\nTotal positions: %d\n", N);

    /* states.npy — (N, 17, 9, 9) */
    {
        std::string path = cfg.output_dir + "/states.npy";
        int dims[] = {N, 17, BOARD_SIZE, BOARD_SIZE};
        if (npy_write_float32(path.c_str(), all_states.data(),
                              N * 17 * NUM_POSITIONS, 4, dims) != 0) {
            fprintf(stderr, "ERROR: failed to write %s\n", path.c_str());
            return 1;
        }
        fprintf(stderr, "Wrote: %s\n", path.c_str());
    }

    /* policies.npy — (N, 82) */
    {
        std::string path = cfg.output_dir + "/policies.npy";
        int dims[] = {N, ACTION_SIZE};
        if (npy_write_float32(path.c_str(), all_policies.data(),
                              N * ACTION_SIZE, 2, dims) != 0) {
            fprintf(stderr, "ERROR: failed to write %s\n", path.c_str());
            return 1;
        }
        fprintf(stderr, "Wrote: %s\n", path.c_str());
    }

    /* values.npy — (N,) */
    {
        std::string path = cfg.output_dir + "/values.npy";
        int dims[] = {N};
        if (npy_write_float32(path.c_str(), all_values.data(),
                              N, 1, dims) != 0) {
            fprintf(stderr, "ERROR: failed to write %s\n", path.c_str());
            return 1;
        }
        fprintf(stderr, "Wrote: %s\n", path.c_str());
    }

    /* timing.json — per-phase breakdown and counters for the Python runner */
    {
        std::string path = cfg.output_dir + "/timing.json";
        if (FILE *fp = fopen(path.c_str(), "w")) {
            fprintf(fp, "{\n  \"timings\": {\n");
            fprintf(fp, "    \"mcts_simulation\": %.6f,\n", all_timings.mcts_simulation);
            fprintf(fp, "    \"move_selection\": %.6f,\n",  all_timings.move_selection);
            fprintf(fp, "    \"state_copy\": %.6f,\n",      all_timings.state_copy);
            fprintf(fp, "    \"winner_check\": %.6f\n",      all_timings.winner_check);
            fprintf(fp, "  },\n  \"metrics\": {\n");
            fprintf(fp, "    \"completed_games\": %d,\n",      all_timings.completed_games);
            fprintf(fp, "    \"total_game_moves\": %lld,\n",    all_timings.total_game_moves);
            fprintf(fp, "    \"resignations\": %d,\n",       all_timings.resignations);
            fprintf(fp, "    \"resign_disabled_candidates\": %d,\n",
                    all_timings.resign_disabled_candidates);
            fprintf(fp, "    \"resign_false_positives\": %d\n",
                    all_timings.resign_false_positives);
            fprintf(fp, "  }\n}\n");
            fclose(fp);
            fprintf(stderr, "Wrote: %s\n", path.c_str());
        }
    }

    return 0;
}
