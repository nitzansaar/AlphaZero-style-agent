/*
 * gtp_engine.cpp — GTP v2 interface for the AlphaZero-Go engine.
 *
 * Wraps go_engine.c + mcts.cpp + nn_inference.cpp behind stdin/stdout GTP.
 * Single-threaded; one MCTS search per genmove call.
 *
 * Usage:
 *   ./gtp_engine <model_ts.pt> [--sims N] [--batch N] [--cuda]
 *               [--add-noise] [--noise-alpha F] [--noise-frac F]
 *               [--resign-threshold F]
 *
 * Sabaki:
 *   Engine path : /path/to/go/gtp_engine
 *   Arguments   : /path/to/model_ts.pt --sims 800 --cuda
 *
 * Compile:  make gtp_engine
 */

#include "nn_inference.h"
#include "mcts.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/* ── Thread-local NN (single-threaded engine; pointer always valid) ─── */

static thread_local NNInference *tl_nn = nullptr;

static void nn_callback(const float *planes, int batch_size,
                        float *values,  float *policies)
{
    tl_nn->eval(planes, batch_size, values, policies);
}

/* ══════════════════════════════════════════════════════════════════════
 * GTP coordinate helpers
 * ══════════════════════════════════════════════════════════════════════ */

/* Letter → column index (0-based).  'I' is skipped per GTP convention. */
static int col_char_to_idx(char c)
{
    c = (char)toupper((unsigned char)c);
    if (c < 'A' || c > 'T') return -1;
    int idx = c - 'A';
    if (c > 'I') idx--;          /* skip I */
    return (idx >= 0 && idx < BOARD_SIZE) ? idx : -1;
}

/* Column index → letter (0→'A', 7→'H', 8→'J', …). */
static char col_idx_to_char(int idx)
{
    char c = (char)('A' + idx);
    if (c >= 'I') c++;           /* skip I */
    return c;
}

/*
 * GTP vertex string → internal action index.
 *   "pass"   → PASS_ACTION
 *   "resign" → -2  (sentinel)
 *   "D5"     → row * BOARD_SIZE + col
 *   error    → -1
 */
static int gtp_to_action(const std::string &vertex)
{
    if (vertex.empty()) return -1;
    std::string v = vertex;
    for (char &c : v) c = (char)toupper((unsigned char)c);

    if (v == "PASS")   return PASS_ACTION;
    if (v == "RESIGN") return -2;
    if (v.size() < 2)  return -1;

    int col = col_char_to_idx(v[0]);
    if (col < 0) return -1;

    int row_num;
    try { row_num = std::stoi(v.substr(1)); }
    catch (...) { return -1; }

    int row = BOARD_SIZE - row_num;   /* GTP row 1 = our bottom row */
    if (row < 0 || row >= BOARD_SIZE) return -1;
    return row * BOARD_SIZE + col;
}

/* Internal action index → GTP vertex string. */
static std::string action_to_gtp(int action)
{
    if (action == PASS_ACTION) return "pass";
    int row = action / BOARD_SIZE;
    int col = action % BOARD_SIZE;
    return std::string(1, col_idx_to_char(col)) +
           std::to_string(BOARD_SIZE - row);
}

/* "black"/"b" → +1,  "white"/"w" → -1,  otherwise 0. */
static int parse_color(const std::string &s)
{
    std::string l = s;
    for (char &c : l) c = (char)tolower((unsigned char)c);
    if (l == "black" || l == "b") return  1;
    if (l == "white" || l == "w") return -1;
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════
 * Engine state
 * ══════════════════════════════════════════════════════════════════════ */

struct Engine {
    /* Current game position.
     * state is always in canonical form: current player's stones = +1. */
    GoState  state;
    int      player;       /* absolute player to move: +1 Black, -1 White */
    float    komi;

    /* Undo stack — each play/genmove pushes one entry. */
    struct Snapshot { GoState state; int player; };
    std::vector<Snapshot> history;

    /* MCTS settings */
    int   num_sims;
    int   batch_size;
    bool  add_noise;
    float noise_alpha;
    float noise_frac;
    float resign_threshold;   /* resign when root Q < -threshold; <0 = off */

    /* NN and node pool (owned by main) */
    NNInference *nn;
    NodePool    *pool;
};

static void engine_clear(Engine &eng)
{
    eng.state  = go_initial_state();
    eng.player = 1;   /* Black moves first */
    eng.history.clear();
}

static GoState state_oriented_to_player(const GoState &state,
                                        int current_player,
                                        int requested_player)
{
    GoState oriented = state;
    if (requested_player != current_player) {
        for (int i = 0; i < NUM_POSITIONS; i++)
            oriented.board[i] = (int8_t)(-oriented.board[i]);
    }
    return oriented;
}

/* ══════════════════════════════════════════════════════════════════════
 * GTP response helpers
 * ══════════════════════════════════════════════════════════════════════ */

/* GTP success response.  id < 0 means no id was present. */
static void gtp_ok(int id, const std::string &body = "")
{
    if (id >= 0)
        std::cout << '=' << id << (body.empty() ? "" : " ") << body;
    else
        std::cout << "= " << body;
    std::cout << "\n\n";
    std::cout.flush();
}

static void gtp_err(int id, const std::string &msg)
{
    if (id >= 0)
        std::cout << '?' << id << ' ' << msg;
    else
        std::cout << "? " << msg;
    std::cout << "\n\n";
    std::cout.flush();
}

/* ══════════════════════════════════════════════════════════════════════
 * Board display (for showboard)
 * ══════════════════════════════════════════════════════════════════════ */

static std::string board_to_string(const Engine &eng)
{
    /* Convert canonical board to absolute: Black=+1, White=-1. */
    int8_t abs_board[NUM_POSITIONS];
    for (int i = 0; i < NUM_POSITIONS; i++)
        abs_board[i] = (int8_t)(eng.state.board[i] * eng.player);

    std::string out = "\n";

    /* Column labels (top) */
    out += "   ";
    for (int c = 0; c < BOARD_SIZE; c++) {
        out += col_idx_to_char(c);
        out += ' ';
    }
    out += '\n';

    for (int r = 0; r < BOARD_SIZE; r++) {
        char lbl[4];
        int gtp_row = BOARD_SIZE - r;
        snprintf(lbl, sizeof(lbl), "%2d ", gtp_row);
        out += lbl;
        for (int c = 0; c < BOARD_SIZE; c++) {
            int v = abs_board[r * BOARD_SIZE + c];
            out += (v ==  1) ? 'X' : (v == -1) ? 'O' : '.';
            out += ' ';
        }
        out += lbl;
        out += '\n';
    }

    /* Column labels (bottom) */
    out += "   ";
    for (int c = 0; c < BOARD_SIZE; c++) {
        out += col_idx_to_char(c);
        out += ' ';
    }
    return out;
}

/* ══════════════════════════════════════════════════════════════════════
 * Scoring
 * ══════════════════════════════════════════════════════════════════════ */

static std::string compute_final_score(const Engine &eng)
{
    /* Un-canonicalize so Black=+1, White=-1. */
    int8_t abs_board[NUM_POSITIONS];
    for (int i = 0; i < NUM_POSITIONS; i++)
        abs_board[i] = (int8_t)(eng.state.board[i] * eng.player);

    int bs, ws;
    go_count_territory(abs_board, &bs, &ws);
    float diff = (float)bs - ((float)ws + eng.komi);

    char buf[32];
    if      (diff > 0.0f) snprintf(buf, sizeof(buf), "B+%.1f", diff);
    else if (diff < 0.0f) snprintf(buf, sizeof(buf), "W+%.1f", -diff);
    else                  return "0";
    return buf;
}

/* ══════════════════════════════════════════════════════════════════════
 * MCTS move generation
 * ══════════════════════════════════════════════════════════════════════ */

/* Returns action index, or -2 for resign. */
static int run_genmove(Engine &eng)
{
    if (go_game_ended(&eng.state)) return PASS_ACTION;

    /* Build history: [current, 1-ago, 2-ago, ...] newest-first. */
    GoState gtp_hist[8];
    gtp_hist[0] = eng.state;
    int ghlen = 1;
    for (int i = (int)eng.history.size() - 1; i >= 0 && ghlen < 8; i--)
        gtp_hist[ghlen++] = eng.history[i].state;

    mcts_init_root(eng.pool, &eng.state, eng.player);
    mcts_simulate(eng.pool, nn_callback,
                  eng.num_sims, eng.batch_size, eng.add_noise,
                  gtp_hist, ghlen,
                  eng.noise_alpha, eng.noise_frac);

    /* Optional resign check on root Q. */
    if (eng.resign_threshold >= 0.0f) {
        const Node *root = &eng.pool->nodes[0];
        if (root->visits > 0) {
            float Q = root->total_value / (float)root->visits;
            if (Q < -eng.resign_threshold) return -2;
        }
    }

    float probs[ACTION_SIZE];
    return mcts_select_move(eng.pool, /*temperature=*/0.0f, probs);
}

/* ══════════════════════════════════════════════════════════════════════
 * GTP command dispatch
 * ══════════════════════════════════════════════════════════════════════ */

static void handle_command(Engine                          &eng,
                           int                              id,
                           const std::string               &cmd,
                           const std::vector<std::string>  &args)
{
    /* ── Administrative ─────────────────────────────────────────────── */
    if (cmd == "protocol_version") {
        gtp_ok(id, "2");

    } else if (cmd == "name") {
        gtp_ok(id, "AlphaZero-Go");

    } else if (cmd == "version") {
        gtp_ok(id, "1.0");

    } else if (cmd == "list_commands") {
        gtp_ok(id,
            "protocol_version\nname\nversion\nlist_commands\nknown_command\nquit\n"
            "boardsize\nclear_board\nkomi\nplay\ngenmove\nreg_genmove\nundo\n"
            "showboard\nfinal_score\ntime_settings\ntime_left");

    } else if (cmd == "known_command") {
        if (args.empty()) { gtp_err(id, "missing argument"); return; }
        static const char *known[] = {
            "protocol_version", "name", "version", "list_commands",
            "known_command", "quit", "boardsize", "clear_board", "komi",
            "play", "genmove", "reg_genmove", "undo", "showboard",
            "final_score", "time_settings", "time_left", nullptr
        };
        bool found = false;
        for (int i = 0; known[i]; i++)
            if (args[0] == known[i]) { found = true; break; }
        gtp_ok(id, found ? "true" : "false");

    } else if (cmd == "quit") {
        gtp_ok(id);
        exit(0);

    /* ── Board setup ────────────────────────────────────────────────── */
    } else if (cmd == "boardsize") {
        if (args.empty()) { gtp_err(id, "missing argument"); return; }
        int sz;
        try { sz = std::stoi(args[0]); } catch (...) { gtp_err(id, "invalid size"); return; }
        if (sz != BOARD_SIZE) { gtp_err(id, "unacceptable size"); return; }
        engine_clear(eng);
        gtp_ok(id);

    } else if (cmd == "clear_board") {
        engine_clear(eng);
        gtp_ok(id);

    } else if (cmd == "komi") {
        if (args.empty()) { gtp_err(id, "missing argument"); return; }
        try { eng.komi = std::stof(args[0]); } catch (...) { gtp_err(id, "invalid komi"); return; }
        gtp_ok(id);

    /* ── Move commands ──────────────────────────────────────────────── */
    } else if (cmd == "play") {
        if (args.size() < 2) { gtp_err(id, "missing arguments"); return; }
        int color = parse_color(args[0]);
        if (color == 0) { gtp_err(id, "invalid color"); return; }
        int action = gtp_to_action(args[1]);
        if (action == -1) { gtp_err(id, "invalid vertex"); return; }
        if (action == -2) { gtp_err(id, "resign is not a valid play move"); return; }

        GoState oriented = state_oriented_to_player(eng.state, eng.player, color);

        eng.history.push_back({eng.state, eng.player});
        eng.state  = go_next_state_canonical(&oriented, action);
        eng.player = -color;
        gtp_ok(id);

    } else if (cmd == "genmove" || cmd == "reg_genmove") {
        if (args.empty()) { gtp_err(id, "missing color"); return; }
        int color = parse_color(args[0]);
        if (color == 0) { gtp_err(id, "invalid color"); return; }

        GoState before_state = eng.state;
        int before_player = eng.player;
        eng.state = state_oriented_to_player(eng.state, eng.player, color);
        eng.player = color;

        int action = run_genmove(eng);

        if (action == -2) {
            /* Resign: record the move but don't update the board state. */
            if (cmd == "genmove")
                eng.history.push_back({before_state, before_player});
            else {
                eng.state = before_state;
                eng.player = before_player;
            }
            gtp_ok(id, "resign");
            return;
        }

        if (cmd == "genmove") {
            eng.history.push_back({before_state, before_player});
            eng.state  = go_next_state_canonical(&eng.state, action);
            eng.player = -eng.player;
        } else {
            eng.state = before_state;
            eng.player = before_player;
        }
        /* reg_genmove returns the move without playing it. */
        gtp_ok(id, action_to_gtp(action));

    } else if (cmd == "undo") {
        if (eng.history.empty()) { gtp_err(id, "cannot undo"); return; }
        eng.state  = eng.history.back().state;
        eng.player = eng.history.back().player;
        eng.history.pop_back();
        gtp_ok(id);

    /* ── Query commands ─────────────────────────────────────────────── */
    } else if (cmd == "showboard") {
        gtp_ok(id, board_to_string(eng));

    } else if (cmd == "final_score") {
        gtp_ok(id, compute_final_score(eng));

    /* ── Time management (no-ops: MCTS is simulation-bounded) ────────── */
    } else if (cmd == "time_settings" || cmd == "time_left") {
        gtp_ok(id);

    } else {
        gtp_err(id, "unknown command");
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Main GTP loop
 * ══════════════════════════════════════════════════════════════════════ */

static void gtp_loop(Engine &eng)
{
    std::string line;
    while (std::getline(std::cin, line)) {
        /* Strip comments and trailing whitespace. */
        auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        while (!line.empty() &&
               (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (line.empty()) continue;

        /* Tokenize. */
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string tok;
        while (iss >> tok) tokens.push_back(tok);
        if (tokens.empty()) continue;

        /* Extract optional numeric ID prefix. */
        int id = -1;
        size_t cmd_start = 0;
        if (!tokens.empty() && !tokens[0].empty() &&
            std::isdigit((unsigned char)tokens[0][0])) {
            try {
                id = std::stoi(tokens[0]);
                cmd_start = 1;
            } catch (...) { /* not an id */ }
        }
        if (cmd_start >= tokens.size()) continue;

        std::string cmd = tokens[cmd_start];
        for (char &c : cmd) c = (char)tolower((unsigned char)c);

        std::vector<std::string> args(tokens.begin() + (int)cmd_start + 1,
                                      tokens.end());

        handle_command(eng, id, cmd, args);
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Argument parsing & main
 * ══════════════════════════════════════════════════════════════════════ */

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s <model_ts.pt> [options]\n"
        "\n"
        "Options:\n"
        "  --sims  N              MCTS simulations per move  (default: 800)\n"
        "  --batch N              NN leaf-batch size          (default: 32)\n"
        "  --cuda                 use GPU for NN inference\n"
        "  --add-noise            add Dirichlet noise to root priors\n"
        "  --noise-alpha F        Dirichlet alpha             (default: 0.03*BOARD_SIZE)\n"
        "  --noise-frac F         root prior noise fraction   (default: 0.25)\n"
        "  --resign-threshold F   resign when root Q < -F    (default: off)\n"
        "\n"
        "Communicates via GTP v2 on stdin/stdout.\n"
        "Point Sabaki to this binary; pass the .pt path as the first argument.\n",
        prog);
}

int main(int argc, char *argv[])
{
    if (argc < 2) { usage(argv[0]); return 1; }

    /* ── Default config ─────────────────────────────────────────────── */
    Engine eng;
    eng.num_sims         = 800;
    eng.batch_size       = 32;
    eng.add_noise        = false;
    eng.noise_alpha      = DIR_ALPHA;
    eng.noise_frac       = DIR_FRAC;
    eng.resign_threshold = -1.0f;   /* disabled */
    eng.komi             = KOMI;
    eng.nn               = nullptr;
    eng.pool             = nullptr;
    bool use_cuda        = false;
    std::string model_path = argv[1];

    /* ── Parse flags ────────────────────────────────────────────────── */
    for (int i = 2; i < argc; i++) {
        if      (strcmp(argv[i], "--sims")  == 0 && i+1 < argc)
            eng.num_sims   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i+1 < argc)
            eng.batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cuda")  == 0)
            use_cuda = true;
        else if (strcmp(argv[i], "--add-noise") == 0)
            eng.add_noise = true;
        else if (strcmp(argv[i], "--noise-alpha") == 0 && i+1 < argc) {
            eng.noise_alpha = (float)atof(argv[++i]);
            eng.add_noise = true;
        }
        else if (strcmp(argv[i], "--noise-frac") == 0 && i+1 < argc) {
            eng.noise_frac = (float)atof(argv[++i]);
            eng.add_noise = true;
        }
        else if (strcmp(argv[i], "--resign-threshold") == 0 && i+1 < argc)
            eng.resign_threshold = (float)atof(argv[++i]);
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* ── Initialise ─────────────────────────────────────────────────── */
    go_init();
    engine_clear(eng);

    NNInference nn(model_path, use_cuda);
    tl_nn  = &nn;
    eng.nn = &nn;

    eng.pool = new NodePool;

    /* Log config to stderr (doesn't interfere with GTP on stdout). */
    fprintf(stderr, "=== AlphaZero-Go GTP engine ===\n");
    fprintf(stderr, "Model : %s\n", model_path.c_str());
    fprintf(stderr, "Sims  : %d\n", eng.num_sims);
    fprintf(stderr, "Batch : %d\n", eng.batch_size);
    fprintf(stderr, "Noise : %s\n", eng.add_noise ? "yes" : "no");
    if (eng.add_noise)
        fprintf(stderr, "Noise params: alpha=%.4f frac=%.4f\n",
                eng.noise_alpha, eng.noise_frac);
    fprintf(stderr, "CUDA  : %s\n", use_cuda ? "yes" : "no");
    if (eng.resign_threshold >= 0.0f)
        fprintf(stderr, "Resign: Q < -%.2f\n", eng.resign_threshold);
    fprintf(stderr, "================================\n");

    gtp_loop(eng);

    delete eng.pool;
    return 0;
}
