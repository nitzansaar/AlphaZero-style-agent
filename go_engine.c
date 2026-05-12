#include "go_engine.h"

#include <string.h>
#include <stdio.h>

/* ── Neighbour lookup table ───────────────────────────────────────────── */

static int  NEIGHBOR_TABLE[NUM_POSITIONS][4];
static int  NEIGHBOR_COUNT[NUM_POSITIONS];
static int  initialized = 0;

void go_init(void)
{
    if (initialized) return;

    for (int idx = 0; idx < NUM_POSITIONS; idx++) {
        int row   = idx / BOARD_SIZE;
        int col   = idx % BOARD_SIZE;
        int count = 0;

        if (row > 0)             NEIGHBOR_TABLE[idx][count++] = (row-1)*BOARD_SIZE + col;
        if (row < BOARD_SIZE-1)  NEIGHBOR_TABLE[idx][count++] = (row+1)*BOARD_SIZE + col;
        if (col > 0)             NEIGHBOR_TABLE[idx][count++] = row*BOARD_SIZE + (col-1);
        if (col < BOARD_SIZE-1)  NEIGHBOR_TABLE[idx][count++] = row*BOARD_SIZE + (col+1);

        NEIGHBOR_COUNT[idx] = count;
    }

    initialized = 1;
}

GoState go_initial_state(void)
{
    GoState s;
    memset(s.board, 0, NUM_POSITIONS);
    s.ko_point           = -1;
    s.consecutive_passes = 0;
    return s;
}

int go_get_neighbors(int idx, int neighbors_out[4])
{
    int n = NEIGHBOR_COUNT[idx];
    for (int i = 0; i < n; i++)
        neighbors_out[i] = NEIGHBOR_TABLE[idx][i];
    return n;
}

/* ── Group / liberty finding ──────────────────────────────────────────── */

/*
 * Iterative DFS.  Uses three fixed-size flag arrays so there are no heap
 * allocations.  Each cell is pushed onto the stack at most once (guarded
 * by in_stack[]), so the stack never exceeds NUM_POSITIONS entries.
 */
int go_find_group(const int8_t *board, int idx,
                  int *group_out, int *group_size_out)
{
    if (board[idx] == 0) {
        *group_size_out = 0;
        return 0;
    }

    int8_t color = board[idx];

    /* in_stack[] prevents any cell from being pushed more than once, so
     * every popped cell is fresh — no secondary visited guard needed. */
    int8_t liberty_seen[NUM_POSITIONS];
    int8_t in_stack[NUM_POSITIONS];
    memset(liberty_seen,  0, NUM_POSITIONS);
    memset(in_stack,      0, NUM_POSITIONS);

    int stack[NUM_POSITIONS];
    int stack_top = 0;
    stack[stack_top++] = idx;
    in_stack[idx] = 1;

    int group_size   = 0;
    int liberty_count = 0;

    while (stack_top > 0) {
        int current = stack[--stack_top];
        group_out[group_size++] = current;

        int n = NEIGHBOR_COUNT[current];
        for (int i = 0; i < n; i++) {
            int nb = NEIGHBOR_TABLE[current][i];

            if (board[nb] == 0) {
                if (!liberty_seen[nb]) {
                    liberty_seen[nb] = 1;
                    liberty_count++;
                }
            } else if (board[nb] == color && !in_stack[nb]) {
                in_stack[nb] = 1;
                stack[stack_top++] = nb;
            }
        }
    }

    *group_size_out = group_size;
    return liberty_count;
}

/* ── Capture ──────────────────────────────────────────────────────────── */
/*
scans board for opponent groups with zero liberties and removes them
*/

int go_capture_dead_stones(int8_t *board, int player)
{
    int8_t opponent = (int8_t)(-player);
    int    captured  = 0;

    int8_t visited[NUM_POSITIONS];
    memset(visited, 0, NUM_POSITIONS);

    int group[NUM_POSITIONS];
    int group_size;

    for (int idx = 0; idx < NUM_POSITIONS; idx++) {
        if (board[idx] != opponent || visited[idx]) continue;

        int liberties = go_find_group(board, idx, group, &group_size);

        for (int i = 0; i < group_size; i++)
            visited[group[i]] = 1;

        if (liberties == 0) {
            for (int i = 0; i < group_size; i++)
                board[group[i]] = 0;
            captured += group_size;
        }
    }

    return captured;
}

/* ── Suicide check ────────────────────────────────────────────────────── */

int go_is_suicide(const int8_t *board, int idx, int player)
{
    /* Place the stone on a temporary copy. */
    int8_t test[NUM_POSITIONS];
    memcpy(test, board, NUM_POSITIONS);
    test[idx] = (int8_t)player;

    int8_t opponent = (int8_t)(-player);

    /* If placing the stone captures at least one opponent group, it is
       not suicide regardless of own liberties. */
    int n = NEIGHBOR_COUNT[idx];
    for (int i = 0; i < n; i++) {
        int nb = NEIGHBOR_TABLE[idx][i];
        if (test[nb] == opponent) {
            int group[NUM_POSITIONS], gs;
            int liberties = go_find_group(test, nb, group, &gs);
            if (liberties == 0) return 0;   /* captures → not suicide */
        }
    }

    /* Check whether our own group has at least one liberty. */
    int group[NUM_POSITIONS], gs;
    int liberties = go_find_group(test, idx, group, &gs);
    return (liberties == 0) ? 1 : 0;
}

/* ── Ko check ────────────────────────────────────────────────────────── */

int go_would_be_ko(int action, int ko_point)
{
    return (action == ko_point) ? 1 : 0;
}

/* ── Validity ─────────────────────────────────────────────────────────── */

int go_is_valid_move(const GoState *state, int action, int player)
{
    if (action == PASS_ACTION)               return 1;
    if (action < 0 || action >= NUM_POSITIONS) return 0;
    if (state->board[action] != 0)           return 0;
    if (go_would_be_ko(action, state->ko_point)) return 0;
    if (go_is_suicide(state->board, action, player)) return 0;
    return 1;
}

void go_get_valid_moves(const GoState *state, int player, float *action_mask)
{
    for (int i = 0; i < ACTION_SIZE; i++)
        action_mask[i] = 0.0f;

    for (int a = 0; a < NUM_POSITIONS; a++)
        if (go_is_valid_move(state, a, player))
            action_mask[a] = 1.0f;

    action_mask[PASS_ACTION] = 1.0f;   /* pass is always legal */
}

/* ── Apply move ───────────────────────────────────────────────────────── */

GoState go_apply_move(const GoState *state, int action, int player)
{
    GoState ns = *state;   /* copy — board is a value type here */

    if (action == PASS_ACTION) {
        ns.ko_point           = -1;
        ns.consecutive_passes = state->consecutive_passes + 1;
        return ns;
    }

    ns.consecutive_passes = 0;
    ns.board[action] = (int8_t)player;

    int captured = go_capture_dead_stones(ns.board, player);

    /* Detect ko: exactly one stone captured, the capturing stone is a
       singleton with exactly one liberty (the just-vacated cell). */
    int ko_point = -1;
    if (captured == 1) {
        int group[NUM_POSITIONS], gs;
        int liberties = go_find_group(ns.board, action, group, &gs);
        if (gs == 1 && liberties == 1) {
            int nb_count = NEIGHBOR_COUNT[action];
            for (int i = 0; i < nb_count; i++) {
                int nb = NEIGHBOR_TABLE[action][i];
                if (ns.board[nb] == 0) {
                    ko_point = nb;
                    break;
                }
            }
        }
    }

    ns.ko_point = ko_point;
    return ns;
}

GoState go_next_state_canonical(const GoState *state, int action)
{
    /* In canonical form the current player is always +1. */
    GoState ns = go_apply_move(state, action, 1);

    /* Flip the board so the next player sees themselves as +1. */
    for (int i = 0; i < NUM_POSITIONS; i++)
        ns.board[i] = (int8_t)(-ns.board[i]);

    return ns;
}

/* ── Territory scoring ────────────────────────────────────────────────── */

/*
 * BFS over empty cells only.  Stones are pre-marked as visited so the
 * main loop never tries to start a region from a stone position.
 * When expanding an empty cell, stone neighbours update the border flags
 * but are not pushed onto the stack.
 */
void go_count_territory(const int8_t *board,
                        int *black_score_out, int *white_score_out)
{
    int bs = 0, ws = 0;

    /* Count stones directly. */
    for (int i = 0; i < NUM_POSITIONS; i++) {
        if      (board[i] ==  1) bs++;
        else if (board[i] == -1) ws++;
    }

    /* Mark stone positions so they are never BFS start points. */
    int8_t visited[NUM_POSITIONS];
    memset(visited, 0, NUM_POSITIONS);
    for (int i = 0; i < NUM_POSITIONS; i++)
        if (board[i] != 0) visited[i] = 1;

    int stack[NUM_POSITIONS];

    for (int start = 0; start < NUM_POSITIONS; start++) {
        if (visited[start]) continue;

        int  region_size  = 0;
        int  borders_black = 0, borders_white = 0;
        int  stack_top    = 0;

        stack[stack_top++] = start;
        visited[start]     = 1;

        while (stack_top > 0) {
            int cur = stack[--stack_top];
            region_size++;

            int nc = NEIGHBOR_COUNT[cur];
            for (int i = 0; i < nc; i++) {
                int nb = NEIGHBOR_TABLE[cur][i];
                if      (board[nb] ==  1) borders_black = 1;
                else if (board[nb] == -1) borders_white = 1;
                else if (!visited[nb]) {
                    visited[nb]        = 1;
                    stack[stack_top++] = nb;
                }
            }
        }

        if      (borders_black && !borders_white) bs += region_size;
        else if (borders_white && !borders_black) ws += region_size;
        /* Neutral / dame if borders both or neither — not counted. */
    }

    *black_score_out = bs;
    *white_score_out = ws;
}

/* ── Game-end detection ───────────────────────────────────────────────── */

int go_game_ended(const GoState *state)
{
    return (state->consecutive_passes >= 2) ? 1 : 0;
}

int go_get_winner(const GoState *state, int perspective)
{
    int8_t board[NUM_POSITIONS];
    memcpy(board, state->board, NUM_POSITIONS);

    /* If the state is in canonical form (perspective == -1) flip it back
       to absolute form so that 1 == black and -1 == white. */
    if (perspective == -1)
        for (int i = 0; i < NUM_POSITIONS; i++)
            board[i] = (int8_t)(-board[i]);

    int bs, ws;
    go_count_territory(board, &bs, &ws);

    float white_total = (float)ws + KOMI;

    if ((float)bs > white_total) return  1;
    if (white_total > (float)bs) return -1;
    return 0;
}

/* ── Neural-network input planes ─────────────────────────────────────── */

void go_board_to_planes_17_with_history(const GoState *states, int n_states,
                                         int absolute_player, float *planes_out)
{
    int cap = (n_states < 8) ? n_states : 8;
    for (int i = 0; i < 17 * NUM_POSITIONS; i++) planes_out[i] = 0.0f;

    for (int h = 0; h < cap; h++) {
        /* At slot h, canonical perspective has flipped h times.
         * Current player's stones are +1 for even h, -1 for odd h. */
        int sign = (h % 2 == 0) ? 1 : -1;
        float *cur = planes_out + h       * NUM_POSITIONS;
        float *opp = planes_out + (h + 8) * NUM_POSITIONS;
        for (int i = 0; i < NUM_POSITIONS; i++) {
            int v  = (int)states[h].board[i] * sign;
            cur[i] = (v ==  1) ? 1.0f : 0.0f;
            opp[i] = (v == -1) ? 1.0f : 0.0f;
        }
    }

    float color_val = (absolute_player == 1) ? 1.0f : 0.0f;
    float *p16 = planes_out + 16 * NUM_POSITIONS;
    for (int i = 0; i < NUM_POSITIONS; i++) p16[i] = color_val;
}

/* ── Debug render ─────────────────────────────────────────────────────── */

void go_render(const GoState *state)
{
    printf("  ");
    for (int col = 0; col < BOARD_SIZE; col++) printf("%2d", col);
    printf("\n");

    for (int row = 0; row < BOARD_SIZE; row++) {
        printf("%2d", row);
        for (int col = 0; col < BOARD_SIZE; col++) {
            int idx = row * BOARD_SIZE + col;
            char c = '.';
            if      (state->board[idx] ==  1) c = 'X';
            else if (state->board[idx] == -1) c = 'O';
            printf(" %c", c);
        }
        printf("\n");
    }

    printf("ko=%d  passes=%d\n", state->ko_point, state->consecutive_passes);
}
