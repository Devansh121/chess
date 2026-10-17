#include "eval.h"

#include <algorithm>
#include <cstdlib>

namespace eval {

namespace {

constexpr int PHASE_WEIGHT[6] = {0, 1, 1, 2, 4, 0};
constexpr int MAX_PHASE = 24;
constexpr int BISHOP_PAIR = 30;

int mg_table[6][64];  // from White's side, a1 = 0
int eg_table[6][64];

// 0 on the corners .. 6 on the four centre squares (always an integer).
int centrality(int sq) {
  int f2 = std::abs(2 * file_of(sq) - 7), r2 = std::abs(2 * rank_of(sq) - 7);  // twice the distance from centre
  return 7 - (f2 + r2) / 2;
}

int floor_div(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

}  // namespace

void init() {
  static bool done = false;
  if (done) return;
  done = true;
  const int pawn_mg_rank[8] = {0, 0, 2, 6, 14, 30, 50, 0};
  const int pawn_eg_rank[8] = {0, 0, 5, 15, 30, 55, 90, 0};
  const int king_home_file[8] = {10, 20, 5, -5, -5, 5, 20, 10};
  for (int sq = 0; sq < 64; sq++) {
    int f = file_of(sq), r = rank_of(sq), c = centrality(sq);
    bool centre_file = f == 3 || f == 4;
    mg_table[PAWN][sq] = pawn_mg_rank[r] + (centre_file && (r == 3 || r == 4) ? 10 : 0);
    eg_table[PAWN][sq] = pawn_eg_rank[r];
    mg_table[KNIGHT][sq] = -30 + 8 * c;
    eg_table[KNIGHT][sq] = -25 + 6 * c;
    mg_table[BISHOP][sq] = -10 + 4 * c;
    eg_table[BISHOP][sq] = -8 + 3 * c;
    mg_table[ROOK][sq] = (r == 6 ? 15 : 0) + (centre_file ? 5 : 0);
    eg_table[ROOK][sq] = r == 6 ? 5 : 0;
    mg_table[QUEEN][sq] = -5 + 2 * c;
    eg_table[QUEEN][sq] = -10 + 3 * c;
    mg_table[KING][sq] = r == 0 ? king_home_file[f] : -25 * std::min(r, 3);
    eg_table[KING][sq] = -18 + 6 * c;
  }
}

int evaluate_white(const Position& pos) {
  int mg = 0, eg = 0, phase = 0;
  for (int color = WHITE; color <= BLACK; color++) {
    int sign = color == WHITE ? 1 : -1;
    for (int type = PAWN; type <= KING; type++) {
      U64 b = pos.pieces[color][type];
      phase += PHASE_WEIGHT[type] * popcount(b);
      while (b) {
        int sq = pop_lsb(b);
        int idx = color == WHITE ? sq : sq ^ 56;
        mg += sign * (PIECE_VALUE[type] + mg_table[type][idx]);
        eg += sign * (PIECE_VALUE[type] + eg_table[type][idx]);
      }
    }
  }
  phase = std::min(phase, MAX_PHASE);
  int score = floor_div(mg * phase + eg * (MAX_PHASE - phase), MAX_PHASE);
  if (popcount(pos.pieces[WHITE][BISHOP]) >= 2) score += BISHOP_PAIR;
  if (popcount(pos.pieces[BLACK][BISHOP]) >= 2) score -= BISHOP_PAIR;
  return score;
}

int evaluate(const Position& pos) {
  int s = evaluate_white(pos);
  return pos.side == WHITE ? s : -s;
}

}  // namespace eval
