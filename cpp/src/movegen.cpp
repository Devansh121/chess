#include "movegen.h"

namespace {

void add_pawn_moves(MoveList& list, int from, int to, bool capture, bool promotes, bool noisy_only) {
  if (!promotes) {
    list.add(make_move(from, to, capture ? CAPTURE : QUIET));
    return;
  }
  int base = capture ? PROMO_CAPTURE : PROMO;
  list.add(make_move(from, to, base + 3));  // queen
  if (noisy_only) return;
  for (int p = 0; p < 3; p++) list.add(make_move(from, to, base + p));  // knight, bishop, rook
}

void add_piece_moves(MoveList& list, int from, U64 targets, U64 enemies) {
  while (targets) {
    int to = pop_lsb(targets);
    list.add(make_move(from, to, (enemies & bit(to)) ? CAPTURE : QUIET));
  }
}

}  // namespace

void generate(const Position& pos, MoveList& list, bool noisy_only) {
  int us = pos.side, them = us ^ 1;
  U64 occupied = pos.all();
  U64 enemies = pos.occ[them];
  U64 empty = ~occupied;
  U64 targets = noisy_only ? enemies : ~pos.occ[us];

  // Pawns
  int up = us == WHITE ? 8 : -8;
  U64 promo_from = us == WHITE ? 0x00FF000000000000ULL : 0x000000000000FF00ULL;
  U64 start_rank = us == WHITE ? 0x000000000000FF00ULL : 0x00FF000000000000ULL;
  U64 pawns = pos.pieces[us][PAWN];
  while (pawns) {
    int from = pop_lsb(pawns);
    bool promotes = bit(from) & promo_from;
    int to = from + up;
    if (empty & bit(to)) {
      if (!noisy_only || promotes) add_pawn_moves(list, from, to, false, promotes, noisy_only);
      if (!noisy_only && (bit(from) & start_rank) && (empty & bit(to + up)))
        list.add(make_move(from, to + up, DOUBLE_PUSH));
    }
    U64 caps = bb::pawn_attacks[us][from] & enemies;
    while (caps) add_pawn_moves(list, from, pop_lsb(caps), true, promotes, noisy_only);
    if (pos.ep != NO_SQ && (bb::pawn_attacks[us][from] & bit(pos.ep))) list.add(make_move(from, pos.ep, EP_CAPTURE));
  }

  for (U64 b = pos.pieces[us][KNIGHT]; b;) {
    int from = pop_lsb(b);
    add_piece_moves(list, from, bb::knight_attacks[from] & targets, enemies);
  }
  for (U64 b = pos.pieces[us][BISHOP]; b;) {
    int from = pop_lsb(b);
    add_piece_moves(list, from, bb::bishop_attacks(from, occupied) & targets, enemies);
  }
  for (U64 b = pos.pieces[us][ROOK]; b;) {
    int from = pop_lsb(b);
    add_piece_moves(list, from, bb::rook_attacks(from, occupied) & targets, enemies);
  }
  for (U64 b = pos.pieces[us][QUEEN]; b;) {
    int from = pop_lsb(b);
    add_piece_moves(list, from, bb::queen_attacks(from, occupied) & targets, enemies);
  }
  int ksq = pos.king_sq(us);
  add_piece_moves(list, ksq, bb::king_attacks[ksq] & targets, enemies);

  if (noisy_only) return;
  // Castling: path empty, king not in check and not passing through an attacked square.
  // The destination square is checked by the normal legality test.
  int rank = us == WHITE ? 0 : 56;
  int oo = us == WHITE ? WHITE_OO : BLACK_OO, ooo = us == WHITE ? WHITE_OOO : BLACK_OOO;
  if ((pos.castling & oo) && !(occupied & (bit(rank + 5) | bit(rank + 6))) && !pos.attacked(rank + 4, them) &&
      !pos.attacked(rank + 5, them))
    list.add(make_move(rank + 4, rank + 6, KING_CASTLE));
  if ((pos.castling & ooo) && !(occupied & (bit(rank + 1) | bit(rank + 2) | bit(rank + 3))) &&
      !pos.attacked(rank + 4, them) && !pos.attacked(rank + 3, them))
    list.add(make_move(rank + 4, rank + 2, QUEEN_CASTLE));
}

bool is_legal(const Position& pos, Move m) {
  Position next = pos.make(m);
  return !next.attacked(next.king_sq(pos.side), next.side);
}

void generate_legal(const Position& pos, MoveList& list) {
  MoveList all;
  generate(pos, all);
  for (Move m : all)
    if (is_legal(pos, m)) list.add(m);
}

U64 perft(const Position& pos, int depth) {
  MoveList list;
  generate_legal(pos, list);
  if (depth <= 1) return depth == 1 ? list.size : 1;
  U64 nodes = 0;
  for (Move m : list) nodes += perft(pos.make(m), depth - 1);
  return nodes;
}
