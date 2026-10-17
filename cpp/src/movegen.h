// Move generation. generate() produces pseudo-legal moves (the king may be left in check); a move is
// legal if the mover's king is not attacked after it is made. Castling through or out of check is
// rejected during generation.
#pragma once

#include "position.h"

struct MoveList {
  Move moves[MAX_MOVES];
  int size = 0;
  void add(Move m) { moves[size++] = m; }
  Move* begin() { return moves; }
  Move* end() { return moves + size; }
};

// noisy_only: captures and queen promotions only (for quiescence search).
void generate(const Position& pos, MoveList& list, bool noisy_only = false);
bool is_legal(const Position& pos, Move m);
void generate_legal(const Position& pos, MoveList& list);

// Number of leaf nodes of the legal move tree to `depth` (move-generator test).
U64 perft(const Position& pos, int depth);
