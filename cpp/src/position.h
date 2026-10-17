// Board state. Positions are small (~200 bytes) and updated by copy-make: make() returns a new
// position, so there is no unmake to get wrong. The hash is a Zobrist key updated incrementally.
#pragma once

#include <string>

#include "bitboard.h"
#include "types.h"

// Castling rights bits.
constexpr int WHITE_OO = 1, WHITE_OOO = 2, BLACK_OO = 4, BLACK_OOO = 8;

struct Position {
  U64 pieces[2][6];
  U64 occ[2];
  uint8_t board[64];  // piece code per square, NO_PIECE if empty
  int side;
  int castling;
  int ep;  // en passant target square, NO_SQ if none (only set when a capture is actually possible)
  int halfmove;
  int fullmove;
  U64 hash;

  static void init();  // attack tables and Zobrist keys; call once at startup

  bool set_fen(const std::string& fen);
  std::string fen() const;

  U64 all() const { return occ[WHITE] | occ[BLACK]; }
  int king_sq(int color) const { return lsb(pieces[color][KING]); }
  bool attacked(int sq, int by) const;
  // All pieces of both colours attacking `sq`, with `occupied` as the blockers (for exchange evaluation).
  U64 attackers_to(int sq, U64 occupied) const;
  bool in_check() const { return attacked(king_sq(side), side ^ 1); }
  bool has_non_pawn_material(int color) const {
    return pieces[color][KNIGHT] | pieces[color][BISHOP] | pieces[color][ROOK] | pieces[color][QUEEN];
  }

  Position make(Move m) const;
  Position make_null() const;  // pass the turn (for null-move pruning)

  // Recompute the hash from scratch (tests check it matches the incremental one).
  U64 compute_hash() const;

 private:
  void clear();
  void put(int code, int sq);
  void remove(int sq);
};

namespace zobrist {
extern U64 piece_sq[12][64];
extern U64 castling[16];
extern U64 ep_file[8];
extern U64 side;
}  // namespace zobrist

std::string square_name(int sq);
std::string move_to_uci(Move m);
