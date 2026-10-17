// Basic types shared by the whole engine: bitboards, squares, pieces and the 16-bit move encoding.
#pragma once

#include <cstdint>

using U64 = uint64_t;

enum Color : int { WHITE = 0, BLACK = 1 };
enum PieceType : int { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING };

// Piece code = color * 6 + type; NO_PIECE marks an empty square in the mailbox.
constexpr int NO_PIECE = 12;
constexpr int NO_SQ = 64;
constexpr int MAX_MOVES = 256;
constexpr int MAX_PLY = 128;

inline int piece_code(int color, int type) { return color * 6 + type; }
inline int piece_color(int code) { return code / 6; }
inline int piece_type(int code) { return code % 6; }

// Squares: a1 = 0, b1 = 1, ..., h8 = 63.
inline int file_of(int sq) { return sq & 7; }
inline int rank_of(int sq) { return sq >> 3; }
inline U64 bit(int sq) { return 1ULL << sq; }

inline int lsb(U64 b) { return __builtin_ctzll(b); }
inline int popcount(U64 b) { return __builtin_popcountll(b); }
inline int pop_lsb(U64& b) {
  int s = lsb(b);
  b &= b - 1;
  return s;
}

// Move: bits 0-5 from, 6-11 to, 12-15 flag.
// Flags: bit 2 = capture, bit 3 = promotion; promotion piece = KNIGHT + (flag & 3).
using Move = uint16_t;
enum MoveFlag : int {
  QUIET = 0,
  DOUBLE_PUSH = 1,
  KING_CASTLE = 2,
  QUEEN_CASTLE = 3,
  CAPTURE = 4,
  EP_CAPTURE = 5,
  PROMO = 8,           // + 0..3 for N, B, R, Q
  PROMO_CAPTURE = 12,  // + 0..3 for N, B, R, Q
};
constexpr Move NO_MOVE = 0;  // a1a1 is never a legal move

inline Move make_move(int from, int to, int flag) { return Move(from | (to << 6) | (flag << 12)); }
inline int move_from(Move m) { return m & 63; }
inline int move_to(Move m) { return (m >> 6) & 63; }
inline int move_flag(Move m) { return m >> 12; }
inline bool is_capture(Move m) { return move_flag(m) & 4; }
inline bool is_promotion(Move m) { return move_flag(m) & 8; }
inline int promotion_type(Move m) { return KNIGHT + (move_flag(m) & 3); }
