// Precomputed attack tables. Sliding pieces use magic bitboards: the relevant blockers of a square are
// multiplied by a "magic" constant so their top bits index a per-square table of attack sets. The magics
// are found at startup by a seeded random search, so no tables are hard-coded.
#pragma once

#include "types.h"

namespace bb {

constexpr U64 FILE_A = 0x0101010101010101ULL;
constexpr U64 FILE_H = FILE_A << 7;
constexpr U64 RANK_1 = 0xFFULL;
constexpr U64 RANK_8 = RANK_1 << 56;

extern U64 knight_attacks[64];
extern U64 king_attacks[64];
extern U64 pawn_attacks[2][64];  // squares a pawn of that colour on sq attacks

struct Magic {
  U64 mask;
  U64 magic;
  U64* attacks;
  int shift;
  U64 index(U64 occ) const { return ((occ & mask) * magic) >> shift; }
};
extern Magic rook_magics[64];
extern Magic bishop_magics[64];

void init();  // must run once before anything else

inline U64 rook_attacks(int sq, U64 occ) {
  const Magic& m = rook_magics[sq];
  return m.attacks[m.index(occ)];
}
inline U64 bishop_attacks(int sq, U64 occ) {
  const Magic& m = bishop_magics[sq];
  return m.attacks[m.index(occ)];
}
inline U64 queen_attacks(int sq, U64 occ) { return rook_attacks(sq, occ) | bishop_attacks(sq, occ); }

// Slow ray-walking reference, used to build the tables and in tests.
U64 slider_attacks_slow(int sq, U64 occ, bool rook);

}  // namespace bb
