#include "bitboard.h"

#include <cstdlib>
#include <vector>

namespace bb {

U64 knight_attacks[64];
U64 king_attacks[64];
U64 pawn_attacks[2][64];
Magic rook_magics[64];
Magic bishop_magics[64];

namespace {

std::vector<U64> rook_table;
std::vector<U64> bishop_table;

// Squares reached by stepping (df, dr) from sq, if still on the board.
U64 step(int sq, int df, int dr) {
  int f = file_of(sq) + df, r = rank_of(sq) + dr;
  return (f >= 0 && f < 8 && r >= 0 && r < 8) ? bit(r * 8 + f) : 0;
}

const int ROOK_DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
const int BISHOP_DIRS[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

// Blockers that matter for a slider on sq: its rays, minus the last square of each ray
// (a piece on the edge can't block anything further).
U64 relevant_mask(int sq, bool rook) {
  U64 mask = 0;
  for (auto& d : rook ? ROOK_DIRS : BISHOP_DIRS) {
    int f = file_of(sq) + d[0], r = rank_of(sq) + d[1];
    while (f + d[0] >= 0 && f + d[0] < 8 && r + d[1] >= 0 && r + d[1] < 8) {
      mask |= bit(r * 8 + f);
      f += d[0];
      r += d[1];
    }
  }
  return mask;
}

struct Rng {
  U64 s;
  U64 next() {  // xorshift64*
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ULL;
  }
  U64 sparse() { return next() & next() & next(); }  // few set bits make good magic candidates
};

void init_magics(bool rook, Magic* magics, std::vector<U64>& table, Rng& rng) {
  // Size the shared table first so per-square pointers stay valid.
  size_t total = 0;
  for (int sq = 0; sq < 64; sq++) total += size_t(1) << popcount(relevant_mask(sq, rook));
  table.assign(total, 0);

  size_t offset = 0;
  std::vector<U64> occs, atts, used;
  for (int sq = 0; sq < 64; sq++) {
    U64 mask = relevant_mask(sq, rook);
    int bits = popcount(mask);
    size_t size = size_t(1) << bits;

    // Every subset of the mask (carry-rippler enumeration) with its true attack set.
    occs.clear();
    atts.clear();
    U64 sub = 0;
    do {
      occs.push_back(sub);
      atts.push_back(slider_attacks_slow(sq, sub, rook));
      sub = (sub - mask) & mask;
    } while (sub);

    Magic& m = magics[sq];
    m.mask = mask;
    m.shift = 64 - bits;
    m.attacks = &table[offset];
    used.assign(size, 0);
    for (;;) {
      U64 magic = rng.sparse();
      if (popcount((mask * magic) & 0xFF00000000000000ULL) < 6) continue;  // quick reject of weak candidates
      m.magic = magic;
      std::fill(used.begin(), used.end(), 0);
      std::fill(m.attacks, m.attacks + size, 0);
      bool ok = true;
      for (size_t i = 0; i < occs.size() && ok; i++) {
        U64 idx = m.index(occs[i]);
        // Two blocker sets may share a slot only if they give the same attacks.
        if (!used[idx]) {
          used[idx] = 1;
          m.attacks[idx] = atts[i];
        } else if (m.attacks[idx] != atts[i]) {
          ok = false;
        }
      }
      if (ok) break;
    }
    offset += size;
  }
}

}  // namespace

U64 slider_attacks_slow(int sq, U64 occ, bool rook) {
  U64 att = 0;
  for (auto& d : rook ? ROOK_DIRS : BISHOP_DIRS) {
    int f = file_of(sq) + d[0], r = rank_of(sq) + d[1];
    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
      int s = r * 8 + f;
      att |= bit(s);
      if (occ & bit(s)) break;
      f += d[0];
      r += d[1];
    }
  }
  return att;
}

void init() {
  static bool done = false;
  if (done) return;
  done = true;
  for (int sq = 0; sq < 64; sq++) {
    knight_attacks[sq] = step(sq, 1, 2) | step(sq, 2, 1) | step(sq, 2, -1) | step(sq, 1, -2) |
                         step(sq, -1, -2) | step(sq, -2, -1) | step(sq, -2, 1) | step(sq, -1, 2);
    king_attacks[sq] = step(sq, 1, 0) | step(sq, 1, 1) | step(sq, 0, 1) | step(sq, -1, 1) |
                       step(sq, -1, 0) | step(sq, -1, -1) | step(sq, 0, -1) | step(sq, 1, -1);
    pawn_attacks[WHITE][sq] = step(sq, -1, 1) | step(sq, 1, 1);
    pawn_attacks[BLACK][sq] = step(sq, -1, -1) | step(sq, 1, -1);
  }
  Rng rng{0x9E3779B97F4A7C15ULL};
  init_magics(true, rook_magics, rook_table, rng);
  init_magics(false, bishop_magics, bishop_table, rng);
}

}  // namespace bb
