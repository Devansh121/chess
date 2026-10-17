// Transposition table: a fixed-size array of 16-byte entries indexed by the position hash.
// Replacement: keep the deeper entry unless the slot holds a different position.
#pragma once

#include <vector>

#include "types.h"

enum Bound : uint8_t { BOUND_NONE = 0, BOUND_EXACT = 1, BOUND_LOWER = 2, BOUND_UPPER = 3 };

struct TTEntry {
  U64 key;
  Move move;
  int16_t score;
  int8_t depth;
  uint8_t bound;
  uint16_t pad;
};

class TT {
 public:
  void resize(size_t megabytes);
  void clear();
  // Returns the entry for this key, or nullptr.
  const TTEntry* probe(U64 key) const;
  void store(U64 key, Move move, int score, int depth, Bound bound);
  // Permille of sampled slots in use, for UCI "hashfull".
  int hashfull() const;

 private:
  std::vector<TTEntry> table_;
  size_t mask_ = 0;
};
