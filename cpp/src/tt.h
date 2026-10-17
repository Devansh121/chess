// Transposition table: a fixed-size array of 16-byte slots indexed by the position hash, shared by all
// search threads without locks. Each slot is two 8-byte words: the packed data, and the key XOR the
// data. A read that sees half of one write and half of another fails the key check and is ignored,
// so concurrent writes can't produce a wrong entry (the standard "lockless hashing" trick).
//
// Replacement: an entry from the current search that is more than 2 plies deeper than the new one is
// kept (so shallow quiescence results don't evict deep ones); entries from older searches, and
// anything shallower, are replaced.
#pragma once

#include <atomic>
#include <vector>

#include "types.h"

enum Bound : uint8_t { BOUND_NONE = 0, BOUND_EXACT = 1, BOUND_LOWER = 2, BOUND_UPPER = 3 };

struct TTEntry {
  Move move = NO_MOVE;
  int16_t score = 0;
  int8_t depth = 0;
  uint8_t bound = BOUND_NONE;
  uint8_t generation = 0;
};

class TT {
 public:
  void resize(size_t megabytes);
  void clear();
  void new_search() { generation_++; }
  // Copies the entry for this key into `out`; false if there is none.
  bool probe(U64 key, TTEntry& out) const;
  // Start loading the slot for `key` into cache (call as soon as a child position's hash is known).
  void prefetch(U64 key) const { __builtin_prefetch(&table_[key & mask_]); }
  void store(U64 key, Move move, int score, int depth, Bound bound);
  // Permille of sampled slots in use, for UCI "hashfull".
  int hashfull() const;

 private:
  struct Slot {
    std::atomic<U64> check;  // key ^ data
    std::atomic<U64> data;
  };
  static U64 pack(const TTEntry& e);
  static TTEntry unpack(U64 d);

  std::vector<Slot> table_;
  size_t mask_ = 0;
  uint8_t generation_ = 0;
};
