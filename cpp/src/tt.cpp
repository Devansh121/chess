#include "tt.h"

void TT::resize(size_t megabytes) {
  size_t count = 1;
  while (count * 2 * sizeof(TTEntry) <= megabytes * 1024 * 1024) count *= 2;  // power of two for masking
  table_.assign(count, TTEntry{});
  mask_ = count - 1;
}

void TT::clear() { std::fill(table_.begin(), table_.end(), TTEntry{}); }

const TTEntry* TT::probe(U64 key) const {
  const TTEntry& e = table_[key & mask_];
  return (e.bound != BOUND_NONE && e.key == key) ? &e : nullptr;
}

void TT::store(U64 key, Move move, int score, int depth, Bound bound) {
  TTEntry& e = table_[key & mask_];
  if (e.bound != BOUND_NONE) {
    if (e.key == key) {
      if (depth < e.depth && bound != BOUND_EXACT) return;
    } else if (e.generation == generation_ && e.depth > depth + 2) {
      return;  // a much deeper result from this search: worth more than the new one
    }
  }
  if (move == NO_MOVE && e.key == key) move = e.move;  // keep the old best move if we have none
  e = TTEntry{key, move, int16_t(score), int8_t(depth), uint8_t(bound), generation_, 0};
}

int TT::hashfull() const {
  int used = 0;
  for (size_t i = 0; i < 1000 && i < table_.size(); i++) used += table_[i].bound != BOUND_NONE;
  return used;
}
