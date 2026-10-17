#include "tt.h"

void TT::resize(size_t megabytes) {
  size_t count = 1;
  while (count * 2 * sizeof(Slot) <= megabytes * 1024 * 1024) count *= 2;  // power of two for masking
  table_ = std::vector<Slot>(count);
  mask_ = count - 1;
  clear();
}

void TT::clear() {
  for (Slot& s : table_) {
    s.check.store(0, std::memory_order_relaxed);
    s.data.store(0, std::memory_order_relaxed);
  }
}

U64 TT::pack(const TTEntry& e) {
  return U64(e.move) | (U64(uint16_t(e.score)) << 16) | (U64(uint8_t(e.depth)) << 32) | (U64(e.bound) << 40) |
         (U64(e.generation) << 48);
}

TTEntry TT::unpack(U64 d) {
  TTEntry e;
  e.move = Move(d & 0xFFFF);
  e.score = int16_t(uint16_t(d >> 16));
  e.depth = int8_t(uint8_t(d >> 32));
  e.bound = uint8_t(d >> 40);
  e.generation = uint8_t(d >> 48);
  return e;
}

bool TT::probe(U64 key, TTEntry& out) const {
  const Slot& s = table_[key & mask_];
  U64 data = s.data.load(std::memory_order_relaxed);
  U64 check = s.check.load(std::memory_order_relaxed);
  if ((check ^ data) != key) return false;
  out = unpack(data);
  return out.bound != BOUND_NONE;
}

void TT::store(U64 key, Move move, int score, int depth, Bound bound) {
  Slot& s = table_[key & mask_];
  U64 old_data = s.data.load(std::memory_order_relaxed);
  bool same = (s.check.load(std::memory_order_relaxed) ^ old_data) == key;
  TTEntry old = unpack(old_data);
  if (old.bound != BOUND_NONE) {
    if (same) {
      if (depth < old.depth && bound != BOUND_EXACT) return;
    } else if (old.generation == generation_ && old.depth > depth + 2) {
      return;  // a much deeper result from this search: worth more than the new one
    }
  }
  if (move == NO_MOVE && same) move = old.move;  // keep the old best move if we have none
  TTEntry e{move, int16_t(score), int8_t(depth), uint8_t(bound), generation_};
  U64 data = pack(e);
  s.data.store(data, std::memory_order_relaxed);
  s.check.store(key ^ data, std::memory_order_relaxed);
}

int TT::hashfull() const {
  int used = 0;
  for (size_t i = 0; i < 1000 && i < table_.size(); i++)
    used += unpack(table_[i].data.load(std::memory_order_relaxed)).bound != BOUND_NONE;
  return used;
}
