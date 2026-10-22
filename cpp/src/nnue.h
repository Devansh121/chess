// Value network evaluation.
//
// Inputs are the same 768 piece-square features as nn/features.py: ((enemy * 6 + type) * 64 + square),
// seen from one side ("perspective"), with ranks mirrored when that side is Black. Each position keeps
// two accumulators, one per perspective: the first-layer sums b1 + sum(w1[feature]). They are updated
// incrementally from the parent position by comparing the two positions' piece bitboards, which covers
// captures, castling, en passant and promotion without special cases.
//
// arch 1: clip(acc[side to move]) -> hidden2 -> 1
// arch 2: [clip(acc[side to move]), clip(acc[other side])] -> hidden2 -> 1
// arch 3: arch 2 with king buckets: each perspective mirrors files when its own king is on files e-h and
//         uses one of num_buckets row sets, picked by that king's square. A king move that changes the
//         bucket or crosses the mirror line refreshes that perspective's accumulator; the other stays incremental.
// arch 4: arch 3 with output buckets: output_buckets copies of the hidden and output layers, one picked by
//         the number of pieces on the board, (pieces - 2) * output_buckets / 32.
//
// The search uses a quantized integer version: w1/b1 scaled by QA = 127 into int16, so the clipped
// activation range [0, 1] becomes [0, 127] and fits a uint8; w2 scaled into int8 by the largest factor
// that fits this network's weights (qb, at least QB_MIN); the 32 -> 1
// output layer stays in float. With AVX2 the accumulator updates use 16-bit lanes and the hidden layer
// uses maddubs (uint8 x int8 pairs; 2 * 127 * 127 < 32767, so it can't saturate) on 8 outputs at a time.
// The float functions (refresh_float, logit_float) are the reference the tests compare against.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "position.h"

namespace nnue {

constexpr int NUM_FEATURES = 768;
constexpr int MAX_HIDDEN1 = 512;
constexpr int QA = 127;
constexpr int QB_MIN = 32;  // refuse networks whose hidden weights would need a coarser scale

struct Network {
  int arch = 0;
  int hidden1 = 0, hidden2 = 0;
  int num_buckets = 1;
  bool mirror = false;              // arch 3: mirror files when the perspective's king is on files e-h
  uint8_t bucket[64] = {};          // arch 3: bucket by own-king square (perspective view, after mirroring)
  int output_buckets = 1;           // arch 4: hidden/output layer sets, picked by piece count
  int output_bucket(int pieces) const {
    return std::clamp((pieces - 2) * output_buckets / 32, 0, output_buckets - 1);
  }
  int perspectives() const { return arch == 1 ? 1 : 2; }
  float cp_per_logit = 0;
  // Float weights as trained (reference).
  std::vector<float> w1;  // [num_buckets * NUM_FEATURES + 1][hidden1]; the last row is padding (unused)
  std::vector<float> b1;  // [hidden1]
  std::vector<float> w2;  // [output_buckets][hidden2][perspectives * hidden1]
  std::vector<float> b2;  // [output_buckets][hidden2]
  std::vector<float> w3;  // [output_buckets][hidden2]
  std::vector<float> b3;  // [output_buckets]
  // Quantized copies used by the search.
  std::vector<int16_t> w1q, b1q;
  std::vector<int8_t> w2q;  // [output_buckets][hidden2][perspectives * hidden1]
  int qb = 0;               // scale of w2q, chosen per network

  // Loads a .nnue file written by scripts/export_net.py and quantizes it. Returns an error message,
  // empty on success (including a refusal if the weights could overflow int16).
  std::string load(const std::string& path);
  std::string load_from_memory(const void* data, size_t size);
  // The network built into the binary (nets/default.nnue at compile time); error if there is none.
  std::string load_embedded();
};

struct Accumulator {
  alignas(64) int16_t v[2][MAX_HIDDEN1];  // [perspective][hidden1], quantized by QA
};
struct FloatAccumulator {
  alignas(64) float v[2][MAX_HIDDEN1];
};

// How one perspective sees the board: ranks flipped for Black, files flipped when mirroring, and the
// first row of the king bucket's row set.
struct View {
  int flip = 0;  // XOR applied to the square
  int base = 0;  // bucket * NUM_FEATURES
  bool operator==(const View&) const = default;
};

inline View view(const Network& net, const Position& pos, int perspective) {
  View v;
  v.flip = perspective == WHITE ? 0 : 56;
  if (net.num_buckets > 1 || net.mirror) {
    int k = pos.king_sq(perspective) ^ v.flip;
    if (net.mirror && file_of(k) >= 4) v.flip ^= 7;
    v.base = net.bucket[pos.king_sq(perspective) ^ v.flip] * NUM_FEATURES;
  }
  return v;
}

// Feature index of a piece seen from `perspective`.
inline int feature(const View& v, int perspective, int color, int type, int sq) {
  return v.base + ((color == perspective ? 0 : 1) * 6 + type) * 64 + (sq ^ v.flip);
}

// Per-thread cache for king-bucket refreshes: one accumulator per (perspective, bucket, mirror half) with
// the pieces it was built from, so a refresh only applies the pieces that changed since that bucket was
// last used instead of re-adding every piece. Results are identical to a plain refresh.
class RefreshCache {
 public:
  void reset(const Network& net);  // call whenever the network changes
  void refresh(const Network& net, const Position& pos, int perspective, const View& v, int16_t* out);

 private:
  struct Entry {
    alignas(64) int16_t acc[MAX_HIDDEN1];
    U64 pieces[2][6];
    bool valid = false;
  };
  std::vector<Entry> entries_;  // [perspective][bucket][mirrored]
  int buckets_ = 0;
};

void refresh(const Network& net, const Position& pos, Accumulator& acc);
// child accumulator = parent accumulator + (pieces added) - (pieces removed) going parent -> child.
// A perspective whose king changed bucket is rebuilt, through `cache` when one is given.
void update(const Network& net, const Position& parent, const Position& child, const Accumulator& parent_acc,
            Accumulator& child_acc, RefreshCache* cache = nullptr);
// `pieces` (kings included) picks the output bucket of arch 4 networks; other archs ignore it.
float logit(const Network& net, const Accumulator& acc, int side_to_move, int pieces = 32);
// Centipawns for the side to move, capped at +/-5000 to stay clear of mate scores.
int evaluate(const Network& net, const Accumulator& acc, int side_to_move, int pieces = 32);

// Float reference.
void refresh_float(const Network& net, const Position& pos, FloatAccumulator& acc);
float logit_float(const Network& net, const FloatAccumulator& acc, int side_to_move, int pieces = 32);

}  // namespace nnue
