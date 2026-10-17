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
//
// The search uses a quantized integer version: w1/b1 scaled by QA = 255 into int16, so the clipped
// activation range [0, 1] becomes [0, 255]; w2 scaled by QB = 256 into int16; the 32 -> 1 output layer
// stays in float. With AVX2 the accumulator updates and the hidden layer use 256-bit integer SIMD.
// The float functions (refresh_float, logit_float) are the reference the tests compare against.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "position.h"

namespace nnue {

constexpr int NUM_FEATURES = 768;
constexpr int MAX_HIDDEN1 = 512;
constexpr int QA = 255;
constexpr int QB = 256;

struct Network {
  int arch = 0;
  int hidden1 = 0, hidden2 = 0;
  float cp_per_logit = 0;
  // Float weights as trained (reference).
  std::vector<float> w1;  // [NUM_FEATURES + 1][hidden1]; the last row is padding (unused)
  std::vector<float> b1;  // [hidden1]
  std::vector<float> w2;  // [hidden2][arch * hidden1]
  std::vector<float> b2;  // [hidden2]
  std::vector<float> w3;  // [hidden2]
  float b3 = 0;
  // Quantized copies used by the search.
  std::vector<int16_t> w1q, b1q, w2q;

  // Loads a .nnue file written by scripts/export_net.py and quantizes it. Returns an error message,
  // empty on success (including a refusal if the weights could overflow int16).
  std::string load(const std::string& path);
};

struct Accumulator {
  alignas(64) int16_t v[2][MAX_HIDDEN1];  // [perspective][hidden1], quantized by QA
};
struct FloatAccumulator {
  alignas(64) float v[2][MAX_HIDDEN1];
};

// Feature index of a piece seen from `perspective`.
inline int feature(int perspective, int color, int type, int sq) {
  int s = perspective == WHITE ? sq : sq ^ 56;
  return ((color == perspective ? 0 : 1) * 6 + type) * 64 + s;
}

void refresh(const Network& net, const Position& pos, Accumulator& acc);
// child accumulator = parent accumulator + (pieces added) - (pieces removed) going parent -> child.
void update(const Network& net, const Position& parent, const Position& child, const Accumulator& parent_acc,
            Accumulator& child_acc);
float logit(const Network& net, const Accumulator& acc, int side_to_move);
// Centipawns for the side to move, capped at +/-5000 to stay clear of mate scores.
int evaluate(const Network& net, const Accumulator& acc, int side_to_move);

// Float reference.
void refresh_float(const Network& net, const Position& pos, FloatAccumulator& acc);
float logit_float(const Network& net, const FloatAccumulator& acc, int side_to_move);

}  // namespace nnue
