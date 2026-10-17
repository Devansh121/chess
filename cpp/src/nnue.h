// Value network evaluation (float reference implementation).
//
// Inputs are the same 768 piece-square features as nn/features.py: ((enemy * 6 + type) * 64 + square),
// seen from one side ("perspective"), with ranks mirrored when that side is Black. Each position keeps
// two accumulators, one per perspective: the first-layer sums b1 + sum(w1[feature]). They are updated
// incrementally from the parent position by comparing the two positions' piece bitboards, which covers
// captures, castling, en passant and promotion without special cases.
//
// arch 1: clip(acc[side to move]) -> hidden2 -> 1
// arch 2: [clip(acc[side to move]), clip(acc[other side])] -> hidden2 -> 1
#pragma once

#include <string>
#include <vector>

#include "position.h"

namespace nnue {

constexpr int NUM_FEATURES = 768;
constexpr int MAX_HIDDEN1 = 512;

struct Network {
  int arch = 0;
  int hidden1 = 0, hidden2 = 0;
  float cp_per_logit = 0;
  std::vector<float> w1;  // [NUM_FEATURES + 1][hidden1]; the last row is padding (unused here)
  std::vector<float> b1;  // [hidden1]
  std::vector<float> w2;  // [hidden2][arch * hidden1]
  std::vector<float> b2;  // [hidden2]
  std::vector<float> w3;  // [hidden2]
  float b3 = 0;

  // Loads a .nnue file written by scripts/export_net.py. Returns an error message, empty on success.
  std::string load(const std::string& path);
};

struct Accumulator {
  alignas(64) float v[2][MAX_HIDDEN1];  // [perspective][hidden1]
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
// Centipawns for the side to move, capped at +/-5000 to stay clear of mate scores.
int evaluate(const Network& net, const Accumulator& acc, int side_to_move);
float logit(const Network& net, const Accumulator& acc, int side_to_move);

}  // namespace nnue
