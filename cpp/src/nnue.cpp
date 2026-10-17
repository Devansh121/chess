#include "nnue.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace nnue {

namespace {

template <typename T>
bool read_raw(std::ifstream& in, T* out, size_t count) {
  in.read(reinterpret_cast<char*>(out), std::streamsize(count * sizeof(T)));
  return bool(in);
}

inline void add_row(float* acc, const float* row, int n) {
  for (int i = 0; i < n; i++) acc[i] += row[i];
}
inline void sub_row(float* acc, const float* row, int n) {
  for (int i = 0; i < n; i++) acc[i] -= row[i];
}

}  // namespace

std::string Network::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return "cannot open " + path;
  char magic[4];
  uint32_t header[5];
  float cp_scale;
  if (!read_raw(in, magic, 4) || std::memcmp(magic, "CHNN", 4) != 0) return "not a .nnue file: " + path;
  if (!read_raw(in, header, 5) || !read_raw(in, &cp_scale, 1)) return "truncated header";
  uint32_t version = header[0], a = header[1], inputs = header[2], h1 = header[3], h2 = header[4];
  if (version != 1) return "unsupported version " + std::to_string(version);
  if ((a != 1 && a != 2) || inputs != NUM_FEATURES + 1 || h1 == 0 || h1 > MAX_HIDDEN1 || h2 == 0)
    return "unsupported dimensions";
  arch = int(a);
  hidden1 = int(h1);
  hidden2 = int(h2);
  cp_per_logit = cp_scale / std::log(10.0f);
  w1.resize(size_t(inputs) * h1);
  b1.resize(h1);
  w2.resize(size_t(h2) * arch * h1);
  b2.resize(h2);
  w3.resize(h2);
  if (!read_raw(in, w1.data(), w1.size()) || !read_raw(in, b1.data(), b1.size()) ||
      !read_raw(in, w2.data(), w2.size()) || !read_raw(in, b2.data(), b2.size()) ||
      !read_raw(in, w3.data(), w3.size()) || !read_raw(in, &b3, 1))
    return "truncated weights";
  return "";
}

void refresh(const Network& net, const Position& pos, Accumulator& acc) {
  for (int p = WHITE; p <= BLACK; p++) {
    float* v = acc.v[p];
    std::copy(net.b1.begin(), net.b1.end(), v);
    for (int c = WHITE; c <= BLACK; c++)
      for (int t = PAWN; t <= KING; t++)
        for (U64 b = pos.pieces[c][t]; b;) {
          int sq = pop_lsb(b);
          add_row(v, &net.w1[size_t(feature(p, c, t, sq)) * net.hidden1], net.hidden1);
        }
  }
}

void update(const Network& net, const Position& parent, const Position& child, const Accumulator& parent_acc,
            Accumulator& child_acc) {
  int n = net.hidden1;
  for (int p = WHITE; p <= BLACK; p++) std::memcpy(child_acc.v[p], parent_acc.v[p], sizeof(float) * n);
  for (int c = WHITE; c <= BLACK; c++)
    for (int t = PAWN; t <= KING; t++) {
      U64 diff = parent.pieces[c][t] ^ child.pieces[c][t];
      if (!diff) continue;
      for (U64 removed = diff & parent.pieces[c][t]; removed;) {
        int sq = pop_lsb(removed);
        for (int p = WHITE; p <= BLACK; p++) sub_row(child_acc.v[p], &net.w1[size_t(feature(p, c, t, sq)) * n], n);
      }
      for (U64 added = diff & child.pieces[c][t]; added;) {
        int sq = pop_lsb(added);
        for (int p = WHITE; p <= BLACK; p++) add_row(child_acc.v[p], &net.w1[size_t(feature(p, c, t, sq)) * n], n);
      }
    }
}

float logit(const Network& net, const Accumulator& acc, int side_to_move) {
  int n = net.hidden1;
  float h1[2 * MAX_HIDDEN1];
  for (int i = 0; i < n; i++) h1[i] = std::clamp(acc.v[side_to_move][i], 0.0f, 1.0f);
  if (net.arch == 2)
    for (int i = 0; i < n; i++) h1[n + i] = std::clamp(acc.v[side_to_move ^ 1][i], 0.0f, 1.0f);
  int width = net.arch * n;
  float out = net.b3;
  for (int j = 0; j < net.hidden2; j++) {
    const float* row = &net.w2[size_t(j) * width];
    float s = net.b2[j];
    for (int i = 0; i < width; i++) s += row[i] * h1[i];
    out += net.w3[j] * std::clamp(s, 0.0f, 1.0f);
  }
  return out;
}

int evaluate(const Network& net, const Accumulator& acc, int side_to_move) {
  int cp = int(logit(net, acc, side_to_move) * net.cp_per_logit);
  return std::clamp(cp, -5000, 5000);
}

}  // namespace nnue
