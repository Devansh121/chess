#include "nnue.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nnue {

namespace {

template <typename T>
bool read_raw(std::ifstream& in, T* out, size_t count) {
  in.read(reinterpret_cast<char*>(out), std::streamsize(count * sizeof(T)));
  return bool(in);
}

int16_t quantize(float x, int scale) { return int16_t(std::lround(x * scale)); }

// acc += row / acc -= row over n int16 lanes (n is a multiple of 16 for AVX2; scalar tail otherwise).
inline void add_row(int16_t* acc, const int16_t* row, int n) {
  int i = 0;
#if defined(__AVX2__)
  for (; i + 16 <= n; i += 16) {
    __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
    __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_add_epi16(a, r));
  }
#endif
  for (; i < n; i++) acc[i] = int16_t(acc[i] + row[i]);
}
inline void sub_row(int16_t* acc, const int16_t* row, int n) {
  int i = 0;
#if defined(__AVX2__)
  for (; i + 16 <= n; i += 16) {
    __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
    __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + i), _mm256_sub_epi16(a, r));
  }
#endif
  for (; i < n; i++) acc[i] = int16_t(acc[i] - row[i]);
}

// Sum of a[i] * b[i] over n int16 values (n a multiple of 16 for the AVX2 path).
inline int32_t dot_i16(const int16_t* a, const int16_t* b, int n) {
  int i = 0;
  int32_t total = 0;
#if defined(__AVX2__)
  __m256i sum = _mm256_setzero_si256();
  for (; i + 16 <= n; i += 16) {
    __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(x, y));  // pairwise products summed into int32
  }
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
  total = _mm_cvtsi128_si32(s);
#endif
  for (; i < n; i++) total += int32_t(a[i]) * b[i];
  return total;
}

// Clip accumulator values to [0, QA] (the integer form of clamp(x, 0, 1)).
inline void clipped(const int16_t* acc, int16_t* out, int n) {
  int i = 0;
#if defined(__AVX2__)
  const __m256i zero = _mm256_setzero_si256(), top = _mm256_set1_epi16(QA);
  for (; i + 16 <= n; i += 16) {
    __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
    _mm256_store_si256(reinterpret_cast<__m256i*>(out + i), _mm256_min_epi16(_mm256_max_epi16(x, zero), top));
  }
#endif
  for (; i < n; i++) out[i] = int16_t(std::clamp<int>(acc[i], 0, QA));
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

  // Overflow guards: an accumulator is b1 plus at most 32 piece rows; a hidden-layer sum is at most
  // arch * hidden1 products of QA * |w2q|.
  float max_w1 = 0, max_b1 = 0, max_w2 = 0;
  for (float x : w1) max_w1 = std::max(max_w1, std::fabs(x));
  for (float x : b1) max_b1 = std::max(max_b1, std::fabs(x));
  for (float x : w2) max_w2 = std::max(max_w2, std::fabs(x));
  if ((max_b1 + 32 * max_w1) * QA > 32000) return "first-layer weights too large for int16 accumulators";
  if (max_w2 * QB > 32000) return "second-layer weights too large for int16";
  if (double(arch) * h1 * QA * (max_w2 * QB) > 2.0e9) return "second-layer sums could overflow int32";

  w1q.resize(w1.size());
  b1q.resize(b1.size());
  w2q.resize(w2.size());
  for (size_t i = 0; i < w1.size(); i++) w1q[i] = quantize(w1[i], QA);
  for (size_t i = 0; i < b1.size(); i++) b1q[i] = quantize(b1[i], QA);
  for (size_t i = 0; i < w2.size(); i++) w2q[i] = quantize(w2[i], QB);
  return "";
}

void refresh(const Network& net, const Position& pos, Accumulator& acc) {
  int n = net.hidden1;
  for (int p = WHITE; p <= BLACK; p++) {
    int16_t* v = acc.v[p];
    std::memcpy(v, net.b1q.data(), sizeof(int16_t) * n);
    for (int c = WHITE; c <= BLACK; c++)
      for (int t = PAWN; t <= KING; t++)
        for (U64 b = pos.pieces[c][t]; b;) {
          int sq = pop_lsb(b);
          add_row(v, &net.w1q[size_t(feature(p, c, t, sq)) * n], n);
        }
  }
}

void update(const Network& net, const Position& parent, const Position& child, const Accumulator& parent_acc,
            Accumulator& child_acc) {
  int n = net.hidden1;
  for (int p = WHITE; p <= BLACK; p++) std::memcpy(child_acc.v[p], parent_acc.v[p], sizeof(int16_t) * n);
  for (int c = WHITE; c <= BLACK; c++)
    for (int t = PAWN; t <= KING; t++) {
      U64 diff = parent.pieces[c][t] ^ child.pieces[c][t];
      if (!diff) continue;
      for (U64 removed = diff & parent.pieces[c][t]; removed;) {
        int sq = pop_lsb(removed);
        for (int p = WHITE; p <= BLACK; p++) sub_row(child_acc.v[p], &net.w1q[size_t(feature(p, c, t, sq)) * n], n);
      }
      for (U64 added = diff & child.pieces[c][t]; added;) {
        int sq = pop_lsb(added);
        for (int p = WHITE; p <= BLACK; p++) add_row(child_acc.v[p], &net.w1q[size_t(feature(p, c, t, sq)) * n], n);
      }
    }
}

float logit(const Network& net, const Accumulator& acc, int side_to_move) {
  int n = net.hidden1;
  alignas(64) int16_t h1[2 * MAX_HIDDEN1];
  clipped(acc.v[side_to_move], h1, n);
  if (net.arch == 2) clipped(acc.v[side_to_move ^ 1], h1 + n, n);
  int width = net.arch * n;
  const float scale = 1.0f / (float(QA) * QB);
  float out = net.b3;
  for (int j = 0; j < net.hidden2; j++) {
    float s = float(dot_i16(h1, &net.w2q[size_t(j) * width], width)) * scale + net.b2[j];
    out += net.w3[j] * std::clamp(s, 0.0f, 1.0f);
  }
  return out;
}

int evaluate(const Network& net, const Accumulator& acc, int side_to_move) {
  int cp = int(logit(net, acc, side_to_move) * net.cp_per_logit);
  return std::clamp(cp, -5000, 5000);
}

void refresh_float(const Network& net, const Position& pos, FloatAccumulator& acc) {
  int n = net.hidden1;
  for (int p = WHITE; p <= BLACK; p++) {
    float* v = acc.v[p];
    std::copy(net.b1.begin(), net.b1.end(), v);
    for (int c = WHITE; c <= BLACK; c++)
      for (int t = PAWN; t <= KING; t++)
        for (U64 b = pos.pieces[c][t]; b;) {
          const float* row = &net.w1[size_t(feature(p, c, t, pop_lsb(b))) * n];
          for (int i = 0; i < n; i++) v[i] += row[i];
        }
  }
}

float logit_float(const Network& net, const FloatAccumulator& acc, int side_to_move) {
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

}  // namespace nnue
