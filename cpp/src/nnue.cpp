#include "nnue.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace nnue {

namespace {

template <typename T>
bool read_raw(std::istream& in, T* out, size_t count) {
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

// Clip accumulator values to [0, QA] (the integer form of clamp(x, 0, 1)) and narrow to uint8.
inline void clip_to_u8(const int16_t* acc, uint8_t* out, int n) {
  int i = 0;
#if defined(__AVX2__)
  const __m256i zero = _mm256_setzero_si256(), top = _mm256_set1_epi16(QA);
  for (; i + 32 <= n; i += 32) {
    __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i));
    __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(acc + i + 16));
    a = _mm256_min_epi16(_mm256_max_epi16(a, zero), top);
    b = _mm256_min_epi16(_mm256_max_epi16(b, zero), top);
    // packus works per 128-bit lane; the permute restores element order.
    __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), packed);
  }
#endif
  for (; i < n; i++) out[i] = uint8_t(std::clamp<int>(acc[i], 0, QA));
}

// out[j] = sum_i h[i] * w[j * width + i] for j < h2.
inline void hidden_layer(const uint8_t* h, const int8_t* w, int width, int h2, int32_t* out) {
  int j = 0;
#if defined(__AVX2__)
  if (width % 32 == 0) {
    const __m256i ones = _mm256_set1_epi16(1);
    for (; j + 8 <= h2; j += 8) {
      __m256i acc[8];
      for (auto& a : acc) a = _mm256_setzero_si256();
      for (int c = 0; c < width; c += 32) {
        __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(h + c));
        for (int k = 0; k < 8; k++) {
          __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + size_t(j + k) * width + c));
          // uint8 x int8 products summed in pairs to int16, then pairs of those to int32.
          acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(_mm256_maddubs_epi16(x, y), ones));
        }
      }
      // Reduce the 8 accumulators to one vector holding the 8 totals, in order.
      __m256i s01 = _mm256_hadd_epi32(acc[0], acc[1]), s23 = _mm256_hadd_epi32(acc[2], acc[3]);
      __m256i s45 = _mm256_hadd_epi32(acc[4], acc[5]), s67 = _mm256_hadd_epi32(acc[6], acc[7]);
      __m256i s0123 = _mm256_hadd_epi32(s01, s23), s4567 = _mm256_hadd_epi32(s45, s67);
      __m256i sums = _mm256_add_epi32(_mm256_permute2x128_si256(s0123, s4567, 0x20),
                                      _mm256_permute2x128_si256(s0123, s4567, 0x31));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), sums);
    }
  }
#endif
  for (; j < h2; j++) {
    int32_t s = 0;
    for (int i = 0; i < width; i++) s += int32_t(h[i]) * w[size_t(j) * width + i];
    out[j] = s;
  }
}

}  // namespace

std::string Network::load(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return "cannot open " + path;
  std::ostringstream bytes;
  bytes << file.rdbuf();
  std::string data = bytes.str();
  std::string err = load_from_memory(data.data(), data.size());
  return err.empty() ? err : err + " (" + path + ")";
}

std::string Network::load_from_memory(const void* data, size_t size) {
  std::istringstream in(std::string(static_cast<const char*>(data), size), std::ios::binary);
  char magic[4];
  uint32_t header[5];
  float cp_scale;
  if (!read_raw(in, magic, 4) || std::memcmp(magic, "CHNN", 4) != 0) return "not a .nnue file";
  if (!read_raw(in, header, 5) || !read_raw(in, &cp_scale, 1)) return "truncated header";
  uint32_t version = header[0], a = header[1], inputs = header[2], h1 = header[3], h2 = header[4];
  if (version != (a == 3 ? 2u : 1u)) return "unsupported version " + std::to_string(version);
  num_buckets = 1;
  mirror = false;
  std::memset(bucket, 0, sizeof(bucket));
  if (a == 3) {
    uint32_t nb;
    if (!read_raw(in, &nb, 1) || !read_raw(in, bucket, 64)) return "truncated header";
    if (nb < 1 || nb > 64) return "unsupported bucket count";
    for (uint8_t b : bucket)
      if (b >= nb) return "bad bucket map";
    num_buckets = int(nb);
    mirror = true;
  }
  if (a < 1 || a > 3 || inputs != uint32_t(num_buckets * NUM_FEATURES + 1) || h1 == 0 || h1 > MAX_HIDDEN1 ||
      h2 == 0)
    return "unsupported dimensions";
  arch = int(a);
  hidden1 = int(h1);
  hidden2 = int(h2);
  cp_per_logit = cp_scale / std::log(10.0f);
  w1.resize(size_t(inputs) * h1);
  b1.resize(h1);
  w2.resize(size_t(h2) * perspectives() * h1);
  b2.resize(h2);
  w3.resize(h2);
  if (!read_raw(in, w1.data(), w1.size()) || !read_raw(in, b1.data(), b1.size()) ||
      !read_raw(in, w2.data(), w2.size()) || !read_raw(in, b2.data(), b2.size()) ||
      !read_raw(in, w3.data(), w3.size()) || !read_raw(in, &b3, 1))
    return "truncated weights";

  // Overflow guards: an accumulator is b1 plus at most 32 piece rows; a hidden-layer sum is at most
  // perspectives * hidden1 products of QA * |w2q|.
  float max_w1 = 0, max_b1 = 0, max_w2 = 0;
  for (float x : w1) max_w1 = std::max(max_w1, std::fabs(x));
  for (float x : b1) max_b1 = std::max(max_b1, std::fabs(x));
  for (float x : w2) max_w2 = std::max(max_w2, std::fabs(x));
  if ((max_b1 + 32 * max_w1) * QA > 32000) return "first-layer weights too large for int16 accumulators";
  // Hidden-layer int8 scale from the 99.99th-percentile weight rather than the single largest one: a
  // handful of outlier weights would otherwise force a coarse scale on all of them. The few weights
  // beyond the percentile are clamped to +/-127 below.
  std::vector<float> mags(w2.size());
  for (size_t i = 0; i < w2.size(); i++) mags[i] = std::fabs(w2[i]);
  size_t k = std::min(mags.size() - 1, size_t(double(mags.size()) * 0.9999));
  std::nth_element(mags.begin(), mags.begin() + long(k), mags.end());
  float scale_ref = std::max(mags[k], 1e-6f);
  qb = std::min(127, int(127.0f / scale_ref));
  if (qb < QB_MIN) return "second-layer weights too large for int8";

  w1q.resize(w1.size());
  b1q.resize(b1.size());
  w2q.resize(w2.size());
  for (size_t i = 0; i < w1.size(); i++) w1q[i] = quantize(w1[i], QA);
  for (size_t i = 0; i < b1.size(); i++) b1q[i] = quantize(b1[i], QA);
  for (size_t i = 0; i < w2.size(); i++) w2q[i] = int8_t(std::clamp<long>(std::lround(w2[i] * qb), -127, 127));
  return "";
}

namespace {

void refresh_perspective(const Network& net, const Position& pos, int p, int16_t* v) {
  int n = net.hidden1;
  View view_p = view(net, pos, p);
  std::memcpy(v, net.b1q.data(), sizeof(int16_t) * n);
  for (int c = WHITE; c <= BLACK; c++)
    for (int t = PAWN; t <= KING; t++)
      for (U64 b = pos.pieces[c][t]; b;) {
        int sq = pop_lsb(b);
        add_row(v, &net.w1q[size_t(feature(view_p, p, c, t, sq)) * n], n);
      }
}

}  // namespace

void refresh(const Network& net, const Position& pos, Accumulator& acc) {
  for (int p = WHITE; p <= BLACK; p++) refresh_perspective(net, pos, p, acc.v[p]);
}

void update(const Network& net, const Position& parent, const Position& child, const Accumulator& parent_acc,
            Accumulator& child_acc) {
  int n = net.hidden1;
  View views[2];
  bool incremental[2];
  for (int p = WHITE; p <= BLACK; p++) {
    views[p] = view(net, child, p);
    incremental[p] = views[p] == view(net, parent, p);
    if (incremental[p])
      std::memcpy(child_acc.v[p], parent_acc.v[p], sizeof(int16_t) * n);
    else
      refresh_perspective(net, child, p, child_acc.v[p]);  // this side's king changed bucket or mirror half
  }
  for (int c = WHITE; c <= BLACK; c++)
    for (int t = PAWN; t <= KING; t++) {
      U64 diff = parent.pieces[c][t] ^ child.pieces[c][t];
      if (!diff) continue;
      for (U64 removed = diff & parent.pieces[c][t]; removed;) {
        int sq = pop_lsb(removed);
        for (int p = WHITE; p <= BLACK; p++)
          if (incremental[p]) sub_row(child_acc.v[p], &net.w1q[size_t(feature(views[p], p, c, t, sq)) * n], n);
      }
      for (U64 added = diff & child.pieces[c][t]; added;) {
        int sq = pop_lsb(added);
        for (int p = WHITE; p <= BLACK; p++)
          if (incremental[p]) add_row(child_acc.v[p], &net.w1q[size_t(feature(views[p], p, c, t, sq)) * n], n);
      }
    }
}

float logit(const Network& net, const Accumulator& acc, int side_to_move) {
  int n = net.hidden1;
  alignas(64) uint8_t h1[2 * MAX_HIDDEN1];
  clip_to_u8(acc.v[side_to_move], h1, n);
  if (net.perspectives() == 2) clip_to_u8(acc.v[side_to_move ^ 1], h1 + n, n);
  int width = net.perspectives() * n;
  alignas(64) int32_t sums[256];
  int h2 = std::min(net.hidden2, 256);
  hidden_layer(h1, net.w2q.data(), width, h2, sums);
  const float scale = 1.0f / (float(QA) * net.qb);
  float out = net.b3;
  for (int j = 0; j < h2; j++) out += net.w3[j] * std::clamp(float(sums[j]) * scale + net.b2[j], 0.0f, 1.0f);
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
    View view_p = view(net, pos, p);
    std::copy(net.b1.begin(), net.b1.end(), v);
    for (int c = WHITE; c <= BLACK; c++)
      for (int t = PAWN; t <= KING; t++)
        for (U64 b = pos.pieces[c][t]; b;) {
          const float* row = &net.w1[size_t(feature(view_p, p, c, t, pop_lsb(b))) * n];
          for (int i = 0; i < n; i++) v[i] += row[i];
        }
  }
}

float logit_float(const Network& net, const FloatAccumulator& acc, int side_to_move) {
  int n = net.hidden1;
  float h1[2 * MAX_HIDDEN1];
  for (int i = 0; i < n; i++) h1[i] = std::clamp(acc.v[side_to_move][i], 0.0f, 1.0f);
  if (net.perspectives() == 2)
    for (int i = 0; i < n; i++) h1[n + i] = std::clamp(acc.v[side_to_move ^ 1][i], 0.0f, 1.0f);
  int width = net.perspectives() * n;
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
