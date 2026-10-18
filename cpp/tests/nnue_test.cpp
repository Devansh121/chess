// Network evaluation tests with a deterministic random network (written to a temp file and loaded through
// the normal loader): incremental accumulator updates equal a full refresh at every node of small move
// trees (covering captures, castling, en passant, promotion and, for king buckets, king moves that change
// bucket or mirror half), and colour-flipped positions score the same.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../src/movegen.h"
#include "../src/nnue.h"

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) failures++;
}

std::string write_random_net(int arch, int h1, int h2) {
  std::string path = "nnue_test_arch" + std::to_string(arch) + ".nnue";
  std::ofstream out(path, std::ios::binary);
  uint64_t s = 12345 + arch;
  auto rnd = [&s](float scale) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return scale * (float(s % 20001) / 10000.0f - 1.0f);
  };
  // arch 3: the same 4-bucket map as nn/model.py KING_BUCKETS.
  uint32_t buckets = arch == 3 ? 4 : 1;
  uint8_t map[64] = {};
  for (int sq = 0; sq < 64; sq++) map[sq] = sq < 8 ? (file_of(sq) < 2 ? 0 : 1) : sq < 16 ? 2 : 3;
  uint32_t header[5] = {arch == 3 ? 2u : 1u, uint32_t(arch), uint32_t(buckets * nnue::NUM_FEATURES + 1),
                        uint32_t(h1), uint32_t(h2)};
  float cp_scale = 400.0f;
  out.write("CHNN", 4);
  out.write(reinterpret_cast<char*>(header), sizeof(header));
  out.write(reinterpret_cast<char*>(&cp_scale), sizeof(cp_scale));
  if (arch == 3) {
    out.write(reinterpret_cast<char*>(&buckets), sizeof(buckets));
    out.write(reinterpret_cast<char*>(map), sizeof(map));
  }
  auto put = [&](size_t n, float scale) {
    for (size_t i = 0; i < n; i++) {
      float v = rnd(scale);
      out.write(reinterpret_cast<char*>(&v), sizeof(v));
    }
  };
  put(size_t(buckets * nnue::NUM_FEATURES + 1) * h1, 0.05f);  // w1
  put(h1, 0.1f);                                              // b1
  put(size_t(h2) * (arch == 1 ? 1 : 2) * h1, 0.1f);           // w2
  put(h2, 0.1f);                                    // b2
  put(h2, 0.5f);                                    // w3
  put(1, 0.1f);                                     // b3
  return path;
}

int max_diff(const nnue::Network& net, const nnue::Accumulator& a, const nnue::Accumulator& b) {
  int d = 0;
  for (int p = 0; p < 2; p++)
    for (int i = 0; i < net.hidden1; i++) d = std::max(d, std::abs(int(a.v[p][i]) - int(b.v[p][i])));
  return d;
}

// Walk the legal move tree; at every node compare the incrementally updated accumulator with a refresh.
// Also tracks the largest gap between the quantized and float-reference outputs.
// With `cache`, bucket refreshes go through one RefreshCache shared by the whole tree, so its entries are
// reused from unrelated positions in arbitrary order.
int walk(const nnue::Network& net, const Position& pos, const nnue::Accumulator& acc, int depth, long& nodes,
         float& worst_quant, nnue::RefreshCache* cache = nullptr) {
  nnue::Accumulator fresh;
  nnue::refresh(net, pos, fresh);
  int worst = max_diff(net, acc, fresh);
  nnue::FloatAccumulator ref;
  nnue::refresh_float(net, pos, ref);
  worst_quant = std::max(worst_quant, std::fabs(nnue::logit(net, acc, pos.side) - nnue::logit_float(net, ref, pos.side)));
  nodes++;
  if (depth == 0) return worst;
  MoveList list;
  generate_legal(pos, list);
  for (Move m : list) {
    Position next = pos.make(m);
    nnue::Accumulator child;
    nnue::update(net, pos, next, acc, child, cache);
    worst = std::max(worst, walk(net, next, child, depth - 1, nodes, worst_quant, cache));
  }
  return worst;
}

Position colour_flipped(const Position& p) {
  std::string fen = p.fen();
  std::string placement = fen.substr(0, fen.find(' '));
  std::string ranks[8], out;
  int r = 0;
  for (char c : placement) {
    if (c == '/') r++;
    else ranks[r] += std::isupper(c) ? char(std::tolower(c)) : char(std::toupper(c));
  }
  for (int i = 7; i >= 0; i--) out += ranks[i] + (i ? "/" : "");
  Position m;
  m.set_fen(out + (p.side == WHITE ? " b" : " w") + " - - 0 1");
  return m;
}

}  // namespace

int main() {
  Position::init();
  const char* fens[] = {
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  // castling, captures
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                             // en passant
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",      // promotions
  };
  for (int arch = 1; arch <= 3; arch++) {
    nnue::Network net;
    std::string err = net.load(write_random_net(arch, 64, 16));
    check(err.empty(), "load arch " + std::to_string(arch) + (err.empty() ? "" : ": " + err));
    if (!err.empty()) continue;
    for (const char* fen : fens) {
      Position pos;
      pos.set_fen(fen);
      nnue::Accumulator acc;
      nnue::refresh(net, pos, acc);
      long nodes = 0;
      float worst_quant = 0;
      int worst = walk(net, pos, acc, 3, nodes, worst_quant);
      check(worst == 0, "arch " + std::to_string(arch) + " incremental == refresh exactly over " +
                            std::to_string(nodes) + " nodes: " + fen);
      nnue::RefreshCache cache;
      cache.reset(net);
      long cached_nodes = 0;
      float unused = 0;
      check(walk(net, pos, acc, 3, cached_nodes, unused, &cache) == 0,
            "arch " + std::to_string(arch) + " incremental with refresh cache == refresh exactly");
      // int8 hidden weights and 7-bit activations: about twice the int16 version's rounding error.
      check(worst_quant < 0.04f, "arch " + std::to_string(arch) + " quantized within 0.04 of float logit (max " +
                                     std::to_string(worst_quant) + ")");
      Position flipped = colour_flipped(pos);
      nnue::Accumulator a2;
      nnue::refresh(net, flipped, a2);
      float l1 = nnue::logit(net, acc, pos.side), l2 = nnue::logit(net, a2, flipped.side);
      check(std::fabs(l1 - l2) < 1e-4f, "arch " + std::to_string(arch) + " colour-flipped position scores the same");
    }
  }
  // The network built into the binary (nets/default.nnue), when this build has one.
  {
    nnue::Network net;
    std::string err = net.load_embedded();
    if (err == "this build has no embedded network") {
      std::printf("skip embedded network: %s\n", err.c_str());
    } else {
      check(err.empty(), "embedded network loads" + (err.empty() ? std::string() : ": " + err));
      auto eval = [&](const char* fen) {
        Position p;
        p.set_fen(fen);
        nnue::Accumulator acc;
        nnue::refresh(net, p, acc);
        return nnue::evaluate(net, acc, p.side);
      };
      int start = eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
      int queen_up = eval("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
      int queen_down = eval("4k3/8/8/8/8/8/8/3QK3 b - - 0 1");
      check(std::abs(start) < 100, "embedded net: start position near equal (" + std::to_string(start) + ")");
      check(queen_up > 500 && queen_down < -500, "embedded net: a queen up is winning (" + std::to_string(queen_up) +
                                                     ", " + std::to_string(queen_down) + ")");
    }
  }

  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("all nnue tests passed\n");
  return 0;
}
