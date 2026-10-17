// Move-generator tests: perft node counts from standard test positions, Zobrist hash consistency,
// and FEN round trips. Exits non-zero on the first failure.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../src/movegen.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL %s\n", what.c_str());
    failures++;
  }
}

struct PerftCase {
  const char* name;
  const char* fen;
  int depth;
  U64 nodes;
};

// Widely published perft results for these positions.
const PerftCase CASES[] = {
    {"start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
    {"pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
    {"pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
    {"pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
    {"pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
};

// Walk the tree and compare the incrementally updated hash with one computed from scratch.
void check_hashes(const Position& pos, int depth, const char* name) {
  if (pos.hash != pos.compute_hash()) {
    check(false, std::string("hash mismatch in ") + name + " at " + pos.fen());
    std::exit(1);
  }
  if (depth == 0) return;
  MoveList list;
  generate_legal(pos, list);
  for (Move m : list) check_hashes(pos.make(m), depth - 1, name);
}

}  // namespace

int main() {
  Position::init();
  U64 total = 0;
  auto start = std::chrono::steady_clock::now();
  for (const auto& c : CASES) {
    Position pos;
    check(pos.set_fen(c.fen), std::string("parse ") + c.name);
    check(pos.fen() == c.fen, std::string("fen round trip ") + c.name + ": " + pos.fen());
    U64 n = perft(pos, c.depth);
    total += n;
    std::printf("%-9s depth %d: %llu (expected %llu)\n", c.name, c.depth, (unsigned long long)n,
                (unsigned long long)c.nodes);
    check(n == c.nodes, std::string("perft ") + c.name);
    check_hashes(pos, 3, c.name);
  }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("%llu nodes in %.2f s (%.1f M nodes/s, including hash checks)\n", (unsigned long long)total, secs,
              total / secs / 1e6);
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("all perft tests passed\n");
  return 0;
}
