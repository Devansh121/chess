// Search and evaluation tests: mates, tactics, draws, evaluation symmetry, and agreement with the
// Python evaluation on a few positions (the C++ eval is a port with identical numbers).
#include <cstdio>
#include <string>

#include "../src/eval.h"
#include "../src/movegen.h"
#include "../src/search.h"
#include "../src/see.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) failures++;
}

SearchResult search_fen(const char* fen, int depth, Searcher& s) {
  Position pos;
  pos.set_fen(fen);
  s.new_game();
  SearchLimits limits;
  limits.depth = depth;
  return s.search(pos, {}, limits);
}

Position mirrored(const Position& p) {
  // Swap colours and flip ranks, via FEN.
  std::string fen = p.fen(), out;
  std::string placement = fen.substr(0, fen.find(' '));
  std::string rest = fen.substr(fen.find(' ') + 1);
  std::string ranks[8];
  int r = 0;
  for (char c : placement) {
    if (c == '/') r++;
    else ranks[r] += std::isupper(c) ? char(std::tolower(c)) : char(std::toupper(c));
  }
  for (int i = 7; i >= 0; i--) out += ranks[i] + (i ? "/" : "");
  char side = rest[0] == 'w' ? 'b' : 'w';
  Position m;
  m.set_fen(out + " " + side + " - - 0 1");
  return m;
}

}  // namespace

int main() {
  Position::init();
  eval::init();
  Searcher s(16);

  auto r = search_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 4, s);
  check(move_to_uci(r.best) == "a1a8" && r.score == MATE - 1, "back-rank mate in 1: " + move_to_uci(r.best));

  r = search_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 4, s);
  check(move_to_uci(r.best) == "h5f7", "scholar's mate: " + move_to_uci(r.best));

  r = search_fen("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1", 5, s);
  check(r.score == MATE - 3, "two-rook mate in 2 found: " + format_score(r.score));

  r = search_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 3, s);
  check(move_to_uci(r.best) == "d1d5", "takes the free queen: " + move_to_uci(r.best));

  r = search_fen("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", 1, s);
  check(move_to_uci(r.best) != "d1d5" && r.score > 600, "quiescence sees the recapture at depth 1");

  r = search_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3, s);
  check(r.best == NO_MOVE, "stalemate: no move");

  // Repetition: after Ng1-f3-g1 / Ng8-f6-g8 the start position repeats; searching it with that
  // history must treat a repeat as a draw (score 0 for the side that can force it is not required,
  // but the search must not crash and must return a legal move).
  {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::vector<U64> hist;
    for (const char* mv : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
      MoveList l;
      generate_legal(pos, l);
      for (Move m : l)
        if (move_to_uci(m) == mv) {
          hist.push_back(pos.hash);
          pos = pos.make(m);
          break;
        }
    }
    SearchLimits limits;
    limits.depth = 4;
    s.new_game();
    auto rr = s.search(pos, hist, limits);
    check(rr.best != NO_MOVE, "search with game history returns a move");
  }

  // Evaluation: start position balanced, colour-flipped positions mirror, values match the Python eval.
  Position p;
  p.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  check(eval::evaluate_white(p) == 0, "start position evaluates to 0");
  const char* fens[] = {"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8",
                        "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - - 0 1"};
  for (const char* f : fens) {
    p.set_fen(f);
    check(eval::evaluate_white(p) == -eval::evaluate_white(mirrored(p)), std::string("eval mirrors: ") + f);
  }
  // Values from engine/eval.py's evaluate_white on the same positions.
  struct Ref {
    const char* fen;
    int white;
  } refs[] = {{"4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 899},
              {"r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8", 81}};
  for (auto& ref : refs) {
    p.set_fen(ref.fen);
    check(eval::evaluate_white(p) == ref.white,
          std::string("matches Python eval (") + std::to_string(eval::evaluate_white(p)) + " vs " +
              std::to_string(ref.white) + "): " + ref.fen);
  }

  // Static exchange evaluation on hand-built positions.
  struct SeeCase {
    const char* fen;
    const char* move;
    int expected;
    const char* what;
  } see_cases[] = {
      {"4k3/8/8/4n3/8/8/8/4RK2 w - - 0 1", "e1e5", 320, "undefended knight"},
      {"4k3/8/3p4/4p3/8/8/8/4RK2 w - - 0 1", "e1e5", -400, "rook takes a pawn defended by a pawn"},
      {"4k3/8/5p2/4n3/3P4/8/8/4K3 w - - 0 1", "d4e5", 220, "pawn takes a knight defended by a pawn"},
      {"4r1k1/8/8/4p3/8/8/4R3/4RK2 w - - 0 1", "e2e5", 100, "x-ray: the second rook backs up the first"},
      {"4r1k1/8/8/4p3/8/8/4R3/5K2 w - - 0 1", "e2e5", -400, "same without the backup rook"},
  };
  for (auto& c : see_cases) {
    Position sp;
    sp.set_fen(c.fen);
    MoveList l;
    generate_legal(sp, l);
    Move mv = NO_MOVE;
    for (Move m : l)
      if (move_to_uci(m) == c.move) mv = m;
    int got = mv == NO_MOVE ? -99999 : see(sp, mv);
    check(got == c.expected, std::string("SEE ") + c.what + ": " + std::to_string(got) + " (expected " +
                                 std::to_string(c.expected) + ")");
  }

  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("all search tests passed\n");
  return 0;
}
