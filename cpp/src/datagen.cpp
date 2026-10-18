// Self-play data generation: `chess-engine datagen GAMES NODES SEED OUT [RANDOM_PLIES] [OPENINGS]`.
//
// Plays GAMES games of the engine against itself with the default evaluation, about NODES nodes per
// move (each move draws a budget within +/-30%, so games from the same start diverge), each starting
// from a random line of OPENINGS (one UCI move sequence per line, e.g. tools/openings.txt; none = the
// start position) followed by RANDOM_PLIES random legal moves, all seeded. Writes one line per quiet
// position (side to move not in check, chosen move not a capture or promotion, score not a mate):
//   FEN <TAB> search score (centipawns, White's view) <TAB> game result (1 / 0.5 / 0, White's view)
// Games end on mate, stalemate, repetition, the 50-move rule, insufficient material, 400 plies (draw),
// or when one side's score stays beyond 2500 cp for 8 plies (adjudicated win).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "movegen.h"
#include "nnue.h"
#include "search.h"

namespace {

struct Rng {
  U64 s;
  U64 next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
};

struct Sample {
  std::string fen;
  int score_white;
};

bool insufficient(const Position& p) {
  for (int c = WHITE; c <= BLACK; c++)
    if (p.pieces[c][PAWN] | p.pieces[c][ROOK] | p.pieces[c][QUEEN]) return false;
  return popcount(p.pieces[WHITE][KNIGHT] | p.pieces[WHITE][BISHOP] | p.pieces[BLACK][KNIGHT] |
                  p.pieces[BLACK][BISHOP]) <= 1;
}

bool repeated(const std::vector<U64>& history, const Position& p) {
  int n = int(history.size());
  for (int i = n - 2; i >= 0 && i >= n - p.halfmove; i -= 2)
    if (history[i] == p.hash) return true;
  return false;
}

}  // namespace

int run_datagen(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: chess-engine datagen GAMES NODES SEED OUT [RANDOM_PLIES]\n");
    return 1;
  }
  int games = std::atoi(argv[2]);
  U64 nodes = std::strtoull(argv[3], nullptr, 10);
  Rng rng{std::strtoull(argv[4], nullptr, 10) * 0x9E3779B97F4A7C15ULL + 1};
  const char* out_path = argv[5];
  int random_plies = argc > 6 ? std::atoi(argv[6]) : 8;
  std::vector<std::vector<std::string>> openings;
  if (argc > 7) {
    std::ifstream f(argv[7]);
    for (std::string line; std::getline(f, line);) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream words(line);
      std::vector<std::string> moves;
      for (std::string w; words >> w;) moves.push_back(w);
      if (!moves.empty()) openings.push_back(moves);
    }
    if (openings.empty()) {
      std::fprintf(stderr, "no openings in %s\n", argv[7]);
      return 1;
    }
  }

  nnue::Network net;
  Searcher searcher(16);
  if (net.load_embedded().empty()) searcher.set_network(&net);
  std::FILE* out = std::fopen(out_path, "a");
  if (!out) {
    std::fprintf(stderr, "cannot open %s\n", out_path);
    return 1;
  }

  long positions = 0;
  for (int g = 0; g < games; g++) {
    Position pos;
    pos.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::vector<U64> history;
    bool ok = true;
    if (!openings.empty()) {
      for (const std::string& uci : openings[rng.next() % openings.size()]) {
        MoveList legal;
        generate_legal(pos, legal);
        Move found = NO_MOVE;
        for (Move m : legal)
          if (move_to_uci(m) == uci) found = m;
        if (found == NO_MOVE) break;  // a malformed line just ends early
        history.push_back(pos.hash);
        pos = pos.make(found);
      }
    }
    for (int i = 0; i < random_plies; i++) {
      MoveList legal;
      generate_legal(pos, legal);
      if (legal.size == 0) {
        ok = false;
        break;
      }
      history.push_back(pos.hash);
      pos = pos.make(legal.moves[rng.next() % legal.size]);
    }
    if (!ok) {
      g--;
      continue;
    }

    searcher.new_game();
    std::vector<Sample> samples;
    double result = 0.5;  // White's view
    int decisive_run = 0, decisive_sign = 0;
    for (int ply = 0; ply < 400; ply++) {
      if (pos.halfmove >= 100 || insufficient(pos) || repeated(history, pos)) break;
      SearchLimits limits;
      limits.nodes = U64(double(nodes) * (0.7 + 0.6 * double(rng.next() % 1000) / 1000.0));
      SearchResult r = searcher.search(pos, history, limits);
      if (r.best == NO_MOVE) {  // no legal move: mate or stalemate
        if (pos.in_check()) result = pos.side == WHITE ? 0.0 : 1.0;
        break;
      }
      int white_score = pos.side == WHITE ? r.score : -r.score;
      if (!pos.in_check() && !is_capture(r.best) && !is_promotion(r.best) && std::abs(r.score) < MATE_BOUND)
        samples.push_back({pos.fen(), white_score});
      // Adjudicate games that are clearly decided.
      int sign = white_score > 2500 ? 1 : white_score < -2500 ? -1 : 0;
      decisive_run = (sign != 0 && sign == decisive_sign) ? decisive_run + 1 : (sign != 0 ? 1 : 0);
      decisive_sign = sign;
      if (decisive_run >= 8) {
        result = sign > 0 ? 1.0 : 0.0;
        break;
      }
      history.push_back(pos.hash);
      pos = pos.make(r.best);
    }
    for (const Sample& s : samples) std::fprintf(out, "%s\t%d\t%.1f\n", s.fen.c_str(), s.score_white, result);
    positions += long(samples.size());
    if ((g + 1) % 10 == 0) {
      std::fflush(out);
      std::fprintf(stderr, "datagen: %d/%d games, %ld positions\n", g + 1, games, positions);
    }
  }
  std::fclose(out);
  std::fprintf(stderr, "datagen: done, %d games, %ld positions\n", games, positions);
  return 0;
}
