// UCI front end. The search runs on a worker thread so "stop" and "quit" work mid-search.
// Extra commands: "bench [depth]" (fixed-depth node count over test positions, for regression checks)
// and "perft N" on the current position.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "eval.h"
#include "movegen.h"
#include "search.h"

namespace {

const char* NAME = "chess-cpp";
const char* AUTHOR = "Devansh Wassista";
const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const char* BENCH_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "8/8/4k3/8/2p5/8/B2K4/8 w - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
};

void send(const std::string& line) {
  std::cout << line << std::endl;  // endl flushes, which UCI needs
}

Move parse_uci_move(const Position& pos, const std::string& s) {
  MoveList list;
  generate_legal(pos, list);
  for (Move m : list)
    if (move_to_uci(m) == s) return m;
  return NO_MOVE;
}

void print_info(const SearchInfo& info) {
  std::string pv;
  for (Move m : info.pv) pv += " " + move_to_uci(m);
  U64 nps = info.ms > 0 ? U64(info.nodes * 1000.0 / info.ms) : 0;
  send("info depth " + std::to_string(info.depth) + " score " + format_score(info.score) + " nodes " +
       std::to_string(info.nodes) + " nps " + std::to_string(nps) + " time " + std::to_string(U64(info.ms)) + " pv" +
       pv);
}

// Time to spend on this move, from the "go" arguments.
double allocate_ms(const Position& pos, double wtime, double btime, double winc, double binc, int movestogo) {
  double remaining = pos.side == WHITE ? wtime : btime;
  double inc = pos.side == WHITE ? winc : binc;
  if (remaining <= 0) return 0;
  int mtg = movestogo > 0 ? movestogo : 30;
  double t = remaining / mtg + 0.8 * inc;
  return std::max(5.0, std::min(t, remaining / 3) - 10);  // leave a little for overhead
}

void bench(int depth) {
  Searcher s(16);
  U64 total = 0;
  auto start = std::chrono::steady_clock::now();
  for (const char* fen : BENCH_FENS) {
    Position pos;
    pos.set_fen(fen);
    s.new_game();
    SearchLimits limits;
    limits.depth = depth;
    SearchResult r = s.search(pos, {}, limits);
    total += r.nodes;
  }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("bench depth %d: %llu nodes %.2f s %llu nps\n", depth, (unsigned long long)total, secs,
              (unsigned long long)(total / std::max(secs, 1e-9)));
}

}  // namespace

int main(int argc, char** argv) {
  Position::init();
  eval::init();

  if (argc > 1 && std::string(argv[1]) == "bench") {
    bench(argc > 2 ? std::atoi(argv[2]) : 7);
    return 0;
  }

  Searcher searcher(64);
  Position pos;
  pos.set_fen(START_FEN);
  std::vector<U64> history;  // hashes of positions before the current one
  std::thread worker;

  auto stop_search = [&]() {
    searcher.stop = true;
    if (worker.joinable()) worker.join();
  };

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;
    if (cmd == "uci") {
      send(std::string("id name ") + NAME);
      send(std::string("id author ") + AUTHOR);
      send("option name Hash type spin default 64 min 1 max 4096");
      send("uciok");
    } else if (cmd == "isready") {
      send("readyok");
    } else if (cmd == "setoption") {
      std::string tok, name, value;
      in >> tok;  // "name"
      while (in >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
      in >> value;
      if (name == "Hash" && !value.empty()) {
        stop_search();
        searcher.resize_hash(std::max(1, std::stoi(value)));
      }
    } else if (cmd == "ucinewgame") {
      stop_search();
      searcher.new_game();
    } else if (cmd == "position") {
      stop_search();
      std::string tok;
      in >> tok;
      std::string fen = START_FEN;
      if (tok == "fen") {
        fen.clear();
        while (in >> tok && tok != "moves") fen += (fen.empty() ? "" : " ") + tok;
      } else {
        in >> tok;  // "moves" or nothing
      }
      pos.set_fen(fen);
      history.clear();
      while (in >> tok) {
        Move m = parse_uci_move(pos, tok);
        if (m == NO_MOVE) break;
        history.push_back(pos.hash);
        pos = pos.make(m);
      }
    } else if (cmd == "go") {
      stop_search();
      SearchLimits limits;
      double wtime = 0, btime = 0, winc = 0, binc = 0;
      int movestogo = 0;
      std::string tok;
      while (in >> tok) {
        if (tok == "depth") in >> limits.depth;
        else if (tok == "movetime") in >> limits.time_ms;
        else if (tok == "nodes") in >> limits.nodes;
        else if (tok == "wtime") in >> wtime;
        else if (tok == "btime") in >> btime;
        else if (tok == "winc") in >> winc;
        else if (tok == "binc") in >> binc;
        else if (tok == "movestogo") in >> movestogo;
      }
      if (limits.time_ms == 0) limits.time_ms = allocate_ms(pos, wtime, btime, winc, binc, movestogo);
      searcher.stop = false;
      Position root = pos;
      std::vector<U64> hist = history;
      worker = std::thread([&searcher, root, hist, limits]() {
        SearchResult r = searcher.search(root, hist, limits, print_info);
        send("bestmove " + move_to_uci(r.best));
      });
    } else if (cmd == "stop") {
      stop_search();
    } else if (cmd == "perft") {
      int depth = 1;
      in >> depth;
      auto start = std::chrono::steady_clock::now();
      U64 n = perft(pos, depth);
      double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::printf("perft %d: %llu (%.2f s)\n", depth, (unsigned long long)n, secs);
      std::fflush(stdout);
    } else if (cmd == "d") {
      send(pos.fen());
    } else if (cmd == "quit") {
      break;
    }
  }
  stop_search();
  return 0;
}
