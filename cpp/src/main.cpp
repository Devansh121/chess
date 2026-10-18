// UCI front end. The search runs on a worker thread so "stop" and "quit" work mid-search.
// Extra commands: "bench [depth] [net.nnue]" (fixed-depth node count over test positions, for regression checks)
// and "perft N" on the current position.
#include <algorithm>
#include <chrono>
#include <memory>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "eval.h"
#include "movegen.h"
#include "nnue.h"
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

// Soft and hard time limits for this move from the clock: the soft target is an even share of the
// remaining time plus most of the increment; the hard limit allows stretching it in unclear positions
// but never more than 30% of what is left.
void allocate_ms(const Position& pos, double wtime, double btime, double winc, double binc, int movestogo,
                 SearchLimits& limits) {
  double remaining = pos.side == WHITE ? wtime : btime;
  double inc = pos.side == WHITE ? winc : binc;
  if (remaining <= 0) return;
  int mtg = movestogo > 0 ? movestogo : 40;
  double soft = remaining / mtg + 0.75 * inc;
  double hard = std::min(remaining * 0.3, soft * 4) - 10;  // leave a little for overhead
  limits.time_ms = std::max(5.0, hard);
  limits.soft_ms = std::max(5.0, std::min(soft, limits.time_ms));
}

// net_file: "" = the embedded network (hand-written eval if the build has none), "hce" = hand-written.
void bench(int depth, const std::string& net_file) {
  Searcher s(16);
  nnue::Network net;
  if (net_file != "hce") {
    std::string err = net_file.empty() ? net.load_embedded() : net.load(net_file);
    if (err.empty()) {
      s.set_network(&net);
    } else if (!net_file.empty()) {
      std::printf("bench: %s\n", err.c_str());
      return;
    }
  }
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
    bench(argc > 2 ? std::atoi(argv[2]) : 7, argc > 3 ? argv[3] : "");
    return 0;
  }

  Searcher searcher(64);
  nnue::Network network;
  std::string eval_file;
  // The default evaluation is the network built into the binary, if there is one.
  if (network.load_embedded().empty()) {
    eval_file = "<embedded>";
    searcher.set_network(&network);
  }
  // Lazy SMP: helper searchers share the main searcher's transposition table.
  std::vector<std::unique_ptr<Searcher>> helpers;
  auto set_threads = [&](int n) {
    helpers.clear();
    for (int i = 1; i < n; i++) {
      helpers.push_back(std::make_unique<Searcher>(searcher.tt()));
      helpers.back()->set_network(eval_file.empty() ? nullptr : &network);
    }
  };
  auto set_network_all = [&](const nnue::Network* net) {
    searcher.set_network(net);
    for (auto& h : helpers) h->set_network(net);
  };
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
      send("option name EvalFile type string default <embedded>");
      send("option name Threads type spin default 1 min 1 max 256");
      send("uciok");
    } else if (cmd == "isready") {
      send("readyok");
    } else if (cmd == "setoption") {
      std::string tok, name, value;
      in >> tok;  // "name"
      while (in >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
      in >> value;
      if (name == "Threads" && !value.empty()) {
        stop_search();
        set_threads(std::clamp(std::stoi(value), 1, 256));
      } else if (name == "Hash" && !value.empty()) {
        stop_search();
        searcher.resize_hash(std::max(1, std::stoi(value)));
      } else if (name == "EvalFile") {
        stop_search();
        std::string rest;
        std::getline(in, rest);  // paths may contain spaces
        value += rest;
        if (value == "<hce>" || value == "<empty>") {
          set_network_all(nullptr);
          eval_file.clear();
          send("info string using the hand-written evaluation");
        } else {
          bool embedded = value.empty() || value == "<embedded>";
          std::string err = embedded ? network.load_embedded() : network.load(value);
          if (embedded) value = "<embedded>";
          if (err.empty()) {
            eval_file = value;
            set_network_all(&network);
            send("info string loaded network " + value);
          } else {
            set_network_all(nullptr);
            eval_file.clear();
            send("info string failed to load network: " + err);
          }
        }
      }
    } else if (cmd == "ucinewgame") {
      stop_search();
      searcher.new_game();
      for (auto& h : helpers) h->clear_history();
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
      if (limits.time_ms == 0) allocate_ms(pos, wtime, btime, winc, binc, movestogo, limits);
      searcher.stop = false;
      Position root = pos;
      std::vector<U64> hist = history;
      worker = std::thread([&searcher, &helpers, root, hist, limits]() {
        // Helpers search the same position with no limits until the main search finishes; they help
        // through the shared table. Only the main search reports and chooses the move.
        std::vector<std::thread> pool;
        for (auto& h : helpers) {
          Searcher* hp = h.get();
          hp->stop = false;
          pool.emplace_back([hp, root, hist]() { hp->search(root, hist, SearchLimits{}); });
        }
        SearchResult r = searcher.search(root, hist, limits, print_info);
        for (auto& h : helpers) h->stop = true;
        for (auto& th : pool) th.join();
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
    } else if (cmd == "eval") {
      std::string line_out = "eval hand-written " + std::to_string(eval::evaluate(pos));
      if (!eval_file.empty()) {
        nnue::Accumulator acc;
        nnue::refresh(network, pos, acc);
        nnue::FloatAccumulator ref;
        nnue::refresh_float(network, pos, ref);
        char buf[128];
        std::snprintf(buf, sizeof(buf), " network %d logit %.6f float %.6f", nnue::evaluate(network, acc, pos.side),
                      nnue::logit(network, acc, pos.side), nnue::logit_float(network, ref, pos.side));
        line_out += buf;
      }
      send(line_out);
    } else if (cmd == "quit") {
      break;
    }
  }
  stop_search();
  return 0;
}
