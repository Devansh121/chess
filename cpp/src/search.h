// Alpha-beta search (negamax) with iterative deepening, principal variation search, quiescence search,
// transposition table and move ordering (hash move, MVV-LVA captures, killers).
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <vector>

#include "position.h"
#include "tt.h"

constexpr int MATE = 30000;
constexpr int MATE_BOUND = MATE - 1000;  // above this: a forced mate
constexpr int INF = MATE + 1;

struct SearchLimits {
  int depth = 64;
  double time_ms = 0;  // 0 = no time limit
  U64 nodes = 0;       // 0 = no node limit
};

struct SearchInfo {
  int depth;
  int score;
  U64 nodes;
  double ms;
  std::vector<Move> pv;
};

struct SearchResult {
  Move best = NO_MOVE;
  int score = 0;
  int depth = 0;
  U64 nodes = 0;
};

class Searcher {
 public:
  explicit Searcher(size_t hash_mb = 64);
  void new_game();
  void resize_hash(size_t mb) { tt_.resize(mb); }

  // `history`: hashes of earlier positions in the game (for repetition detection), oldest first.
  SearchResult search(const Position& root, const std::vector<U64>& history, const SearchLimits& limits,
                      const std::function<void(const SearchInfo&)>& on_info = nullptr);

  std::atomic<bool> stop{false};

 private:
  int negamax(const Position& pos, int depth, int alpha, int beta, int ply, bool allow_null = true);
  int pvs(const Position& next, int depth, int alpha, int beta, int ply, bool first);
  int quiesce(const Position& pos, int alpha, int beta, int ply);
  int root_search(const Position& pos, int depth, Move prev_best, Move& best);
  bool is_draw(const Position& pos, int ply) const;
  bool time_up();
  void order_moves(const Position& pos, Move* moves, int n, Move hash_move, int ply) const;
  std::vector<Move> extract_pv(const Position& pos, Move first, int max_len) const;

  TT tt_;
  Move killers_[MAX_PLY][2];
  int history_[2][64][64];  // quiet-move history: bonus for cutoffs, malus for quiets tried before one
  void update_history(int side, Move m, int bonus);
  std::vector<U64> stack_;  // hashes from the game start to the current node
  size_t root_index_ = 0;
  U64 nodes_ = 0;
  SearchLimits limits_;
  std::chrono::steady_clock::time_point start_;
  bool aborted_ = false;
  bool can_abort_ = true;
};

// Score shown in UCI: "cp N" or "mate N".
std::string format_score(int score);
