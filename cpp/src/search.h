// Alpha-beta search (negamax) with iterative deepening, principal variation search, quiescence search,
// transposition table and move ordering (hash move, MVV-LVA captures, killers).
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>

#include "nnue.h"
#include "position.h"
#include "tt.h"

constexpr int MATE = 30000;
constexpr int MATE_BOUND = MATE - 1000;  // above this: a forced mate
constexpr int INF = MATE + 1;

struct SearchLimits {
  int depth = 64;
  double time_ms = 0;  // hard limit, never exceeded; 0 = no time limit
  double soft_ms = 0;  // target: no new iteration once past half of it (stretched when the best move is unstable)
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
  // A searcher with its own transposition table.
  explicit Searcher(size_t hash_mb = 64);
  // A searcher sharing `tt` with others (Lazy SMP helper threads).
  explicit Searcher(TT* shared_tt);
  // Evaluate with this network instead of the hand-written evaluation (nullptr: hand-written).
  void set_network(const nnue::Network* net) { net_ = net; }
  void new_game();
  void clear_history();  // killers and history only (helper threads; the shared table is cleared once)
  void resize_hash(size_t mb) { tt_->resize(mb); }
  TT* tt() { return tt_; }
  U64 nodes() const { return nodes_; }

  // `history`: hashes of earlier positions in the game (for repetition detection), oldest first.
  SearchResult search(const Position& root, const std::vector<U64>& history, const SearchLimits& limits,
                      const std::function<void(const SearchInfo&)>& on_info = nullptr);

  std::atomic<bool> stop{false};

 private:
  int negamax(const Position& pos, int depth, int alpha, int beta, int ply, bool allow_null = true);
  int pvs(const Position& next, int depth, int alpha, int beta, int ply, bool first);
  int quiesce(const Position& pos, int alpha, int beta, int ply);
  int root_search(const Position& pos, int depth, Move prev_best, Move& best, int alpha, int beta);
  bool is_draw(const Position& pos, int ply) const;
  bool time_up();
  void order_moves(const Position& pos, Move* moves, int n, Move hash_move, int ply) const;
  std::vector<Move> extract_pv(const Position& pos, Move first, int max_len) const;

  // Static evaluation of `pos` at `ply` (network accumulators are brought up to date lazily).
  int evaluate(const Position& pos, int ply);
  // Called on entering every node: remembers the position for incremental accumulator updates.
  void enter(const Position& pos, int ply) {
    if (net_ && ply < ACC_STACK) {
      pos_at_[ply] = &pos;
      acc_valid_[ply] = false;
    }
  }

  static constexpr int ACC_STACK = MAX_PLY + 64;  // quiescence can go past MAX_PLY
  const nnue::Network* net_ = nullptr;
  std::vector<nnue::Accumulator> acc_ = std::vector<nnue::Accumulator>(ACC_STACK);
  const Position* pos_at_[ACC_STACK] = {};
  bool acc_valid_[ACC_STACK] = {};
  int eval_stack_[ACC_STACK] = {};  // static eval per ply on the current path (-INF when in check)

  std::unique_ptr<TT> own_tt_;
  TT* tt_ = nullptr;
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
