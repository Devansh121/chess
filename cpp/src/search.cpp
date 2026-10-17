#include "search.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "eval.h"
#include "movegen.h"
#include "see.h"

namespace {

// Ordering scores: hash move, captures (MVV-LVA), queen promotions, killers, quiet moves.
constexpr int ORDER_HASH = 10'000'000;
constexpr int ORDER_CAPTURE = 1'000'000;
constexpr int ORDER_PROMOTION = 900'000;
constexpr int ORDER_KILLER = 800'000;
constexpr int ORDER_BAD_CAPTURE = 100'000;  // captures that lose material (SEE < 0): after killers, before quiet moves

int mvv_lva(const Position& pos, Move m) {
  int victim = move_flag(m) == EP_CAPTURE ? PAWN : piece_type(pos.board[move_to(m)]);
  int attacker = piece_type(pos.board[move_from(m)]);
  return eval::PIECE_VALUE[victim] * 10 - eval::PIECE_VALUE[attacker] / 10;
}

// Late-move reduction in plies, by remaining depth and number of moves already searched.
int lmr_table[64][64];
struct LmrInit {
  LmrInit() {
    for (int d = 1; d < 64; d++)
      for (int m = 1; m < 64; m++) lmr_table[d][m] = int(0.75 + std::log(d) * std::log(m) / 2.25);
  }
} lmr_init;

// Mate scores are stored relative to the node so they stay correct when reached by another path.
int to_tt(int score, int ply) { return score > MATE_BOUND ? score + ply : score < -MATE_BOUND ? score - ply : score; }
int from_tt(int score, int ply) {
  return score > MATE_BOUND ? score - ply : score < -MATE_BOUND ? score + ply : score;
}

bool insufficient_material(const Position& pos) {
  for (int c = WHITE; c <= BLACK; c++)
    if (pos.pieces[c][PAWN] | pos.pieces[c][ROOK] | pos.pieces[c][QUEEN]) return false;
  int minors = popcount(pos.pieces[WHITE][KNIGHT] | pos.pieces[WHITE][BISHOP] | pos.pieces[BLACK][KNIGHT] |
                        pos.pieces[BLACK][BISHOP]);
  return minors <= 1;
}

}  // namespace

std::string format_score(int score) {
  if (score > MATE_BOUND) return "mate " + std::to_string((MATE - score + 1) / 2);
  if (score < -MATE_BOUND) return "mate " + std::to_string(-((MATE + score + 1) / 2));
  return "cp " + std::to_string(score);
}

Searcher::Searcher(size_t hash_mb) {
  tt_.resize(hash_mb);
  new_game();
}

void Searcher::new_game() {
  tt_.clear();
  std::memset(killers_, 0, sizeof(killers_));
  std::memset(history_, 0, sizeof(history_));
}

// History gravity: bonuses saturate at +/-HISTORY_MAX, so old information fades as new arrives.
constexpr int HISTORY_MAX = 16384;
void Searcher::update_history(int side, Move m, int bonus) {
  int& h = history_[side][move_from(m)][move_to(m)];
  h += bonus - h * std::abs(bonus) / HISTORY_MAX;
}

bool Searcher::time_up() {
  if (!can_abort_) return false;  // depth 1 always completes, so there is always a searched move
  if (stop.load(std::memory_order_relaxed)) return true;
  if (limits_.nodes && nodes_ >= limits_.nodes) return true;
  if (limits_.time_ms > 0 && (nodes_ & 2047) == 0) {
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    if (ms >= limits_.time_ms) return true;
  }
  return false;
}

bool Searcher::is_draw(const Position& pos, int ply) const {
  if (pos.halfmove >= 100 || insufficient_material(pos)) return true;
  // Any earlier occurrence of this position (same side to move) since the last irreversible move.
  int n = int(stack_.size()) - 1;  // stack_.back() is this position
  for (int i = n - 2; i >= 0 && i >= n - pos.halfmove; i -= 2)
    if (stack_[i] == pos.hash) return true;
  (void)ply;
  return false;
}

void Searcher::order_moves(const Position& pos, Move* moves, int n, Move hash_move, int ply) const {
  int scores[MAX_MOVES];
  for (int i = 0; i < n; i++) {
    Move m = moves[i];
    int s = 0;
    if (m == hash_move)
      s = ORDER_HASH;
    else if (is_capture(m))
      s = (see(pos, m) >= 0 ? ORDER_CAPTURE : ORDER_BAD_CAPTURE) + mvv_lva(pos, m);
    else if (is_promotion(m) && promotion_type(m) == QUEEN)
      s = ORDER_PROMOTION;
    else if (ply < MAX_PLY && m == killers_[ply][0])
      s = ORDER_KILLER;
    else if (ply < MAX_PLY && m == killers_[ply][1])
      s = ORDER_KILLER - 1;
    else
      s = history_[pos.side][move_from(m)][move_to(m)];  // in [-HISTORY_MAX, HISTORY_MAX], below killers
    scores[i] = s;
  }
  // Insertion sort, descending (lists are short).
  for (int i = 1; i < n; i++) {
    Move m = moves[i];
    int s = scores[i], j = i - 1;
    while (j >= 0 && scores[j] < s) {
      moves[j + 1] = moves[j];
      scores[j + 1] = scores[j];
      j--;
    }
    moves[j + 1] = m;
    scores[j + 1] = s;
  }
}

int Searcher::evaluate(const Position& pos, int ply) {
  if (!net_) return eval::evaluate(pos);
  if (ply >= ACC_STACK) {
    nnue::Accumulator scratch;
    nnue::refresh(*net_, pos, scratch);
    return nnue::evaluate(*net_, scratch, pos.side);
  }
  // Walk back to the nearest node on the current path whose accumulator is up to date, then apply the
  // move differences forward. Positions with no evaluation never pay for an update.
  int k = ply;
  while (k > 0 && !acc_valid_[k]) k--;
  if (!acc_valid_[k]) {
    nnue::refresh(*net_, *pos_at_[k], acc_[k]);
    acc_valid_[k] = true;
  }
  for (int j = k + 1; j <= ply; j++) {
    nnue::update(*net_, *pos_at_[j - 1], *pos_at_[j], acc_[j - 1], acc_[j]);
    acc_valid_[j] = true;
  }
  return nnue::evaluate(*net_, acc_[ply], pos.side);
}

int Searcher::quiesce(const Position& pos, int alpha, int beta, int ply) {
  enter(pos, ply);
  nodes_++;
  if (time_up()) {
    aborted_ = true;
    return 0;
  }
  // Quiescence results are stored at depth 0, so any table entry for this position is deep enough.
  Move hash_move = NO_MOVE;
  if (const TTEntry* e = tt_.probe(pos.hash)) {
    hash_move = e->move;
    int s = from_tt(e->score, ply);
    if (e->bound == BOUND_EXACT || (e->bound == BOUND_LOWER && s >= beta) || (e->bound == BOUND_UPPER && s <= alpha))
      return s;
  }
  bool in_check = pos.in_check();
  int alpha_orig = alpha;
  MoveList list;
  if (in_check) {
    generate(pos, list);  // all evasions; no standing pat in check
  } else {
    int stand_pat = evaluate(pos, ply);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;
    generate(pos, list, true);
  }
  order_moves(pos, list.moves, list.size, hash_move, MAX_PLY);
  bool any_legal = false;
  Move best = NO_MOVE;
  for (Move m : list) {
    if (!in_check && !is_capture(m)) continue;  // quiet queen promotions are not searched here (as in stage D)
    if (!in_check && see(pos, m) < 0) continue;  // captures that lose material are never the best way out
    if (!is_legal(pos, m)) continue;
    any_legal = true;
    Position next = pos.make(m);
    stack_.push_back(next.hash);
    int score = -quiesce(next, -beta, -alpha, ply + 1);
    stack_.pop_back();
    if (aborted_) return 0;
    if (score >= beta) {
      tt_.store(pos.hash, m, to_tt(beta, ply), 0, BOUND_LOWER);
      return beta;
    }
    if (score > alpha) {
      alpha = score;
      best = m;
    }
  }
  if (in_check && !any_legal) return -(MATE - ply);
  tt_.store(pos.hash, best, to_tt(alpha, ply), 0, alpha > alpha_orig ? BOUND_EXACT : BOUND_UPPER);
  return alpha;
}

// Principal variation search for one child: the first move gets the full window; later moves get a
// zero-width window that only proves "not better than alpha", and are re-searched if that fails.
int Searcher::pvs(const Position& next, int depth, int alpha, int beta, int ply, bool first) {
  if (first || beta - alpha == 1) return -negamax(next, depth, -beta, -alpha, ply);
  int score = -negamax(next, depth, -alpha - 1, -alpha, ply);
  if (score > alpha && score < beta && !aborted_) score = -negamax(next, depth, -beta, -alpha, ply);
  return score;
}

int Searcher::negamax(const Position& pos, int depth, int alpha, int beta, int ply, bool allow_null) {
  enter(pos, ply);
  nodes_++;
  if (time_up()) {
    aborted_ = true;
    return 0;
  }
  if (is_draw(pos, ply)) return 0;
  // Check extension: checks are forcing and have few replies, so look one ply further.
  bool in_check = pos.in_check();
  if (in_check && ply < MAX_PLY - 1) depth++;
  if (depth <= 0) return quiesce(pos, alpha, beta, ply);

  Move hash_move = NO_MOVE;
  if (const TTEntry* e = tt_.probe(pos.hash)) {
    hash_move = e->move;
    if (e->depth >= depth) {
      int s = from_tt(e->score, ply);
      if (e->bound == BOUND_EXACT || (e->bound == BOUND_LOWER && s >= beta) || (e->bound == BOUND_UPPER && s <= alpha))
        return s;
    }
  }

  // Null-move pruning: if passing the turn still holds beta after a reduced search, a real move almost
  // certainly would too. Skipped in check, in PV nodes, right after another null move, and with only
  // pawns left (zugzwang is common there).
  bool pv_node = beta - alpha > 1;
  int static_eval = in_check ? -INF : evaluate(pos, ply);

  // Reverse futility pruning: close to the leaves, if the static eval beats beta by a margin that
  // grows with the remaining depth, the opponent is unlikely to recover in so few moves.
  if (!pv_node && !in_check && depth <= 6 && std::abs(beta) < MATE_BOUND && static_eval - 80 * depth >= beta)
    return static_eval;

  if (allow_null && !pv_node && depth >= 3 && !in_check && pos.has_non_pawn_material(pos.side) &&
      static_eval >= beta) {
    int r = 2 + depth / 6;
    Position next = pos.make_null();
    stack_.push_back(next.hash);
    int score = -negamax(next, depth - 1 - r, -beta, -beta + 1, ply + 1, false);
    stack_.pop_back();
    if (aborted_) return 0;
    if (score >= beta) return beta;
  }

  MoveList list;
  generate(pos, list);
  order_moves(pos, list.moves, list.size, hash_move, ply);
  int alpha_orig = alpha;
  Move best = NO_MOVE;
  int searched = 0, legal = 0;
  Move quiets_tried[MAX_MOVES];
  int n_quiets = 0;
  for (Move m : list) {
    if (!is_legal(pos, m)) continue;
    legal++;
    Position next = pos.make(m);
    bool quiet = !is_capture(m) && !is_promotion(m);
    bool gives_check = next.in_check();

    // Futility pruning: near the leaves, a quiet move that can't lift the static eval to alpha even
    // with a generous margin is skipped. The first move is always searched.
    if (!pv_node && !in_check && depth <= 3 && searched > 0 && quiet && !gives_check &&
        static_eval + 100 + 100 * depth <= alpha)
      continue;

    stack_.push_back(next.hash);
    int score;
    // Late-move reductions: quiet moves sorted late are rarely best, so search them shallower first
    // with a zero window; if one beats alpha anyway, it gets the normal search.
    bool killer = ply < MAX_PLY && (m == killers_[ply][0] || m == killers_[ply][1]);
    if (searched >= 3 && depth >= 3 && quiet && !killer && !in_check && !gives_check) {
      int r = lmr_table[std::min(depth, 63)][std::min(searched, 63)];
      if (pv_node) r = std::max(0, r - 1);
      int reduced = std::max(1, depth - 1 - r);
      score = -negamax(next, reduced, -alpha - 1, -alpha, ply + 1);
      if (score > alpha && reduced < depth - 1 && !aborted_) score = pvs(next, depth - 1, alpha, beta, ply + 1, false);
    } else {
      score = pvs(next, depth - 1, alpha, beta, ply + 1, searched == 0);
    }
    searched++;
    stack_.pop_back();
    if (aborted_) return 0;
    if (score >= beta) {
      tt_.store(pos.hash, m, to_tt(beta, ply), depth, BOUND_LOWER);
      if (quiet) {
        if (ply < MAX_PLY && killers_[ply][0] != m) {
          killers_[ply][1] = killers_[ply][0];
          killers_[ply][0] = m;
        }
        int bonus = std::min(depth * depth, 400);
        update_history(pos.side, m, bonus);
        for (int i = 0; i < n_quiets; i++) update_history(pos.side, quiets_tried[i], -bonus);
      }
      return beta;
    }
    if (quiet) quiets_tried[n_quiets++] = m;
    if (score > alpha) {
      alpha = score;
      best = m;
    }
  }
  if (legal == 0) return in_check ? -(MATE - ply) : 0;
  if (searched == 0) return alpha;  // every legal move was pruned: a fail low, not a mate or stalemate
  tt_.store(pos.hash, best, to_tt(alpha, ply), depth, alpha > alpha_orig ? BOUND_EXACT : BOUND_UPPER);
  return alpha;
}

// Searches every root move inside (alpha, beta). Returns alpha if nothing beat it (fail low, best stays
// NO_MOVE) and beta as soon as a move reaches it (fail high, best is that move).
int Searcher::root_search(const Position& pos, int depth, Move prev_best, Move& best, int alpha, int beta) {
  MoveList list;
  generate_legal(pos, list);
  order_moves(pos, list.moves, list.size, prev_best, 0);
  best = NO_MOVE;
  int searched = 0;
  for (Move m : list) {
    Position next = pos.make(m);
    stack_.push_back(next.hash);
    int score = pvs(next, depth - 1, alpha, beta, 1, searched == 0);
    searched++;
    stack_.pop_back();
    if (aborted_) return 0;
    if (score > alpha) {
      alpha = score;
      best = m;
      if (alpha >= beta) return beta;
    }
  }
  return alpha;
}

std::vector<Move> Searcher::extract_pv(const Position& root, Move first, int max_len) const {
  std::vector<Move> pv{first};
  Position pos = root.make(first);
  for (int i = 1; i < max_len; i++) {
    const TTEntry* e = tt_.probe(pos.hash);
    if (!e || e->move == NO_MOVE) break;
    MoveList legal;
    generate_legal(pos, legal);
    if (std::find(legal.begin(), legal.end(), e->move) == legal.end()) break;
    pv.push_back(e->move);
    pos = pos.make(e->move);
  }
  return pv;
}

SearchResult Searcher::search(const Position& root, const std::vector<U64>& history, const SearchLimits& limits,
                              const std::function<void(const SearchInfo&)>& on_info) {
  limits_ = limits;
  start_ = std::chrono::steady_clock::now();
  nodes_ = 0;
  aborted_ = false;
  std::memset(killers_, 0, sizeof(killers_));
  stack_ = history;
  stack_.push_back(root.hash);
  enter(root, 0);
  tt_.new_search();

  SearchResult result;
  MoveList legal;
  generate_legal(root, legal);
  if (legal.size == 0) return result;
  result.best = legal.moves[0];

  for (int depth = 1; depth <= limits.depth; depth++) {
    can_abort_ = depth > 1;
    Move best = NO_MOVE;
    Move prev = result.depth ? result.best : NO_MOVE;
    int score;
    if (depth >= 5) {
      // Aspiration window around the previous score; widen the failing side and re-search on a miss.
      int delta = 50;
      int a = std::max(result.score - delta, -INF), b = std::min(result.score + delta, INF);
      for (;;) {
        score = root_search(root, depth, prev, best, a, b);
        if (aborted_) break;
        if (score <= a) {
          a = std::max(a - delta, -INF);
        } else if (score >= b) {
          b = std::min(b + delta, INF);
          prev = best;  // the fail-high move goes first in the re-search
        } else {
          break;
        }
        delta *= 2;
        if (delta > 1000) a = -INF, b = INF;
      }
    } else {
      score = root_search(root, depth, prev, best, -INF, INF);
    }
    if (aborted_) break;
    result.best = best;
    result.score = score;
    result.depth = depth;
    result.nodes = nodes_;
    if (on_info) {
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
      on_info(SearchInfo{depth, score, nodes_, ms, extract_pv(root, best, depth)});
    }
    // A mate the search has fully proven won't change with more depth.
    if ((score > MATE_BOUND && (MATE - score) <= depth) || (score < -MATE_BOUND && (MATE + score) <= depth)) break;
  }
  result.nodes = nodes_;
  return result;
}
