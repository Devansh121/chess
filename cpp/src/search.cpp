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


// Hands out a node's moves in stages, doing each stage's work only if the search gets that far:
// hash move (no scoring at all), captures by MVV-LVA with SEE computed only for the capture about to be
// returned (losers set aside), quiet queen promotions, killers, the set-aside losing captures, then quiet
// moves scored by history only on reaching them. The best remaining move of a stage is picked each call
// instead of sorting everything up front, since most nodes cut off after one or two moves.
class MovePicker {
 public:
  MovePicker(const Position& pos, const MoveList& list, Move hash_move, const Move* killers, const int (*history)[64],
             Move counter = NO_MOVE)
      : pos_(pos), list_(list), hash_(hash_move), history_(history), counter_(counter) {
    killers_[0] = killers ? killers[0] : NO_MOVE;
    killers_[1] = killers ? killers[1] : NO_MOVE;
  }

  Move next() {
    switch (stage_) {
      case HASH:
        stage_ = SPLIT;
        for (int i = 0; i < list_.size; i++)
          if (list_.moves[i] == hash_) return hash_;
        hash_ = NO_MOVE;  // not a move here (or none stored)
        [[fallthrough]];
      case SPLIT:
        split();
        stage_ = CAPTURES;
        [[fallthrough]];
      case CAPTURES:
        while (cap_cur_ < n_caps_) {
          Move m = pick(caps_, cap_scores_, cap_cur_, n_caps_);
          if (see(pos_, m) >= 0) return m;
          bad_[n_bad_++] = m;  // still in MVV-LVA order
        }
        stage_ = PROMOTIONS;
        [[fallthrough]];
      case PROMOTIONS:
        if (promo_cur_ < n_promos_) return promos_[promo_cur_++];
        stage_ = KILLERS;
        [[fallthrough]];
      case KILLERS:
        while (killer_cur_ < 2) {
          Move k = killers_[killer_cur_++];
          if (k != NO_MOVE && k != hash_ && in_quiets(k)) return k;
          if (killer_cur_ == 1 && k != NO_MOVE) killers_[0] = NO_MOVE;  // not playable here; don't skip it later
          if (killer_cur_ == 2 && k != NO_MOVE && !in_quiets(k)) killers_[1] = NO_MOVE;
        }
        stage_ = COUNTER;
        [[fallthrough]];
      case COUNTER:
        stage_ = BAD_CAPTURES;
        if (counter_ != NO_MOVE && counter_ != hash_ && counter_ != killers_[0] && counter_ != killers_[1] &&
            in_quiets(counter_))
          return counter_;
        counter_ = NO_MOVE;  // not returned, so don't skip it among the quiets
        [[fallthrough]];
      case BAD_CAPTURES:
        if (bad_cur_ < n_bad_) return bad_[bad_cur_++];
        stage_ = QUIETS;
        for (int i = 0; i < n_quiets_; i++) quiet_scores_[i] = history_[move_from(quiets_[i])][move_to(quiets_[i])];
        [[fallthrough]];
      case QUIETS:
        while (quiet_cur_ < n_quiets_) {
          Move m = pick(quiets_, quiet_scores_, quiet_cur_, n_quiets_);
          if (m != killers_[0] && m != killers_[1] && m != counter_) return m;  // already returned
        }
        stage_ = DONE;
        [[fallthrough]];
      case DONE:
        return NO_MOVE;
    }
    return NO_MOVE;
  }

 private:
  enum Stage { HASH, SPLIT, CAPTURES, PROMOTIONS, KILLERS, COUNTER, BAD_CAPTURES, QUIETS, DONE };

  void split() {
    for (int i = 0; i < list_.size; i++) {
      Move m = list_.moves[i];
      if (m == hash_) continue;
      if (is_capture(m)) {
        caps_[n_caps_] = m;
        cap_scores_[n_caps_++] = mvv_lva(pos_, m);
      } else if (is_promotion(m) && promotion_type(m) == QUEEN) {
        promos_[n_promos_++] = m;
      } else {
        quiets_[n_quiets_++] = m;
      }
    }
  }

  bool in_quiets(Move m) const {
    for (int i = 0; i < n_quiets_; i++)
      if (quiets_[i] == m) return true;
    return false;
  }

  // Selection step: move the highest-scored remaining entry to position `cur` and return it.
  static Move pick(Move* moves, int* scores, int& cur, int n) {
    int best = cur;
    for (int i = cur + 1; i < n; i++)
      if (scores[i] > scores[best]) best = i;
    std::swap(moves[cur], moves[best]);
    std::swap(scores[cur], scores[best]);
    return moves[cur++];
  }

  const Position& pos_;
  const MoveList& list_;
  Move hash_;
  Move killers_[2];
  const int (*history_)[64];
  Move counter_;
  Stage stage_ = HASH;
  Move caps_[MAX_MOVES], bad_[MAX_MOVES], promos_[MAX_MOVES], quiets_[MAX_MOVES];
  int cap_scores_[MAX_MOVES], quiet_scores_[MAX_MOVES];
  int n_caps_ = 0, n_bad_ = 0, n_promos_ = 0, n_quiets_ = 0;
  int cap_cur_ = 0, bad_cur_ = 0, promo_cur_ = 0, quiet_cur_ = 0, killer_cur_ = 0;
};

}  // namespace

std::string format_score(int score) {
  if (score > MATE_BOUND) return "mate " + std::to_string((MATE - score + 1) / 2);
  if (score < -MATE_BOUND) return "mate " + std::to_string(-((MATE + score + 1) / 2));
  return "cp " + std::to_string(score);
}

Searcher::Searcher(size_t hash_mb) : own_tt_(std::make_unique<TT>()), tt_(own_tt_.get()) {
  tt_->resize(hash_mb);
  new_game();
}

Searcher::Searcher(TT* shared_tt) : tt_(shared_tt) { clear_history(); }

void Searcher::clear_history() {
  std::memset(killers_, 0, sizeof(killers_));
  std::memset(history_, 0, sizeof(history_));
  std::memset(countermoves_, 0, sizeof(countermoves_));
  std::memset(correction_, 0, sizeof(correction_));
}

namespace {
inline size_t pawn_index(const Position& pos, int size) {
  U64 k = pos.pieces[WHITE][PAWN] * 0x9E3779B97F4A7C15ULL ^ pos.pieces[BLACK][PAWN] * 0xC2B2AE3D27D4EB4FULL;
  return size_t((k ^ (k >> 29)) & U64(size - 1));
}
constexpr int CORR_GRAIN = 256, CORR_LIMIT = 256 * 128;  // entries in 1/256 cp, at most +/-128 cp
}  // namespace

int Searcher::correction(const Position& pos) const {
  return correction_[pos.side][pawn_index(pos, CORR_SIZE)] / CORR_GRAIN;
}

void Searcher::update_correction(const Position& pos, int depth, int diff) {
  int& c = correction_[pos.side][pawn_index(pos, CORR_SIZE)];
  int w = std::min(depth + 1, 16);
  c = std::clamp((c * (256 - w) + diff * CORR_GRAIN * w) / 256, -CORR_LIMIT, CORR_LIMIT);
}

void Searcher::new_game() {
  tt_->clear();
  clear_history();
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
  EvalCacheEntry& cached = eval_cache_[pos.hash & (EVAL_CACHE_SIZE - 1)];
  if (cached.valid && cached.key == pos.hash) return cached.score;
  int score = evaluate_network(pos, ply);
  cached = EvalCacheEntry{pos.hash, score, 1};
  return score;
}

int Searcher::evaluate_network(const Position& pos, int ply) {
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
    nnue::update(*net_, *pos_at_[j - 1], *pos_at_[j], acc_[j - 1], acc_[j], &refresh_cache_);
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
  if (TTEntry e; tt_->probe(pos.hash, e)) {
    hash_move = e.move;
    int s = from_tt(e.score, ply);
    if (e.bound == BOUND_EXACT || (e.bound == BOUND_LOWER && s >= beta) || (e.bound == BOUND_UPPER && s <= alpha))
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
  MovePicker picker(pos, list, hash_move, nullptr, history_[pos.side]);
  bool any_legal = false;
  Move best = NO_MOVE;
  for (Move m; (m = picker.next()) != NO_MOVE;) {
    if (!in_check && !is_capture(m)) continue;  // quiet queen promotions are not searched here (as in stage D)
    if (!in_check && see(pos, m) < 0) continue;  // captures that lose material are never the best way out
    Position next = pos.make(m);
    if (next.attacked(next.king_sq(pos.side), next.side)) continue;  // left our king in check
    any_legal = true;
    tt_->prefetch(next.hash);
    stack_.push_back(next.hash);
    int score = -quiesce(next, -beta, -alpha, ply + 1);
    stack_.pop_back();
    if (aborted_) return 0;
    if (score >= beta) {
      tt_->store(pos.hash, m, to_tt(beta, ply), 0, BOUND_LOWER);
      return beta;
    }
    if (score > alpha) {
      alpha = score;
      best = m;
    }
  }
  if (in_check && !any_legal) return -(MATE - ply);
  tt_->store(pos.hash, best, to_tt(alpha, ply), 0, alpha > alpha_orig ? BOUND_EXACT : BOUND_UPPER);
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
  int tt_score = 0, tt_bound = BOUND_NONE;
  if (TTEntry e; tt_->probe(pos.hash, e)) {
    hash_move = e.move;
    tt_score = from_tt(e.score, ply);
    tt_bound = e.bound;
    if (e.depth >= depth) {
      int s = tt_score;
      if (e.bound == BOUND_EXACT || (e.bound == BOUND_LOWER && s >= beta) || (e.bound == BOUND_UPPER && s <= alpha))
        return s;
    }
  }

  // Null-move pruning: if passing the turn still holds beta after a reduced search, a real move almost
  // certainly would too. Skipped in check, in PV nodes, right after another null move, and with only
  // pawns left (zugzwang is common there).
  bool pv_node = beta - alpha > 1;
  // Internal iterative reduction: with no stored best move, ordering here will be poor, so search one
  // ply shallower; the result fills in a hash move for when this position comes back deeper.
  if (hash_move == NO_MOVE && depth >= 4) depth--;
  int raw_eval = in_check ? -INF : evaluate(pos, ply);
  int static_eval = in_check ? -INF : std::clamp(raw_eval + correction(pos), -MATE_BOUND + 1, MATE_BOUND - 1);
  // "Improving": better than this side's eval two plies ago, so reverse futility pruning is more cautious.
  if (ply < ACC_STACK) eval_stack_[ply] = static_eval;
  bool improving = !in_check && ply >= 2 && eval_stack_[ply - 2] != -INF && static_eval > eval_stack_[ply - 2];
  // A stored search score is a better estimate than the static eval when its bound points the right
  // way; the pruning decisions below use it.
  if (!in_check && std::abs(tt_score) < MATE_BOUND &&
      (tt_bound == BOUND_EXACT || (tt_bound == BOUND_LOWER && tt_score > static_eval) ||
       (tt_bound == BOUND_UPPER && tt_score < static_eval)))
    static_eval = tt_score;

  // Reverse futility pruning: close to the leaves, if the static eval beats beta by a margin that
  // grows with the remaining depth, the opponent is unlikely to recover in so few moves.
  if (!pv_node && !in_check && depth <= 6 && std::abs(beta) < MATE_BOUND && static_eval - 80 * (depth - improving) >= beta)
    return static_eval;

  if (allow_null && !pv_node && depth >= 3 && !in_check && pos.has_non_pawn_material(pos.side) &&
      static_eval >= beta) {
    int r = 2 + depth / 6;
    Position next = pos.make_null();
    stack_.push_back(next.hash);
    move_stack_[ply] = NO_MOVE;
    int score = -negamax(next, depth - 1 - r, -beta, -beta + 1, ply + 1, false);
    stack_.pop_back();
    if (aborted_) return 0;
    if (score >= beta) return beta;
  }

  MoveList list;
  generate(pos, list);
  // Counter move: the quiet reply that last refuted the opponent's previous move (piece + destination).
  Move prev = ply >= 1 ? move_stack_[ply - 1] : NO_MOVE;
  Move counter = prev != NO_MOVE ? countermoves_[pos.board[move_to(prev)]][move_to(prev)] : NO_MOVE;
  MovePicker picker(pos, list, hash_move, ply < MAX_PLY ? killers_[ply] : nullptr, history_[pos.side], counter);
  int alpha_orig = alpha;
  Move best = NO_MOVE;
  int searched = 0, legal = 0;
  Move quiets_tried[MAX_MOVES];
  int n_quiets = 0;
  for (Move m; (m = picker.next()) != NO_MOVE;) {
    Position next = pos.make(m);
    if (next.attacked(next.king_sq(pos.side), next.side)) continue;  // left our king in check
    legal++;
    tt_->prefetch(next.hash);
    bool quiet = !is_capture(m) && !is_promotion(m);
    bool gives_check = next.in_check();

    // Futility pruning: near the leaves, a quiet move that can't lift the static eval to alpha even
    // with a generous margin is skipped. The first move is always searched.
    if (!pv_node && !in_check && depth <= 3 && searched > 0 && quiet && !gives_check &&
        static_eval + 100 + 100 * depth <= alpha)
      continue;
    if (!pv_node && !in_check && searched > 0 && alpha > -MATE_BOUND && !gives_check && depth <= 8) {
      // Late-move pruning: with good ordering, quiet moves this far down the list almost never matter.
      if (quiet && searched >= (3 + depth * depth) / (2 - improving)) continue;
      // SEE pruning: moves that lose material on the exchange, by more than the depth can make up for.
      if (see(pos, m) < (quiet ? -60 * depth : -100 * depth)) continue;
    }

    stack_.push_back(next.hash);
    move_stack_[ply] = m;
    int score;
    // Late-move reductions: quiet moves sorted late are rarely best, so search them shallower first
    // with a zero window; if one beats alpha anyway, it gets the normal search.
    bool killer = ply < MAX_PLY && (m == killers_[ply][0] || m == killers_[ply][1]);
    if (searched >= 3 && depth >= 3 && quiet && !killer && !in_check && !gives_check) {
      int r = lmr_table[std::min(depth, 63)][std::min(searched, 63)];
      if (pv_node) r--;
      if (!improving) r++;
      r -= history_[pos.side][move_from(m)][move_to(m)] / 8192;  // -2..+2: trust the move ordering's history
      r = std::max(0, r);
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
      tt_->store(pos.hash, m, to_tt(beta, ply), depth, BOUND_LOWER);
      if (quiet && !in_check && beta > static_eval && std::abs(beta) < MATE_BOUND)
        update_correction(pos, depth, beta - raw_eval);  // a lower bound above the eval: the eval was low
      if (quiet) {
        if (ply < MAX_PLY && killers_[ply][0] != m) {
          killers_[ply][1] = killers_[ply][0];
          killers_[ply][0] = m;
        }
        if (prev != NO_MOVE) countermoves_[pos.board[move_to(prev)]][move_to(prev)] = m;
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
  bool exact = alpha > alpha_orig;
  if (!in_check && (best == NO_MOVE || (!is_capture(best) && !is_promotion(best))) && std::abs(alpha) < MATE_BOUND &&
      (exact || alpha < static_eval))  // exact, or an upper bound below the eval: the eval was high
    update_correction(pos, depth, alpha - raw_eval);
  tt_->store(pos.hash, best, to_tt(alpha, ply), depth, alpha > alpha_orig ? BOUND_EXACT : BOUND_UPPER);
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
    tt_->prefetch(next.hash);
    stack_.push_back(next.hash);
    move_stack_[0] = m;
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
    TTEntry e;
    if (!tt_->probe(pos.hash, e) || e.move == NO_MOVE) break;
    MoveList legal;
    generate_legal(pos, legal);
    if (std::find(legal.begin(), legal.end(), e.move) == legal.end()) break;
    pv.push_back(e.move);
    pos = pos.make(e.move);
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
  eval_stack_[0] = root.in_check() ? -INF : evaluate(root, 0);
  tt_->new_search();

  SearchResult result;
  MoveList legal;
  generate_legal(root, legal);
  if (legal.size == 0) return result;
  result.best = legal.moves[0];

  double time_scale = 1.0;
  Move prev_best_move = NO_MOVE;
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
    // Soft time limit: an iteration usually takes longer than all previous ones together, so don't start
    // one past half the target. A changed best move means the position is unclear: allow more time.
    if (limits.soft_ms > 0) {
      bool changed = depth > 1 && best != prev_best_move;
      time_scale = std::clamp(time_scale * (changed ? 1.3 : 0.92), 0.7, 2.5);
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
      if (ms >= 0.5 * limits.soft_ms * time_scale) break;
    }
    prev_best_move = best;
  }
  result.nodes = nodes_;
  return result;
}
