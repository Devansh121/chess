"""Alpha-beta search (negamax form) with iterative deepening and a time limit.

Scores are centipawns from the side to move's point of view. Mate scores are
MATE - plies_to_mate, so shorter mates score higher.
"""

import time

import chess

from engine.eval import PIECE_VALUE
from engine.eval import evaluate as evaluate_hce

MATE = 100_000
MATE_BOUND = MATE - 1000  # anything above this is a forced mate
INF = MATE + 1


# Transposition-table bound types: the stored score is exact, a lower bound (search failed high),
# or an upper bound (no move raised alpha).
EXACT, LOWER, UPPER = 0, 1, 2
TT_MAX_ENTRIES = 300_000  # ~100 MB of Python tuples; the table is cleared when full
MAX_PLY = 128

# Move-ordering scores: hash move, then captures (MVV-LVA), queen promotions, killers, quiet moves.
ORDER_HASH = 10_000_000
ORDER_CAPTURE = 1_000_000
ORDER_PROMOTION = 900_000
ORDER_KILLER = 800_000


class Timeout(Exception):
  pass


def _to_tt(score, ply):
  """Mate scores are stored relative to the node, not the root, so they stay right when reached via another path."""
  if score > MATE_BOUND:
    return score + ply
  if score < -MATE_BOUND:
    return score - ply
  return score


def _from_tt(score, ply):
  if score > MATE_BOUND:
    return score - ply
  if score < -MATE_BOUND:
    return score + ply
  return score


def mate_in(score):
  """Moves to mate (positive: we mate, negative: we get mated), or None for a normal score."""
  if score > MATE_BOUND:
    return (MATE - score + 1) // 2
  if score < -MATE_BOUND:
    return -((MATE + score + 1) // 2)
  return None


class Searcher:
  def __init__(self, use_tt=True, evaluate=None):
    """`evaluate(board)` -> centipawns for the side to move; defaults to the hand-written evaluation."""
    self.evaluate = evaluate or evaluate_hce
    self.nodes = 0
    self.deadline = None
    self.stop = False
    self.use_tt = use_tt
    # position key -> (depth, score, bound, best_move); kept between moves of a game
    self.tt = {}
    # two quiet moves per ply that recently caused a beta cutoff
    self.killers = [[None, None] for _ in range(MAX_PLY)]

  def search(self, board, time_limit=None, max_depth=64, info=None):
    """Return (best_move, score, depth) for `board`.

    Searches depth 1, 2, 3, ... until `max_depth` or `time_limit` seconds run out, and
    returns the result of the deepest fully finished depth. `info(depth, score, nodes, secs, pv)`
    is called after each finished depth.
    """
    self.nodes = 0
    self.stop = False
    self.killers = [[None, None] for _ in range(MAX_PLY)]
    start = time.monotonic()
    self.deadline = start + time_limit if time_limit else None
    legal = list(board.legal_moves)
    if not legal:
      return None, 0, 0
    best_move, best_score, done_depth = legal[0], 0, 0
    for depth in range(1, max_depth + 1):
      try:
        score, move = self._root(board, depth, best_move if done_depth else None)
      except Timeout:
        break
      best_move, best_score, done_depth = move, score, depth
      if info:
        info(depth, score, self.nodes, time.monotonic() - start, [move])
      if mate_in(score) is not None and abs(mate_in(score)) * 2 <= depth:
        break  # found a forced mate that the search fully proves; deeper won't change it
    return best_move, best_score, done_depth

  def _check_time(self):
    if self.stop or (self.deadline is not None and (self.nodes & 1023) == 0 and time.monotonic() > self.deadline):
      raise Timeout

  def _root(self, board, depth, prev_best):
    """Search every root move; the previous iteration's best move goes first."""
    alpha, beta = -INF, INF
    best_move = None
    for move in self._ordered(board, prev_best, 0):
      board.push(move)
      try:
        score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
      finally:
        board.pop()
      if score > alpha:
        alpha, best_move = score, move
    return alpha, best_move

  def _negamax(self, board, depth, alpha, beta, ply):
    self.nodes += 1
    self._check_time()

    if board.is_repetition(2) or board.halfmove_clock >= 100 or board.is_insufficient_material():
      return 0

    if depth <= 0:
      return self._leaf(board, alpha, beta, ply)

    key = None
    hash_move = None
    if self.use_tt:
      # python-chess's own position key (pieces, side to move, castling, en passant); pinned in requirements.txt
      key = board._transposition_key()
      entry = self.tt.get(key)
      if entry is not None:
        hash_move = entry[3]  # best move from an earlier search of this position, even a shallower one
        if entry[0] >= depth:
          score, bound = _from_tt(entry[1], ply), entry[2]
          if bound == EXACT or (bound == LOWER and score >= beta) or (bound == UPPER and score <= alpha):
            return score

    alpha_orig = alpha
    best_move = None
    any_move = False
    for move in self._ordered(board, hash_move, ply):
      any_move = True
      board.push(move)
      try:
        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
      finally:
        board.pop()
      if score >= beta:
        self._store(key, depth, beta, LOWER, move, ply)
        if not board.is_capture(move) and ply < MAX_PLY:
          killers = self.killers[ply]
          if killers[0] != move:
            killers[1], killers[0] = killers[0], move
        return beta
      if score > alpha:
        alpha, best_move = score, move

    if not any_move:
      return -(MATE - ply) if board.is_check() else 0
    self._store(key, depth, alpha, EXACT if alpha > alpha_orig else UPPER, best_move, ply)
    return alpha

  def _ordered(self, board, hash_move, ply):
    """Legal moves, most promising first, so alpha-beta finds cutoffs early."""
    killers = self.killers[ply] if ply < MAX_PLY else (None, None)

    def score(move):
      if move == hash_move:
        return ORDER_HASH
      if board.is_capture(move):
        return ORDER_CAPTURE + _mvv_lva(board, move)
      if move.promotion == chess.QUEEN:
        return ORDER_PROMOTION
      if move == killers[0]:
        return ORDER_KILLER
      if move == killers[1]:
        return ORDER_KILLER - 1
      return 0

    return sorted(board.legal_moves, key=score, reverse=True)

  def _store(self, key, depth, score, bound, move, ply):
    if key is None:
      return
    if len(self.tt) >= TT_MAX_ENTRIES:
      self.tt.clear()
    old = self.tt.get(key)
    if old is None or depth >= old[0]:
      self.tt[key] = (depth, _to_tt(score, ply), bound, move)

  def _leaf(self, board, alpha, beta, ply):
    return self._quiesce(board, alpha, beta, ply)

  def _quiesce(self, board, alpha, beta, ply):
    """Search captures until the position is quiet, so the static eval never lands mid-exchange."""
    self.nodes += 1
    self._check_time()

    in_check = board.is_check()
    if in_check:
      # No standing pat in check: every evasion must be searched, or mates go unseen.
      moves = list(board.legal_moves)
      if not moves:
        return -(MATE - ply)
    else:
      stand_pat = self.evaluate(board)
      if stand_pat >= beta:
        return beta
      if stand_pat > alpha:
        alpha = stand_pat
      moves = sorted(board.generate_legal_captures(), key=lambda m: _mvv_lva(board, m), reverse=True)

    for move in moves:
      board.push(move)
      try:
        score = -self._quiesce(board, -beta, -alpha, ply + 1)
      finally:
        board.pop()
      if score >= beta:
        return beta
      if score > alpha:
        alpha = score
    return alpha


def _mvv_lva(board, move):
  """Most valuable victim first, then least valuable attacker."""
  victim = board.piece_type_at(move.to_square) or chess.PAWN  # en passant leaves the target square empty
  attacker = board.piece_type_at(move.from_square)
  return PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker] // 10
