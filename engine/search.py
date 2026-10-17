"""Alpha-beta search (negamax form) with iterative deepening and a time limit.

Scores are centipawns from the side to move's point of view. Mate scores are
MATE - plies_to_mate, so shorter mates score higher.
"""

import time

from engine.eval import evaluate

MATE = 100_000
MATE_BOUND = MATE - 1000  # anything above this is a forced mate
INF = MATE + 1


class Timeout(Exception):
  pass


def mate_in(score):
  """Moves to mate (positive: we mate, negative: we get mated), or None for a normal score."""
  if score > MATE_BOUND:
    return (MATE - score + 1) // 2
  if score < -MATE_BOUND:
    return -((MATE + score + 1) // 2)
  return None


class Searcher:
  def __init__(self):
    self.nodes = 0
    self.deadline = None
    self.stop = False

  def search(self, board, time_limit=None, max_depth=64, info=None):
    """Return (best_move, score, depth) for `board`.

    Searches depth 1, 2, 3, ... until `max_depth` or `time_limit` seconds run out, and
    returns the result of the deepest fully finished depth. `info(depth, score, nodes, secs, pv)`
    is called after each finished depth.
    """
    self.nodes = 0
    self.stop = False
    start = time.monotonic()
    self.deadline = start + time_limit if time_limit else None
    legal = list(board.legal_moves)
    if not legal:
      return None, 0, 0
    best_move, best_score, done_depth = legal[0], 0, 0
    for depth in range(1, max_depth + 1):
      try:
        score, move = self._root(board, depth)
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

  def _root(self, board, depth):
    alpha, beta = -INF, INF
    best_move = None
    for move in board.legal_moves:
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

    any_move = False
    for move in board.legal_moves:
      any_move = True
      board.push(move)
      try:
        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
      finally:
        board.pop()
      if score >= beta:
        return beta
      if score > alpha:
        alpha = score

    if not any_move:
      return -(MATE - ply) if board.is_check() else 0
    return alpha

  def _leaf(self, board, alpha, beta, ply):
    if board.is_checkmate():
      return -(MATE - ply)
    if board.is_stalemate():
      return 0
    return evaluate(board)
