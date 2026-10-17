"""Hand-written evaluation: material + piece-square tables, tapered between middlegame and endgame.

Scores are centipawns from White's point of view; `evaluate` flips them for the side to move.
Tables are built from a few simple rules rather than tuned numbers, so each value has a reason.
"""

import chess

PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# Game phase: 24 with all minor/major pieces on the board, 0 with only kings and pawns.
PHASE_WEIGHT = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}
MAX_PHASE = 24

BISHOP_PAIR = 30


def _centrality(sq):
  """0 on the corners .. 6 on the four centre squares."""
  f, r = chess.square_file(sq), chess.square_rank(sq)
  return 7 - (abs(f - 3.5) + abs(r - 3.5))


def _build_tables():
  """Tables indexed by square from White's side (a1 = 0). Black uses the mirrored square."""
  mg = {p: [0] * 64 for p in chess.PIECE_TYPES}
  eg = {p: [0] * 64 for p in chess.PIECE_TYPES}
  pawn_mg_rank = [0, 0, 2, 6, 14, 30, 50, 0]
  pawn_eg_rank = [0, 0, 5, 15, 30, 55, 90, 0]
  king_home_file = [10, 20, 5, -5, -5, 5, 20, 10]
  for sq in chess.SQUARES:
    f, r, c = chess.square_file(sq), chess.square_rank(sq), _centrality(sq)
    centre_file = f in (3, 4)

    # Pawns: worth more as they advance; central pawns on ranks 3-4 control the centre.
    mg[chess.PAWN][sq] = pawn_mg_rank[r] + (10 if centre_file and r in (3, 4) else 0)
    eg[chess.PAWN][sq] = pawn_eg_rank[r]

    # Knights and bishops: strong in the centre, knights badly placed on the rim.
    mg[chess.KNIGHT][sq] = round(-30 + 8 * c)
    eg[chess.KNIGHT][sq] = round(-25 + 6 * c)
    mg[chess.BISHOP][sq] = round(-10 + 4 * c)
    eg[chess.BISHOP][sq] = round(-8 + 3 * c)

    # Rooks: the 7th rank and central files in the middlegame.
    mg[chess.ROOK][sq] = (15 if r == 6 else 0) + (5 if centre_file else 0)
    eg[chess.ROOK][sq] = 5 if r == 6 else 0

    # Queen: mild centralisation; it should not rush out, which material loss already punishes.
    mg[chess.QUEEN][sq] = round(-5 + 2 * c)
    eg[chess.QUEEN][sq] = round(-10 + 3 * c)

    # King: stay home and castled in the middlegame, walk to the centre in the endgame.
    mg[chess.KING][sq] = king_home_file[f] if r == 0 else -25 * min(r, 3)
    eg[chess.KING][sq] = round(-18 + 6 * c)
  return mg, eg


MG_TABLE, EG_TABLE = _build_tables()


def evaluate_white(board):
  """Static evaluation in centipawns from White's point of view."""
  mg = eg = 0
  phase = 0
  bishops = [0, 0]
  for sq, piece in board.piece_map().items():
    p, white = piece.piece_type, piece.color
    idx = sq if white else chess.square_mirror(sq)
    value = PIECE_VALUE[p]
    sign = 1 if white else -1
    mg += sign * (value + MG_TABLE[p][idx])
    eg += sign * (value + EG_TABLE[p][idx])
    phase += PHASE_WEIGHT[p]
    if p == chess.BISHOP:
      bishops[white] += 1
  phase = min(phase, MAX_PHASE)
  score = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
  if bishops[chess.WHITE] >= 2:
    score += BISHOP_PAIR
  if bishops[chess.BLACK] >= 2:
    score -= BISHOP_PAIR
  return score


def evaluate(board):
  """Static evaluation in centipawns from the side to move's point of view (negamax convention)."""
  score = evaluate_white(board)
  return score if board.turn == chess.WHITE else -score
