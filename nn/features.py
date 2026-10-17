"""Position encoding shared by the dataset builder, training and (later) the engine.

A position becomes up to 32 active feature indices out of 768 = 2 sides x 6 piece types x 64 squares,
always seen from the side to move: its own pieces are side 0, the board is mirrored when Black is to move.
"""

import numpy as np

NUM_FEATURES = 768
MAX_PIECES = 32
PAD = 0xFFFF  # unused slots in the fixed-size index array

PIECE_TYPE = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}

# Score -> win probability for the side to move. 400 is the conventional Elo-style scale.
CP_SCALE = 400.0
CP_CLIP = 2000


def feature_index(own, piece_type, square):
  """Index of one piece.

  `own`: the piece belongs to the side to move; `piece_type`: 0..5 (p n b r q k);
  `square`: 0..63 with a1 = 0, from the side to move's view.
  """
  return ((0 if own else 1) * 6 + piece_type) * 64 + square


def fen_features(fen):
  """Active feature indices for a FEN, from the side to move's view. Reads only the board and side-to-move fields."""
  fields = fen.split()
  placement, white_to_move = fields[0], fields[1] == "w"
  out = []
  rank = 7
  file = 0
  for ch in placement:
    if ch == "/":
      rank -= 1
      file = 0
    elif ch.isdigit():
      file += int(ch)
    else:
      white_piece = ch.isupper()
      square = rank * 8 + file
      if not white_to_move:
        square ^= 56  # mirror ranks so the side to move always plays "up the board"
      out.append(feature_index(white_piece == white_to_move, PIECE_TYPE[ch.lower()], square))
      file += 1
  return out


def win_probability(cp=None, mate=None):
  """Target for the side to move. `cp`/`mate` are already from the side to move's point of view."""
  if mate is not None:
    return 1.0 if mate > 0 else 0.0
  cp = max(-CP_CLIP, min(CP_CLIP, cp))
  return 1.0 / (1.0 + 10.0 ** (-cp / CP_SCALE))


def pack(indices):
  """Fixed-size uint16 array (MAX_PIECES), padded with PAD."""
  arr = np.full(MAX_PIECES, PAD, dtype=np.uint16)
  arr[: len(indices)] = indices
  return arr
