"""Evaluate positions with a trained value network, in NumPy (no torch needed at runtime).

Weights come from scripts/export_net.py. The network outputs a logit for the side to move's win
probability; it is converted to centipawns with the inverse of the training target
p = 1 / (1 + 10^(-cp/400)), i.e. cp = logit * 400 / ln(10).
"""

import math

import chess
import numpy as np

from nn.features import NUM_FEATURES, feature_index

CP_PER_LOGIT = 400 / math.log(10)
CP_LIMIT = 5000  # keep network scores well clear of mate scores


class NetEvaluator:
  def __init__(self, path):
    d = np.load(path)
    self.w1 = d["w1"].astype(np.float32)  # [NUM_FEATURES + 1, H1]; last row is padding (zeros)
    self.b1 = d["b1"].astype(np.float32)
    self.w2 = d["w2"].astype(np.float32)  # [H2, H1]
    self.b2 = d["b2"].astype(np.float32)
    self.w3 = d["w3"].astype(np.float32)  # [H2]
    self.b3 = float(d["b3"])
    if self.w1.shape[0] != NUM_FEATURES + 1:
      raise ValueError(f"{path}: expected {NUM_FEATURES + 1} input rows, got {self.w1.shape[0]}")

  def features(self, board):
    stm = board.turn
    return [
      feature_index(piece.color == stm, piece.piece_type - 1, sq if stm == chess.WHITE else sq ^ 56)
      for sq, piece in board.piece_map().items()
    ]

  def logit(self, board):
    h1 = np.clip(self.w1[self.features(board)].sum(axis=0) + self.b1, 0.0, 1.0)
    h2 = np.clip(self.w2 @ h1 + self.b2, 0.0, 1.0)
    return float(self.w3 @ h2) + self.b3

  def __call__(self, board):
    """Centipawns for the side to move."""
    cp = int(self.logit(board) * CP_PER_LOGIT)
    return max(-CP_LIMIT, min(CP_LIMIT, cp))
