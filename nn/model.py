"""Value networks: sparse piece-square inputs -> win probability for the side to move.

v1: 768 -> 256 -> 32 -> 1, one accumulator from the side to move's view.
v2: the same 768 -> 256 first layer applied from both sides' views (shared weights), the two
    accumulators concatenated (side to move first) -> 512 -> 32 -> 1. Both accumulators can be
    updated incrementally as moves are made, whichever side is to move, which is what the C++
    engine needs; the second view also gives the network the opponent's perspective.

The first layer is an embedding sum over the active features (at most 32 per position), which is what
makes NNUE-style networks cheap: evaluating a position touches only the rows of its pieces.
"""

import torch
from torch import nn

from nn.features import NUM_FEATURES, PAD

HIDDEN1 = 256
HIDDEN2 = 32


class ValueNet(nn.Module):
  def __init__(self, hidden1=HIDDEN1, hidden2=HIDDEN2):
    super().__init__()
    # Row NUM_FEATURES is the padding row: unused slots in a position's index array point there and add nothing.
    self.inputs = nn.EmbeddingBag(NUM_FEATURES + 1, hidden1, mode="sum", padding_idx=NUM_FEATURES)
    self.input_bias = nn.Parameter(torch.zeros(hidden1))
    self.l2 = nn.Linear(hidden1, hidden2)
    self.out = nn.Linear(hidden2, 1)
    nn.init.normal_(self.inputs.weight, std=0.05)
    with torch.no_grad():
      self.inputs.weight[NUM_FEATURES].zero_()

  def forward(self, x):
    """`x`: int64 [batch, 32] feature indices, padded with NUM_FEATURES. Returns the logit [batch]."""
    h = torch.clamp(self.inputs(x) + self.input_bias, 0.0, 1.0)  # clipped ReLU, as in NNUE
    h = torch.clamp(self.l2(h), 0.0, 1.0)
    return self.out(h).squeeze(-1)


def other_perspective(x):
  """Feature indices of the same position seen by the other side: own/enemy swapped, ranks mirrored.

  Index layout (nn/features.py): ((enemy * 6 + piece_type) * 64 + square). Padding stays padding.
  """
  side, rest = x // 384, x % 384
  other = ((1 - side) * 6 + rest // 64) * 64 + ((rest % 64) ^ 56)
  return torch.where(x == NUM_FEATURES, x, other)


class ValueNetV2(nn.Module):
  def __init__(self, hidden1=HIDDEN1, hidden2=HIDDEN2):
    super().__init__()
    self.inputs = nn.EmbeddingBag(NUM_FEATURES + 1, hidden1, mode="sum", padding_idx=NUM_FEATURES)
    self.input_bias = nn.Parameter(torch.zeros(hidden1))
    self.l2 = nn.Linear(2 * hidden1, hidden2)
    self.out = nn.Linear(hidden2, 1)
    nn.init.normal_(self.inputs.weight, std=0.05)
    with torch.no_grad():
      self.inputs.weight[NUM_FEATURES].zero_()

  def forward(self, x):
    us = torch.clamp(self.inputs(x) + self.input_bias, 0.0, 1.0)
    them = torch.clamp(self.inputs(other_perspective(x)) + self.input_bias, 0.0, 1.0)
    h = torch.clamp(self.l2(torch.cat([us, them], dim=1)), 0.0, 1.0)
    return self.out(h).squeeze(-1)


ARCHS = {"v1": ValueNet, "v2": ValueNetV2}


def to_model_input(x_uint16):
  """Dataset arrays (uint16, padded with 0xFFFF) -> int64 tensor padded with NUM_FEATURES."""
  x = torch.as_tensor(x_uint16.astype("int64"))
  return torch.where(x == PAD, torch.full_like(x, NUM_FEATURES), x)
