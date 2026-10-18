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


# v3: v2 with king buckets. Each perspective's features are mirrored left-right when its own king is on
# files e-h, then picked from one of NUM_BUCKETS copies of the 768 rows by that king's square (the
# mirrored view only has files a-d). A factoriser row set shared by every bucket is trained alongside and
# folded into each bucket at export, so rarely seen buckets still start from the common piece values.
KING_BUCKETS = [  # by own-king square in the perspective's mirrored view, a1 = 0; files e-h never occur
  0, 0, 1, 1, 0, 0, 0, 0,
  2, 2, 2, 2, 0, 0, 0, 0,
] + [3, 3, 3, 3, 0, 0, 0, 0] * 6  # fmt: skip
NUM_BUCKETS = max(KING_BUCKETS) + 1
OWN_KING = 5  # (own side 0 * 6 + king) in the feature layout


def bucketed(x, num_buckets=NUM_BUCKETS):
  """768-space indices (padded with NUM_FEATURES) -> (bucketed index, mirrored 768-space index), both padded.

  Every position has exactly one own king, which picks the bucket and whether files are mirrored.
  """
  pad = x == NUM_FEATURES
  ksq = torch.where(x // 64 == OWN_KING, x % 64, torch.zeros_like(x)).sum(dim=1, keepdim=True)
  flip = torch.where(ksq % 8 >= 4, 7, 0)
  plain = (x // 64) * 64 + ((x % 64) ^ flip)
  bucket = torch.as_tensor(KING_BUCKETS, device=x.device)[ksq ^ flip]
  full = bucket * NUM_FEATURES + plain
  return torch.where(pad, num_buckets * NUM_FEATURES, full), torch.where(pad, x, plain)


class ValueNetV3(nn.Module):
  def __init__(self, hidden1=HIDDEN1, hidden2=HIDDEN2, num_buckets=NUM_BUCKETS):
    super().__init__()
    self.num_buckets = num_buckets
    rows = num_buckets * NUM_FEATURES
    self.inputs = nn.EmbeddingBag(rows + 1, hidden1, mode="sum", padding_idx=rows)
    self.factor = nn.EmbeddingBag(NUM_FEATURES + 1, hidden1, mode="sum", padding_idx=NUM_FEATURES)
    self.input_bias = nn.Parameter(torch.zeros(hidden1))
    self.l2 = nn.Linear(2 * hidden1, hidden2)
    self.out = nn.Linear(hidden2, 1)
    nn.init.zeros_(self.inputs.weight)  # buckets start as the shared rows and learn the differences
    nn.init.normal_(self.factor.weight, std=0.05)
    with torch.no_grad():
      self.factor.weight[NUM_FEATURES].zero_()

  def accumulate(self, x):
    full, plain = bucketed(x, self.num_buckets)
    return torch.clamp(self.inputs(full) + self.factor(plain) + self.input_bias, 0.0, 1.0)

  def forward(self, x):
    h = torch.cat([self.accumulate(x), self.accumulate(other_perspective(x))], dim=1)
    return self.out(torch.clamp(self.l2(h), 0.0, 1.0)).squeeze(-1)

  def folded_input_weights(self):
    """First-layer rows with the factoriser added to every bucket: [num_buckets * 768 + 1, hidden1]."""
    w = self.inputs.weight.detach().clone()
    shared = self.factor.weight.detach()[:NUM_FEATURES]
    w[: self.num_buckets * NUM_FEATURES] += shared.repeat(self.num_buckets, 1)
    return w


ARCHS = {"v1": ValueNet, "v2": ValueNetV2, "v3": ValueNetV3}


def to_model_input(x_uint16):
  """Dataset arrays (uint16, padded with 0xFFFF) -> int64 tensor padded with NUM_FEATURES."""
  x = torch.as_tensor(x_uint16.astype("int64"))
  return torch.where(x == PAD, torch.full_like(x, NUM_FEATURES), x)
