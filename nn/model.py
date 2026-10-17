"""Value network: 768 sparse inputs -> 256 -> 32 -> 1 (win probability for the side to move).

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


def to_model_input(x_uint16):
  """Dataset arrays (uint16, padded with 0xFFFF) -> int64 tensor padded with NUM_FEATURES."""
  x = torch.as_tensor(x_uint16.astype("int64"))
  return torch.where(x == PAD, torch.full_like(x, NUM_FEATURES), x)
