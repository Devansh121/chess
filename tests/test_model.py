import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nn.features import MAX_PIECES, NUM_FEATURES, PAD, fen_features, pack  # noqa: E402
from nn.model import ValueNet, to_model_input  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")


def test_padding_maps_to_the_zero_row():
  x = to_model_input(np.array([pack([1, 2, 3])]))
  assert x.dtype == torch.int64
  assert x[0, :3].tolist() == [1, 2, 3]
  assert (x[0, 3:] == NUM_FEATURES).all()


def test_padding_contributes_nothing():
  model = ValueNet()
  a = to_model_input(np.array([pack([5, 70, 700])]))
  b = torch.tensor([[5, 70, 700]])
  assert torch.allclose(model(a), model(b))


def test_output_shape_and_param_count():
  model = ValueNet()
  x = to_model_input(np.stack([pack(fen_features("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"))] * 4))
  assert model(x).shape == (4,)
  assert 200_000 < sum(p.numel() for p in model.parameters()) < 210_000


def test_training_lowers_loss_on_a_tiny_dataset(tmp_path):
  # Synthetic task: the target depends on whether feature 0 is present, so a working loop must learn it.
  rng = np.random.default_rng(0)
  for name, n in (("train_000", 4000), ("val_000", 500)):
    x = np.full((n, MAX_PIECES), PAD, np.uint16)
    x[:, :8] = rng.integers(1, NUM_FEATURES, (n, 8))
    has = rng.random(n) < 0.5
    x[has, 0] = 0
    y = np.where(has, 0.9, 0.1).astype(np.float16)
    np.savez(tmp_path / f"{name}.npz", x=x, y=y, cp=np.zeros(n, np.int16))
  out = tmp_path / "nets"
  subprocess.run(
    [
      sys.executable,
      "scripts/train.py",
      "tiny",
      "--data",
      str(tmp_path),
      "--out",
      str(out),
      "--epochs",
      "8",
      "--batch-size",
      "256",
      "--lr",
      "3e-3",
    ],
    check=True,
    cwd=REPO,
    capture_output=True,
  )
  log = json.loads((out / "tiny.json").read_text())
  assert (out / "tiny.pt").exists()
  assert log["epochs"][-1]["val_mse"] < log["epochs"][0]["val_mse"]
  assert log["best_val_mse"] < 0.02  # 0.16 if it learned nothing
