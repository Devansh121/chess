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


@pytest.mark.parametrize("arch", ["v1", "v2"])
def test_training_lowers_loss_on_a_tiny_dataset(tmp_path, arch):
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
      "--arch",
      arch,
    ],
    check=True,
    cwd=REPO,
    capture_output=True,
  )
  log = json.loads((out / "tiny.json").read_text())
  assert (out / "tiny.pt").exists()
  assert log["epochs"][-1]["val_mse"] < log["epochs"][0]["val_mse"]
  assert log["best_val_mse"] < 0.02  # 0.16 if it learned nothing


@pytest.mark.parametrize(
  "fen",
  [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8",
    "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - - 0 1",
  ],
)
def test_other_perspective_is_the_other_sides_view(fen):
  from nn.model import other_perspective

  fields = fen.split()
  flipped = " ".join([fields[0], "b" if fields[1] == "w" else "w"] + fields[2:])
  x = to_model_input(np.array([pack(fen_features(fen))]))
  other = other_perspective(x)[0]
  got = sorted(int(i) for i in other if i != NUM_FEATURES)
  assert got == sorted(fen_features(flipped))


def test_v2_shapes_and_padding():
  from nn.model import ValueNetV2

  model = ValueNetV2()
  a = to_model_input(np.array([pack([5, 70, 700])]))
  b = torch.tensor([[5, 70, 700]])
  assert torch.allclose(model(a), model(b))
  assert model(a).shape == (1,)
  # 769 x 256 shared input weights + 256 biases + 512 x 32 + 32 + 32 + 1 = 213,569
  assert sum(p.numel() for p in model.parameters()) == 213_569


def test_data_repeat_and_init(tmp_path):
  """DIR:N repeats a folder's shards; --init starts from given weights."""
  rng = np.random.default_rng(1)
  for d, n in (("a", 600), ("b", 300)):
    (tmp_path / d).mkdir()
    for name, size in (("train_000", n), ("val_000", 100)):
      x = np.full((size, MAX_PIECES), PAD, np.uint16)
      x[:, :8] = rng.integers(0, NUM_FEATURES, (size, 8))
      np.savez(tmp_path / d / f"{name}.npz", x=x, y=np.full(size, 0.5, np.float16), cp=np.zeros(size, np.int16))
  out = tmp_path / "nets"
  args = ["--out", str(out), "--epochs", "1", "--batch-size", "128", "--arch", "v2"]
  subprocess.run(
    [sys.executable, "scripts/train.py", "base", "--data", str(tmp_path / "a")] + args,
    check=True,
    cwd=REPO,
    capture_output=True,
  )
  subprocess.run(
    [
      sys.executable,
      "scripts/train.py",
      "mix",
      "--data",
      str(tmp_path / "a"),
      "--data",
      f"{tmp_path / 'b'}:3",
      "--init",
      str(out / "base.pt"),
    ]
    + args,
    check=True,
    cwd=REPO,
    capture_output=True,
  )
  log = json.loads((out / "mix.json").read_text())
  assert log["epochs"][0]["positions"] == 600 + 3 * 300
