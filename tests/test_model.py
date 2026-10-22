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


V3_FENS = [  # with the side to move's expected bucket (nn/model.py KING_BUCKETS)
  ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1),  # Ke1, mirrored to d1
  ("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b - - 0 1", 0),  # Kg8: g1 for Black, mirrored b1
  ("2kr3r/ppp2ppp/8/8/8/8/PPP2PPP/1K1R3R w - - 0 1", 0),  # Kb1, not mirrored
  ("8/8/3k4/8/8/4K3/8/8 w - - 0 1", 3),  # Ke3 -> d3
  ("8/5k2/8/8/8/8/1K6/8 b - - 0 1", 2),  # Kf7: f2 for Black, mirrored c2
]


def test_v3_buckets_follow_the_own_king():
  from nn.model import NUM_BUCKETS, bucketed

  x = to_model_input(np.array([pack(fen_features(f)) for f, _ in V3_FENS]))
  full, _ = bucketed(x)
  for row, (fen, want) in zip(full, V3_FENS, strict=True):
    active = row[row < NUM_BUCKETS * NUM_FEATURES]
    assert set((active // NUM_FEATURES).tolist()) == {want}, fen


ENGINE = os.path.join(REPO, "cpp", "build", "chess-engine")


@pytest.mark.skipif(not os.path.exists(ENGINE), reason="needs the C++ engine (cmake build in cpp/build)")
def test_v3_export_matches_the_cpp_engine(tmp_path):
  # A random v3 network exported to .nnue must give the C++ engine's float reference the same logit as torch:
  # this pins the bucket map, mirroring and factoriser folding on both sides.
  from nn.model import ValueNetV3

  torch.manual_seed(0)
  model = ValueNetV3(hidden1=32)
  with torch.no_grad():
    model.inputs.weight.normal_(std=0.02)
    model.inputs.weight[-1].zero_()
  torch.save(model.state_dict(), tmp_path / "v3.pt")
  (tmp_path / "v3.json").write_text(json.dumps({"args": {"arch": "v3", "hidden1": 32, "cp_scale": 400.0}}))
  net = tmp_path / "v3.nnue"
  subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_net.py"), tmp_path / "v3.pt", net], check=True)

  commands = [f"setoption name EvalFile value {net}"]
  for fen, _ in V3_FENS:
    commands += [f"position fen {fen}", "eval"]
  out = subprocess.run([ENGINE], input="\n".join(commands + ["quit"]), capture_output=True, text=True, check=True)
  cpp = [float(line.split(" float ")[1]) for line in out.stdout.splitlines() if " float " in line]
  x = to_model_input(np.array([pack(fen_features(f)) for f, _ in V3_FENS]))
  with torch.no_grad():
    want = model(x).tolist()
  assert len(cpp) == len(want)
  assert np.allclose(cpp, want, atol=1e-4), (cpp, want)


V4_FENS = [fen for fen, _ in V3_FENS] + [
  "8/8/4k3/8/8/3K4/8/8 w - - 0 1",  # 2 pieces: output bucket 0
  "8/5pk1/8/8/2R5/8/5PK1/8 b - - 0 1",  # 5 pieces
  "r4rk1/pp3ppp/8/8/8/8/PP3PPP/R4RK1 w - - 0 1",  # 16 pieces
]


def test_v4_output_buckets_follow_piece_count():
  from nn.model import output_bucket

  x = to_model_input(np.array([pack(fen_features(f)) for f in V4_FENS]))
  pieces = [len(fen_features(f)) for f in V4_FENS]
  assert output_bucket(x).tolist() == [min((p - 2) * 8 // 32, 7) for p in pieces]
  assert output_bucket(x)[len(V3_FENS)].item() == 0


@pytest.mark.skipif(not os.path.exists(ENGINE), reason="C++ engine not built")
def test_v4_export_matches_the_cpp_engine(tmp_path):
  # Same as the v3 check, with output buckets: positions with different piece counts use different heads.
  from nn.model import ValueNetV4

  torch.manual_seed(0)
  model = ValueNetV4(hidden1=32)
  with torch.no_grad():
    model.inputs.weight.normal_(std=0.02)
    model.inputs.weight[-1].zero_()
  torch.save(model.state_dict(), tmp_path / "v4.pt")
  (tmp_path / "v4.json").write_text(json.dumps({"args": {"arch": "v4", "hidden1": 32, "cp_scale": 400.0}}))
  net = tmp_path / "v4.nnue"
  subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_net.py"), tmp_path / "v4.pt", net], check=True)

  commands = [f"setoption name EvalFile value {net}"]
  for fen in V4_FENS:
    commands += [f"position fen {fen}", "eval"]
  out = subprocess.run([ENGINE], input="\n".join(commands + ["quit"]), capture_output=True, text=True, check=True)
  cpp = [float(line.split(" float ")[1]) for line in out.stdout.splitlines() if " float " in line]
  x = to_model_input(np.array([pack(fen_features(f)) for f in V4_FENS]))
  with torch.no_grad():
    want = model(x).tolist()
  assert len(cpp) == len(want)
  assert np.allclose(cpp, want, atol=1e-4), (cpp, want)
