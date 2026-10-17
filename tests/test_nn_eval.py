import subprocess
import sys

import chess
import numpy as np
import pytest

from engine.nn_eval import CP_LIMIT, NetEvaluator
from engine.search import Searcher
from nn.features import NUM_FEATURES, fen_features

REPO = __file__.rsplit("/tests/", 1)[0]
FENS = [
  chess.STARTING_FEN,
  "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8",
  "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - - 0 1",
]


def random_net(path, seed=0, h1=256, h2=32):
  rng = np.random.default_rng(seed)
  w1 = rng.normal(0, 0.05, (NUM_FEATURES + 1, h1)).astype(np.float32)
  w1[NUM_FEATURES] = 0
  np.savez(
    path,
    w1=w1,
    b1=rng.normal(0, 0.1, h1).astype(np.float32),
    w2=rng.normal(0, 0.1, (h2, h1)).astype(np.float32),
    b2=rng.normal(0, 0.1, h2).astype(np.float32),
    w3=rng.normal(0, 0.5, h2).astype(np.float32),
    b3=np.float32(0.1),
  )
  return path


@pytest.mark.parametrize("fen", FENS)
def test_board_features_match_dataset_features(fen, tmp_path):
  net = NetEvaluator(random_net(tmp_path / "n.npz"))
  assert sorted(net.features(chess.Board(fen))) == sorted(fen_features(fen))


@pytest.mark.parametrize("fen", FENS)
def test_colour_flipped_position_scores_the_same(fen, tmp_path):
  net = NetEvaluator(random_net(tmp_path / "n.npz"))
  board = chess.Board(fen)
  assert net(board) == net(board.mirror())


def test_scores_are_bounded(tmp_path):
  net = NetEvaluator(random_net(tmp_path / "n.npz"))
  net.b3 = 1e6
  assert net(chess.Board()) == CP_LIMIT


def test_search_runs_with_the_network(tmp_path):
  net = NetEvaluator(random_net(tmp_path / "n.npz"))
  board = chess.Board()
  move, _, depth = Searcher(evaluate=net).search(board, max_depth=2)
  assert move in board.legal_moves and depth == 2


def test_matches_the_torch_model(tmp_path):
  torch = pytest.importorskip("torch")
  from nn.features import pack
  from nn.model import ValueNet, to_model_input

  model = ValueNet()
  torch.save(model.state_dict(), tmp_path / "n.pt")
  subprocess.run([sys.executable, "scripts/export_net.py", str(tmp_path / "n.pt")], check=True, cwd=REPO)
  net = NetEvaluator(tmp_path / "n.npz")
  for fen in FENS:
    expected = model(to_model_input(np.array([pack(fen_features(fen))]))).item()
    assert net.logit(chess.Board(fen)) == pytest.approx(expected, abs=1e-5)


def test_uci_evalfile_option(tmp_path):
  path = random_net(tmp_path / "n.npz")
  script = f"uci\nsetoption name EvalFile value {path}\nisready\nposition startpos\ngo depth 2\n"
  proc = subprocess.Popen(
    [sys.executable, "-m", "engine.uci"], cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
  )
  proc.stdin.write(script)
  proc.stdin.flush()
  lines = []
  for line in proc.stdout:
    lines.append(line.strip())
    if line.startswith("bestmove"):
      break
  proc.stdin.write("quit\n")
  proc.stdin.close()
  proc.wait(timeout=5)
  assert "option name EvalFile type string default <empty>" in lines
  assert "readyok" in lines
  assert chess.Move.from_uci(lines[-1].split()[1]) in chess.Board().legal_moves


def test_cp_scale_from_the_weights_file(tmp_path):
  import math

  path = random_net(tmp_path / "n.npz")
  default = NetEvaluator(path)
  assert default.cp_per_logit == pytest.approx(400 / math.log(10))
  d = dict(np.load(path))
  np.savez(tmp_path / "wide.npz", **d, cp_scale=1000.0)
  wide = NetEvaluator(tmp_path / "wide.npz")
  assert wide.cp_per_logit == pytest.approx(1000 / math.log(10))
  board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8")
  assert wide.logit(board) == default.logit(board)


@pytest.mark.parametrize("arch", ["v1", "v2"])
def test_cpp_engine_matches_torch(tmp_path, arch):
  """The C++ engine's network evaluation (via its 'eval' command) equals the PyTorch model's output."""
  import json
  import os

  torch = pytest.importorskip("torch")
  binary = os.path.join(REPO, "cpp", "build", "chess-engine")
  if not os.path.exists(binary):
    pytest.skip("C++ engine not built (cmake -S cpp -B cpp/build && cmake --build cpp/build)")
  from nn.features import pack
  from nn.model import ARCHS, to_model_input

  torch.manual_seed(1)
  model = ARCHS[arch]()
  torch.save(model.state_dict(), tmp_path / "n.pt")
  (tmp_path / "n.json").write_text(json.dumps({"args": {"arch": arch, "cp_scale": 400.0}}))
  subprocess.run(
    [sys.executable, "scripts/export_net.py", str(tmp_path / "n.pt"), str(tmp_path / "n.nnue")], check=True, cwd=REPO
  )
  script = f"setoption name EvalFile value {tmp_path / 'n.nnue'}\n"
  for fen in FENS:
    script += f"position fen {fen}\neval\n"
  out = subprocess.run([binary], input=script + "quit\n", capture_output=True, text=True, timeout=30).stdout
  rows = [line.split() for line in out.splitlines() if line.startswith("eval ")]
  assert len(rows) == len(FENS), out
  with torch.no_grad():
    for fen, row in zip(FENS, rows, strict=True):
      quantized, reference = float(row[row.index("logit") + 1]), float(row[row.index("float") + 1])
      expected = model(to_model_input(np.array([pack(fen_features(fen))]))).item()
      assert reference == pytest.approx(expected, abs=1e-4), fen  # float path: same maths as PyTorch
      assert quantized == pytest.approx(expected, abs=0.05), fen  # int16 path: rounding only
