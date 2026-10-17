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
