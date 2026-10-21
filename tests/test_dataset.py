import os
import subprocess
import sys

import chess
import numpy as np
import pytest

from nn.features import PAD, feature_index, fen_features, win_probability

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import build_dataset  # noqa: E402

FENS = [
  chess.STARTING_FEN,
  "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8",
  "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - - 0 1",
  "8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - - 0 1",
]


def reference_features(fen):
  """Same encoding built with python-chess, to cross-check the fast FEN parser."""
  board = chess.Board(fen)
  stm = board.turn
  out = []
  for sq, piece in board.piece_map().items():
    square = sq if stm == chess.WHITE else chess.square_mirror(sq)
    out.append(feature_index(piece.color == stm, piece.piece_type - 1, square))
  return sorted(out)


@pytest.mark.parametrize("fen", FENS)
def test_fast_parser_matches_python_chess(fen):
  assert sorted(fen_features(fen)) == reference_features(fen)


@pytest.mark.parametrize("fen", FENS)
def test_colour_flipped_position_has_identical_features(fen):
  # The same position with colours swapped and the other side to move is the same position for the side to move.
  mirrored = chess.Board(fen).mirror().fen()
  assert sorted(fen_features(fen)) == sorted(fen_features(mirrored))


def test_start_position_own_king_on_e1():
  feats = fen_features(chess.STARTING_FEN)
  assert len(feats) == 32
  assert feature_index(True, 5, chess.E1) in feats
  assert feature_index(False, 5, chess.E1 ^ 56) in feats


def test_win_probability():
  assert win_probability(cp=0) == pytest.approx(0.5)
  assert win_probability(cp=400) == pytest.approx(10 / 11)
  assert win_probability(cp=-400) == pytest.approx(1 / 11)
  assert win_probability(cp=100_000) == win_probability(cp=2000)
  assert win_probability(mate=3) == 1.0
  assert win_probability(mate=-1) == 0.0


def test_convert_row_flips_scores_for_black():
  # Lichess scores are from White's view: +69 with Black to move is bad for the side to move.
  _, _, y, cp = build_dataset.convert_row("7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -\t69\t\t46\n")
  assert cp == -69 and y < 0.5
  _, _, y, cp = build_dataset.convert_row("6k1/6p1/8/4K3/4NN2/8/8/8 w - -\t\t15\t95\n")
  assert cp == 2000 and y == 1.0
  _, _, y, _ = build_dataset.convert_row("6k1/6p1/8/4K3/4NN2/8/8/8 b - -\t\t15\t95\n")
  assert y == 0.0


def test_end_to_end_small_file(tmp_path):
  rows = "".join(f"{fen.rsplit(' ', 2)[0]}\t{i * 10}\t\t30\n" for i, fen in enumerate(FENS * 50))
  src = tmp_path / "evals.tsv"
  src.write_text(rows)
  subprocess.run(["zstd", "-q", str(src), "-o", str(tmp_path / "evals.tsv.zst")], check=True)
  out = tmp_path / "processed"
  subprocess.run(
    [sys.executable, "scripts/build_dataset.py", "--input", str(tmp_path / "evals.tsv.zst"), "--out", str(out)],
    check=True,
    cwd=os.path.join(os.path.dirname(__file__), ".."),
    capture_output=True,
  )
  total = 0
  for name in os.listdir(out):
    d = np.load(out / name)
    assert d["x"].shape[1] == 32 and d["x"].dtype == np.uint16
    assert len(d["x"]) == len(d["y"]) == len(d["cp"])
    assert ((d["x"] < 768) | (d["x"] == PAD)).all()
    total += len(d["y"])
  assert total == 200


def test_convert_row_datagen_blends_score_and_result(monkeypatch):
  monkeypatch.setattr(build_dataset, "WDL", 0.25)
  fen = FENS[1]  # White to move
  _, _, y, cp = build_dataset.convert_row(f"{fen}\t120\t0.0\n")
  assert cp == 120
  assert y == pytest.approx(0.75 * win_probability(cp=120))
  black = FENS[2]  # Black to move: score and result flip to Black's view
  _, _, y, cp = build_dataset.convert_row(f"{black}\t120\t0.0\n")
  assert cp == -120
  assert y == pytest.approx(0.75 * win_probability(cp=-120) + 0.25)
  assert build_dataset.convert_row(f"{fen}\t120\t\n") is None  # cut-off last line


def test_convert_row_datagen_rejects_two_kings():
  spliced = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNK w - - 0 1"
  assert build_dataset.convert_row(f"{spliced}\t10\t0.5\n") is None
