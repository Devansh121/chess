import math
import os
import sys

import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import match  # noqa: E402


def test_elo_diff_even_score_is_zero():
  assert match.elo_diff(5, 10) == ("", 0.0)


def test_elo_diff_matches_logistic_formula():
  bound, diff = match.elo_diff(7.5, 10)
  assert bound == ""
  assert math.isclose(diff, -400 * math.log10(1 / 0.75 - 1))


def test_elo_diff_reports_bounds_at_extremes():
  assert match.elo_diff(0, 10)[0] == "<"
  assert match.elo_diff(10, 10)[0] == ">"


def test_random_game_ends_legally():
  result, board = match.play_game(match.Random(), match.Random())
  assert result in ("1-0", "0-1", "1/2-1/2")
  assert len(board.move_stack) <= 400
  replay = chess.Board()
  for move in board.move_stack:
    assert move in replay.legal_moves
    replay.push(move)
