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


def test_openings_file_is_legal_and_distinct():
  openings = match.load_openings(match.OPENINGS)
  assert len(openings) >= 40
  positions = set()
  for moves in openings:
    board = chess.Board()
    for move in moves:
      assert move in board.legal_moves
      board.push(move)
    positions.add(board.board_fen())
  assert len(positions) == len(openings)


def test_missing_openings_file_means_start_position():
  assert match.load_openings("/nonexistent") == [[]]
  assert match.load_openings("") == [[]]


def test_game_starts_from_opening():
  opening = match.load_openings(match.OPENINGS)[0]
  _, board = match.play_game(match.Random(), match.Random(), opening)
  assert board.move_stack[: len(opening)] == opening


def test_elo_margin_shrinks_with_more_games():
  assert match.elo_margin(30, 20, 30) > match.elo_margin(300, 200, 300)
  assert 50 < match.elo_margin(30, 20, 30) < 100


def test_randomized_opening_is_reproducible_and_legal():
  opening = match.load_openings(match.OPENINGS)[3]
  a = match.randomized_opening(opening, 4, "seed-7")
  b = match.randomized_opening(opening, 4, "seed-7")
  c = match.randomized_opening(opening, 4, "seed-8")
  assert a == b
  assert a[: len(opening)] == opening and len(a) == len(opening) + 4
  assert a != c
  board = chess.Board()
  for move in a:
    assert move in board.legal_moves
    board.push(move)
