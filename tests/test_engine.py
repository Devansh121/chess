import subprocess
import sys

import chess
import pytest

from engine.eval import evaluate, evaluate_white
from engine.search import MATE_BOUND, Searcher, mate_in

REPO = __file__.rsplit("/tests/", 1)[0]


def best(fen, depth):
  move, score, _ = Searcher().search(chess.Board(fen), max_depth=depth)
  return move, score


# --- evaluation ---


def test_start_position_is_balanced():
  assert evaluate_white(chess.Board()) == 0


def test_eval_is_symmetric_for_mirrored_positions():
  board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
  assert evaluate_white(board) == -evaluate_white(board.mirror())


def test_eval_is_from_side_to_move():
  board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
  assert evaluate(board) > 800
  board.turn = chess.BLACK
  assert evaluate(board) < -800


# --- search ---


@pytest.mark.parametrize(
  "fen, mate",
  [
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),  # back-rank mate
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),  # scholar's mate
  ],
)
def test_finds_mate_in_one(fen, mate):
  move, score = best(fen, 2)
  assert move.uci() == mate
  assert mate_in(score) == 1


def test_finds_mate_in_two():
  # 1. Ra7 (cuts off the 7th rank) Kg8 2. Rb8#
  board = chess.Board("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1")
  move, score, _ = Searcher().search(board, max_depth=4)
  assert score > MATE_BOUND
  assert mate_in(score) == 2


def test_takes_free_queen():
  move, _ = best("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 2)
  assert move.uci() == "d1d5"


def test_quiescence_sees_the_recapture():
  # Qxd5 grabs a pawn but cxd5 wins the queen; a depth-1 search without quiescence would play it.
  move, score = best("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", 1)
  assert move.uci() != "d1d5"
  assert score > 600


def test_does_not_leave_queen_hanging_to_a_pawn():
  board = chess.Board("4k3/8/8/4p3/3Q4/8/8/4K3 w - - 0 1")
  move, _, _ = Searcher().search(board, max_depth=3)
  board.push(move)
  queen = board.pieces(chess.QUEEN, chess.WHITE).pop()
  assert not any(m.to_square == queen for m in board.legal_moves if board.is_capture(m))


def test_stalemate_scores_zero():
  board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
  assert board.is_stalemate()
  assert Searcher().search(board, max_depth=2)[0] is None


def test_time_limit_returns_a_legal_move():
  board = chess.Board()
  move, _, depth = Searcher().search(board, time_limit=0.2)
  assert move in board.legal_moves
  assert depth >= 1


def test_mate_in_helper():
  assert mate_in(100) is None
  assert mate_in(100_000 - 1) == 1
  assert mate_in(100_000 - 3) == 2
  assert mate_in(-(100_000 - 2)) == -1


# --- UCI ---


def test_uci_session():
  script = "uci\nisready\nposition startpos moves e2e4\ngo depth 2\n"
  proc = subprocess.Popen(
    [sys.executable, "-m", "engine.uci"],
    cwd=REPO,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
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
  assert "uciok" in lines and "readyok" in lines
  board = chess.Board()
  board.push_uci("e2e4")
  assert chess.Move.from_uci(lines[-1].split()[1]) in board.legal_moves


# --- transposition table ---

# King-and-pawn endgame: few moves per position and many transpositions (king routes), so depth 5 is fast.
ENDGAME = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"


def test_tt_searches_fewer_nodes():
  without, with_tt = Searcher(use_tt=False), Searcher()
  without.search(chess.Board(ENDGAME), max_depth=5)
  with_tt.search(chess.Board(ENDGAME), max_depth=5)
  assert with_tt.nodes < without.nodes


def test_tt_is_reused_by_the_next_search():
  s = Searcher()
  s.search(chess.Board(ENDGAME), max_depth=5)
  first = s.nodes
  s.search(chess.Board(ENDGAME), max_depth=5)
  assert s.nodes < first / 2


def test_tt_mate_scores_round_trip():
  from engine.search import MATE, _from_tt, _to_tt

  for score in (MATE - 5, -(MATE - 7), 123, -45):
    assert _from_tt(_to_tt(score, 3), 3) == score
  # a mate found 3 plies from the root is stored as 'mate from this node' and re-read correctly 5 plies deep
  assert _from_tt(_to_tt(MATE - 10, 3), 5) == MATE - 12


def test_tt_finds_the_same_mate_in_two():
  board = chess.Board("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1")
  s = Searcher()
  s.search(board, max_depth=4)
  _, score, _ = s.search(board, max_depth=4)  # second search answers mostly from the table
  assert mate_in(score) == 2


# --- move ordering ---


def test_ordering_hash_move_then_captures_then_killers():
  # White can capture the d5 queen (Rxd5) or the b5 pawn (Bxb5); quiet moves otherwise.
  board = chess.Board("4k3/8/8/1p1q4/8/8/8/3RKB2 w - - 0 1")
  s = Searcher()
  hash_move = chess.Move.from_uci("e1f2")
  killer = chess.Move.from_uci("f1e2")
  s.killers[3] = [killer, None]
  moves = s._ordered(board, hash_move, 3)
  assert moves[0] == hash_move
  assert moves[1] == chess.Move.from_uci("d1d5")  # takes the queen: most valuable victim
  assert moves[2] == chess.Move.from_uci("f1b5")  # takes a pawn
  assert moves[3] == killer
  assert sorted(moves, key=str) == sorted(board.legal_moves, key=str)


def test_quiet_cutoff_move_becomes_a_killer():
  s = Searcher()
  s.search(chess.Board(), max_depth=4)
  assert any(k is not None and not chess.Board().is_capture(k) for pair in s.killers for k in pair)


def test_ordering_cuts_nodes_without_changing_the_answer():
  fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8"
  ordered = Searcher()
  move, score, _ = ordered.search(chess.Board(fen), max_depth=3)
  unordered = Searcher()
  unordered._ordered = lambda board, hash_move, ply: list(board.legal_moves)
  move2, score2, _ = unordered.search(chess.Board(fen), max_depth=3)
  # Same minimax value; the chosen move can differ when several moves tie on that value.
  assert score == score2
  board = chess.Board(fen)
  assert move in board.legal_moves and move2 in board.legal_moves
  assert ordered.nodes < unordered.nodes / 2
