#!/usr/bin/env python3
"""Play two engines against each other and estimate the Elo difference.

usage: tools/match.py ENGINE_A ENGINE_B GAMES [--pgn OUT.pgn] [--movetime SECONDS] [--start-game N]

Games start from the positions in tools/openings.txt: game i plays opening i // 2, and
consecutive games swap colours, so each opening is played once from each side. To split a
match across parallel processes, give each shard a different even --start-game.

engines:
  random       uniformly random legal move
  sf<ELO>      Stockfish with UCI_LimitStrength at that Elo (1320..3190)
  tc-nn1       twitchchess Part 1: value net, 1 ply, no search
  tc-classic   twitchchess master: material + mobility, minimax
  ours         this repo's engine over UCI (python -m engine.uci)
  ours:PATH    the engine in another checkout, e.g. a git worktree of an older commit
  oursnn:FILE  this repo's engine with the value network in FILE (.npz from scripts/export_net.py)
  uci:PATH     any UCI engine binary (e.g. cpp/build/chess-engine), given --our-time per move,
               or --nodes per move (reproducible, independent of machine speed);
               UCI options after '@': uci:PATH@EvalFile=nets/x.nnue,Threads=4

env:
  STOCKFISH    path to the Stockfish binary (default ~/.local/share/stockfish/stockfish-17.1)
  TWITCHCHESS  path to a geohot/twitchchess checkout (needed for tc-* engines)
"""

import argparse
import contextlib
import io
import math
import os
import random
import sys
import time

import chess
import chess.engine
import chess.pgn
import numpy as np

STOCKFISH = os.environ.get("STOCKFISH", os.path.expanduser("~/.local/share/stockfish/stockfish-17.1"))
TWITCHCHESS = os.environ.get("TWITCHCHESS", "")
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class Random:
  def move(self, board, limit=None):
    return random.choice(list(board.legal_moves))

  def close(self):
    pass


class Stockfish:
  def __init__(self, elo, movetime):
    self.limit = chess.engine.Limit(time=movetime)
    self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1, "Hash": 16})

  def move(self, board, limit=None):
    return self.engine.play(board, limit or self.limit).move

  def close(self):
    self.engine.quit()


class UciEngine:
  """Any UCI engine binary, thinking a fixed time (or a fixed number of nodes) per move."""

  def __init__(self, spec, movetime, nodes=0):
    path, _, opts = spec.partition("@")
    self.limit = chess.engine.Limit(nodes=nodes) if nodes else chess.engine.Limit(time=movetime)
    self.engine = chess.engine.SimpleEngine.popen_uci(os.path.abspath(path))
    options = dict(kv.split("=", 1) for kv in opts.split(",") if kv)
    if "EvalFile" in options:
      options["EvalFile"] = os.path.abspath(options["EvalFile"])
    if options:
      self.engine.configure(options)

  def move(self, board, limit=None):
    return self.engine.play(board, limit or self.limit).move

  def close(self):
    self.engine.quit()


class Ours:
  """Our engine as a UCI subprocess, so two versions (different checkouts) can play each other."""

  def __init__(self, path, movetime, options=None):
    self.limit = chess.engine.Limit(time=movetime)
    self.engine = chess.engine.SimpleEngine.popen_uci([sys.executable, "-m", "engine.uci"], cwd=path)
    if options:
      self.engine.configure(options)

  def move(self, board, limit=None):
    return self.engine.play(board, limit or self.limit).move

  def close(self):
    self.engine.quit()


@contextlib.contextmanager
def in_twitchchess():
  """twitchchess loads nets/value.pth by relative path, so imports and evals run from its checkout."""
  if not TWITCHCHESS:
    sys.exit("set TWITCHCHESS to a geohot/twitchchess checkout")
  prev = os.getcwd()
  os.chdir(TWITCHCHESS)
  try:
    yield
  finally:
    os.chdir(prev)


class TwitchchessNet:
  """Part 1 engine: score every legal move with the value net, pick the best for the side to move."""

  def __init__(self):
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, TWITCHCHESS)
    with in_twitchchess():
      from state import State
      from train import Net

      self.model = Net()
      self.model.load_state_dict(torch.load("nets/value.pth", map_location="cpu"))
    self.model.eval()
    self.torch, self.State = torch, State

  def move(self, board, limit=None):
    moves = list(board.legal_moves)
    kids = []
    for m in moves:
      board.push(m)
      kids.append(self.State(board.copy()).serialize())
      board.pop()
    with self.torch.no_grad():
      vals = self.model(self.torch.from_numpy(np.stack(kids)).float())[:, 0].tolist()
    pick = max if board.turn == chess.WHITE else min
    return moves[pick(range(len(moves)), key=lambda i: vals[i])]

  def close(self):
    pass


class TwitchchessClassic:
  """master engine: play.py's ClassicValuator + explore_leaves, exactly as its web UI calls them."""

  def __init__(self):
    sys.path.insert(0, TWITCHCHESS)
    with in_twitchchess(), contextlib.redirect_stdout(io.StringIO()):
      import play
      from state import State
    self.play, self.State = play, State
    self.v = play.ClassicValuator()

  def move(self, board, limit=None):
    with contextlib.redirect_stdout(io.StringIO()):
      ret = self.play.explore_leaves(self.State(board.copy()), self.v)
    return sorted(ret, key=lambda x: x[0], reverse=board.turn)[0][1]

  def close(self):
    pass


def make(name, movetime, our_time, nodes=0):
  if name == "ours" or name.startswith("ours:"):
    return Ours(os.path.abspath(name[5:] or REPO), our_time)
  if name.startswith("uci:"):
    return UciEngine(name[4:], our_time, nodes)
  if name.startswith("oursnn:"):
    return Ours(REPO, our_time, {"EvalFile": os.path.abspath(name[7:])})
  if name == "random":
    return Random()
  if name.startswith("sf"):
    return Stockfish(int(name[2:]), movetime)
  if name == "tc-nn1":
    return TwitchchessNet()
  if name == "tc-classic":
    return TwitchchessClassic()
  sys.exit(f"unknown engine {name}")


OPENINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openings.txt")


def load_openings(path):
  """Openings as lists of moves; [[]] (the start position) if the file is missing."""
  if not path or not os.path.exists(path):
    return [[]]
  with open(path) as f:
    lines = [line.split() for line in f if line.strip() and not line.startswith("#")]
  return [[chess.Move.from_uci(m) for m in line] for line in lines] or [[]]


def randomized_opening(opening, plies, seed):
  """The opening plus `plies` random legal moves chosen by `seed` (same seed -> same moves)."""
  rng = random.Random(seed)
  board = chess.Board()
  for move in opening:
    board.push(move)
  moves = list(opening)
  for _ in range(plies):
    legal = list(board.legal_moves)
    if not legal or board.is_game_over():
      break
    move = rng.choice(legal)
    board.push(move)
    moves.append(move)
  return moves


def parse_tc(tc):
  """'10+0.1' -> (10.0, 0.1) seconds; None or '' -> None."""
  if not tc:
    return None
  base, _, inc = tc.partition("+")
  return float(base), float(inc or 0)


def play_game(white, black, opening=(), max_plies=400, tc=None):
  """Plays one game. With `tc` = (base, increment) seconds both sides play on a clock that the engines see
  in their limits; running out of time loses (a draw if the opponent has only a king)."""
  board = chess.Board()
  for move in opening:
    board.push(move)
  clock = {chess.WHITE: tc[0], chess.BLACK: tc[0]} if tc else None
  while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
    player = white if board.turn == chess.WHITE else black
    if clock is None:
      board.push(player.move(board))
      continue
    limit = chess.engine.Limit(
      white_clock=clock[chess.WHITE], black_clock=clock[chess.BLACK], white_inc=tc[1], black_inc=tc[1]
    )
    side = board.turn
    start = time.monotonic()
    move = player.move(board, limit)
    clock[side] -= time.monotonic() - start
    if clock[side] < 0:
      only_king = board.occupied_co[not side] == board.kings & board.occupied_co[not side]
      return ("1/2-1/2" if only_king else ("0-1" if side == chess.WHITE else "1-0")), board
    clock[side] += tc[1]
    board.push(move)
  result = board.result(claim_draw=True)
  return ("1/2-1/2" if result == "*" else result), board


def elo_diff(score, games):
  """Elo difference implied by a score; at 0% or 100% report a bound using half a game."""
  s = min(max(score, 0.5), games - 0.5) / games
  bound = "<" if score == 0 else ">" if score == games else ""
  return bound, -400 * math.log10(1 / s - 1)


def elo_margin(wins, draws, losses):
  """Half-width of the ~95% interval on the Elo difference (normal approximation on the mean score)."""
  n = wins + draws + losses
  if n < 2:
    return float("inf")
  s = (wins + 0.5 * draws) / n
  var = (wins * (1 - s) ** 2 + draws * (0.5 - s) ** 2 + losses * s**2) / n
  se = math.sqrt(var / n)

  def elo(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / p - 1)

  return (elo(s + 1.96 * se) - elo(s - 1.96 * se)) / 2


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("a")
  ap.add_argument("b")
  ap.add_argument("games", type=int)
  ap.add_argument("--pgn")
  ap.add_argument("--movetime", type=float, default=0.1, help="Stockfish seconds per move")
  ap.add_argument("--our-time", type=float, default=0.5, help="our engine's seconds per move")
  ap.add_argument("--openings", default=OPENINGS, help="opening file; '' plays every game from the start position")
  ap.add_argument("--start-game", type=int, default=0, help="index of the first game, for sharding")
  ap.add_argument("--nodes", type=int, default=0, help="nodes per move for uci:PATH engines (instead of time)")
  ap.add_argument(
    "--random-plies", type=int, default=0, help="random moves after each opening; both games of a pair share them"
  )
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--tc", default="", help="clock time control BASE+INC in seconds, e.g. 10+0.1 (overrides times)")
  args = ap.parse_args()
  openings = load_openings(args.openings)
  tc = parse_tc(args.tc)
  pgn = os.path.abspath(args.pgn) if args.pgn else None

  a = make(args.a, args.movetime, args.our_time, args.nodes)
  b = make(args.b, args.movetime, args.our_time, args.nodes)
  wins = draws = losses = 0
  start = time.time()
  for n in range(args.games):
    i = args.start_game + n
    a_white = i % 2 == 0
    opening = openings[(i // 2) % len(openings)]
    if args.random_plies:
      opening = randomized_opening(opening, args.random_plies, f"{args.seed}-{i // 2}")
    result, board = play_game(a, b, opening, tc=tc) if a_white else play_game(b, a, opening, tc=tc)
    a_score = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
    if not a_white:
      a_score = 1.0 - a_score
    wins += a_score == 1.0
    draws += a_score == 0.5
    losses += a_score == 0.0
    if pgn:
      game = chess.pgn.Game.from_board(board)
      game.headers.update(
        {"White": args.a if a_white else args.b, "Black": args.b if a_white else args.a, "Result": result}
      )
      with open(pgn, "a") as f:
        print(game, file=f, end="\n\n")
    print(
      f"game {n + 1}/{args.games}: {args.a} {'WDL'[[1.0, 0.5, 0.0].index(a_score)]} "
      f"({len(board.move_stack)} plies)  +{wins} ={draws} -{losses}  {time.time() - start:.0f}s",
      flush=True,
    )

  score = wins + 0.5 * draws
  bound, diff = elo_diff(score, args.games)
  margin = elo_margin(wins, draws, losses)
  print(
    f"FINAL {args.a} vs {args.b}: +{wins} ={draws} -{losses}  score {score}/{args.games}  "
    f"Elo diff {bound}{diff:+.0f} +/- {margin:.0f}"
  )
  a.close()
  b.close()


if __name__ == "__main__":
  main()
