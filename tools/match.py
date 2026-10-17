#!/usr/bin/env python3
"""Play two engines against each other and estimate the Elo difference.

usage: tools/match.py ENGINE_A ENGINE_B GAMES [--pgn OUT.pgn] [--movetime SECONDS]

engines:
  random       uniformly random legal move
  sf<ELO>      Stockfish with UCI_LimitStrength at that Elo (1320..3190)
  tc-nn1       twitchchess Part 1: value net, 1 ply, no search
  tc-classic   twitchchess master: material + mobility, minimax

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


class Random:
  def move(self, board):
    return random.choice(list(board.legal_moves))

  def close(self):
    pass


class Stockfish:
  def __init__(self, elo, movetime):
    self.limit = chess.engine.Limit(time=movetime)
    self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1, "Hash": 16})

  def move(self, board):
    return self.engine.play(board, self.limit).move

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

  def move(self, board):
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

  def move(self, board):
    with contextlib.redirect_stdout(io.StringIO()):
      ret = self.play.explore_leaves(self.State(board.copy()), self.v)
    return sorted(ret, key=lambda x: x[0], reverse=board.turn)[0][1]

  def close(self):
    pass


def make(name, movetime):
  if name == "random":
    return Random()
  if name.startswith("sf"):
    return Stockfish(int(name[2:]), movetime)
  if name == "tc-nn1":
    return TwitchchessNet()
  if name == "tc-classic":
    return TwitchchessClassic()
  sys.exit(f"unknown engine {name}")


def play_game(white, black, max_plies=400):
  board = chess.Board()
  while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
    player = white if board.turn == chess.WHITE else black
    board.push(player.move(board))
  result = board.result(claim_draw=True)
  return ("1/2-1/2" if result == "*" else result), board


def elo_diff(score, games):
  """Elo difference implied by a score; at 0% or 100% report a bound using half a game."""
  s = min(max(score, 0.5), games - 0.5) / games
  bound = "<" if score == 0 else ">" if score == games else ""
  return bound, -400 * math.log10(1 / s - 1)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("a")
  ap.add_argument("b")
  ap.add_argument("games", type=int)
  ap.add_argument("--pgn")
  ap.add_argument("--movetime", type=float, default=0.1, help="Stockfish seconds per move")
  args = ap.parse_args()
  pgn = os.path.abspath(args.pgn) if args.pgn else None

  a, b = make(args.a, args.movetime), make(args.b, args.movetime)
  wins = draws = losses = 0
  start = time.time()
  for i in range(args.games):
    a_white = i % 2 == 0
    result, board = play_game(a, b) if a_white else play_game(b, a)
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
      f"game {i + 1}/{args.games}: {args.a} {'WDL'[[1.0, 0.5, 0.0].index(a_score)]} "
      f"({len(board.move_stack)} plies)  +{wins} ={draws} -{losses}  {time.time() - start:.0f}s",
      flush=True,
    )

  score = wins + 0.5 * draws
  bound, diff = elo_diff(score, args.games)
  print(
    f"FINAL {args.a} vs {args.b}: +{wins} ={draws} -{losses}  score {score}/{args.games}  Elo diff {bound}{diff:+.0f}"
  )
  a.close()
  b.close()


if __name__ == "__main__":
  main()
