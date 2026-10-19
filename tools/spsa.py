#!/usr/bin/env python3
"""Tune the engine's search margins (its tune:: UCI options) with SPSA self-play.

Each iteration perturbs every parameter by +/-c_k at random, plays game pairs between the two perturbed
engines (same random opening, colours swapped), and moves every parameter towards the side that scored
better. Gains follow the fishtest schedule: c_k = c / k^0.101, a_k = a / (A + k)^0.602, set from the
per-parameter end values c_end (step size at the end) and r_end (learning rate at the end).

usage: tools/spsa.py ENGINE OUT.json [--iterations N] [--workers W] [--nodes N] [--resume]

The state is written to OUT.json after every iteration; --resume continues from it.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import load_openings, play_game, randomized_opening  # noqa: E402

R_END = 0.002


def engine_params(path):
  """The engine's tunable spin options: {name: (default, min, max)} (Hash and Threads excluded)."""
  out = subprocess.run([path], input="uci\nquit\n", capture_output=True, text=True, check=True).stdout
  params = {}
  for line in out.splitlines():
    tok = line.split()
    if tok[:2] == ["option", "name"] and "spin" in tok and tok[2] not in ("Hash", "Threads"):
      params[tok[2]] = tuple(int(tok[tok.index(k) + 1]) for k in ("default", "min", "max"))
  return params


class Player:
  def __init__(self, path, nodes, evalfile):
    self.engine = chess.engine.SimpleEngine.popen_uci(path)
    if evalfile:
      self.engine.configure({"EvalFile": evalfile})
    self.limit = chess.engine.Limit(nodes=nodes)
    self.game = None

  def configure(self, values, game):
    self.engine.configure({k: int(round(v)) for k, v in values.items()})
    self.game = game  # a new object makes python-chess send ucinewgame

  def move(self, board, limit=None):
    return self.engine.play(board, self.limit, game=self.game).move


_players = None


def _init(path, nodes, evalfile):
  global _players
  _players = (Player(path, nodes, evalfile), Player(path, nodes, evalfile))


def _play_pair(task):
  """Score of the + engine over a game pair (0..2)."""
  plus, minus, opening = task
  a, b = _players
  score = 0.0
  for white_is_plus in (True, False):
    game = object()
    a.configure(plus, game)
    b.configure(minus, game)
    white, black = (a, b) if white_is_plus else (b, a)
    result, _ = play_game(white, black, opening)
    points = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
    score += points if white_is_plus else 1 - points
  return score


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("engine")
  ap.add_argument("out")
  ap.add_argument("--iterations", type=int, default=2000)
  ap.add_argument("--workers", type=int, default=6)
  ap.add_argument("--nodes", type=int, default=25000)
  ap.add_argument("--evalfile", default="")
  ap.add_argument("--openings", default=os.path.join(os.path.dirname(__file__), "openings.txt"))
  ap.add_argument("--random-plies", type=int, default=4)
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--resume", action="store_true")
  args = ap.parse_args()

  path = os.path.abspath(args.engine)
  spec = engine_params(path)
  n = args.iterations
  big_a = 0.1 * n
  c_end = {k: max((hi - lo) / 20, 1.0) for k, (_, lo, hi) in spec.items()}
  if args.resume:
    with open(args.out) as f:
      state = json.load(f)
  else:
    state = {"iteration": 0, "theta": {k: float(d) for k, (d, _, _) in spec.items()}, "history": []}

  openings = load_openings(args.openings)
  rng = random.Random(args.seed + state["iteration"])
  evalfile = os.path.abspath(args.evalfile) if args.evalfile else ""
  with mp.Pool(args.workers, initializer=_init, initargs=(path, args.nodes, evalfile)) as pool:
    for k in range(state["iteration"] + 1, n + 1):
      theta = state["theta"]
      delta = {p: rng.choice((-1, 1)) for p in theta}
      ck = {p: c_end[p] * n**0.101 / k**0.101 for p in theta}
      plus, minus = {}, {}
      for p, (_, lo, hi) in spec.items():
        plus[p] = min(max(theta[p] + ck[p] * delta[p], lo), hi)
        minus[p] = min(max(theta[p] - ck[p] * delta[p], lo), hi)
      tasks = []
      for _ in range(args.workers):
        opening = rng.choice(openings)
        tasks.append((plus, minus, randomized_opening(opening, args.random_plies, rng.randrange(1 << 30))))
      scores = pool.map(_play_pair, tasks)
      result = sum(2 * (s - 1) for s in scores)  # + engine's wins minus losses, in games
      for p, (_, lo, hi) in spec.items():
        a = R_END * c_end[p] ** 2 * (big_a + n) ** 0.602
        ak = a / (big_a + k) ** 0.602
        theta[p] = min(max(theta[p] + ak / ck[p] * result * delta[p], lo), hi)
      state["iteration"] = k
      state["history"].append(result)
      with open(args.out + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
      os.replace(args.out + ".tmp", args.out)
      if k % 10 == 0:
        recent = state["history"][-100:]
        print(
          f"iter {k}/{n} recent {sum(recent):+.1f} " + " ".join(f"{p}={v:.1f}" for p, v in theta.items()),
          flush=True,
        )


if __name__ == "__main__":
  main()
