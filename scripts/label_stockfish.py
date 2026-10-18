#!/usr/bin/env python3
"""Score positions with full-strength Stockfish, writing the same TSV format as fetch_evals.py.

Input: lines starting with a FEN (e.g. from `chess-engine datagen`, extra tab-separated columns ignored).
Output: fen <TAB> cp <TAB> mate <TAB> depth, scores from White's point of view, so build_dataset.py can
convert it like the Lichess data. Positions are scored in parallel, one Stockfish process per worker.

usage: scripts/label_stockfish.py IN.tsv OUT.tsv [--depth D] [--workers N]
"""

import argparse
import os
import sys
from multiprocessing import Pool

import chess
import chess.engine

STOCKFISH = os.environ.get("STOCKFISH", os.path.expanduser("~/.local/share/stockfish/stockfish-17.1"))

_engine = None
_depth = 0


def _init(depth):
  global _engine, _depth
  _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
  _engine.configure({"Threads": 1, "Hash": 16})
  _depth = depth


def _label(fen):
  board = chess.Board(fen)
  info = _engine.analyse(board, chess.engine.Limit(depth=_depth))
  score = info["score"].white()
  depth = info.get("depth", _depth)
  if score.is_mate():
    return f"{fen}\t\t{score.mate()}\t{depth}\n"
  return f"{fen}\t{score.score()}\t\t{depth}\n"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("input")
  ap.add_argument("output")
  ap.add_argument("--depth", type=int, default=10)
  ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
  args = ap.parse_args()

  with open(args.input) as f:
    fens = list(dict.fromkeys(line.split("\t", 1)[0].strip() for line in f if line.strip()))
  print(f"{len(fens):,} unique positions, depth {args.depth}, {args.workers} workers", file=sys.stderr, flush=True)
  done = 0
  with Pool(args.workers, initializer=_init, initargs=(args.depth,)) as pool, open(args.output, "w") as out:
    for line in pool.imap(_label, fens, chunksize=64):
      out.write(line)
      done += 1
      if done % 50_000 == 0:
        print(f"{done:,} labelled", file=sys.stderr, flush=True)
  print(f"done: {done:,} positions -> {args.output}", file=sys.stderr, flush=True)


if __name__ == "__main__":
  main()
