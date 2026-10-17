"""UCI front end: `python -m engine.uci`. Lets chess GUIs, Stockfish match tools and tools/match.py drive the engine."""

import sys
import threading

import chess

from engine.search import Searcher, mate_in

NAME = "chess"
AUTHOR = "Devansh Wassista"


def time_for_move(board, args):
  """Seconds to think, from the `go` arguments. Defaults to 1 second."""
  if "movetime" in args:
    return args["movetime"] / 1000
  side = "w" if board.turn == chess.WHITE else "b"
  if f"{side}time" in args:
    remaining = args[f"{side}time"] / 1000
    inc = args.get(f"{side}inc", 0) / 1000
    moves_to_go = args.get("movestogo", 30)
    return max(0.05, min(remaining / moves_to_go + 0.8 * inc, remaining / 3))
  if "depth" in args or "infinite" in args:
    return None
  return 1.0


def parse_position(tokens):
  board = chess.Board()
  if tokens[0] == "fen":
    end = tokens.index("moves") if "moves" in tokens else len(tokens)
    board = chess.Board(" ".join(tokens[1:end]))
    tokens = tokens[end:]
  elif tokens[0] == "startpos":
    tokens = tokens[1:]
  if tokens and tokens[0] == "moves":
    for uci in tokens[1:]:
      board.push_uci(uci)
  return board


def parse_go(tokens):
  args = {}
  i = 0
  while i < len(tokens):
    key = tokens[i]
    if key == "infinite":
      args["infinite"] = True
      i += 1
    elif i + 1 < len(tokens):
      args[key] = int(tokens[i + 1])
      i += 2
    else:
      i += 1
  return args


def send(line):
  sys.stdout.write(line + "\n")
  sys.stdout.flush()


def format_score(score):
  mate = mate_in(score)
  return f"mate {mate}" if mate is not None else f"cp {score}"


def main():
  board = chess.Board()
  searcher = Searcher()
  thread = None

  def go(board, args):
    def info(depth, score, nodes, secs, pv):
      nps = int(nodes / secs) if secs > 0 else 0
      send(
        f"info depth {depth} score {format_score(score)} nodes {nodes} nps {nps} "
        f"time {int(secs * 1000)} pv {' '.join(m.uci() for m in pv)}"
      )

    move, _, _ = searcher.search(
      board, time_limit=time_for_move(board, args), max_depth=args.get("depth", 64), info=info
    )
    send(f"bestmove {move.uci() if move else '0000'}")

  for line in sys.stdin:
    tokens = line.split()
    if not tokens:
      continue
    cmd = tokens[0]
    if cmd == "uci":
      send(f"id name {NAME}")
      send(f"id author {AUTHOR}")
      send("uciok")
    elif cmd == "isready":
      send("readyok")
    elif cmd == "ucinewgame":
      searcher = Searcher()
    elif cmd == "position":
      board = parse_position(tokens[1:])
    elif cmd == "go":
      thread = threading.Thread(target=go, args=(board.copy(), parse_go(tokens[1:])), daemon=True)
      thread.start()
    elif cmd == "stop":
      searcher.stop = True
      if thread:
        thread.join()
    elif cmd == "quit":
      searcher.stop = True
      break


if __name__ == "__main__":
  main()
