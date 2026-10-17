#!/usr/bin/env python3
"""Total the FINAL lines of match logs (one per shard) and print the combined Elo estimate.

usage: tools/summarize.py runs/NAME/*.log
"""

import re
import sys

from match import elo_diff, elo_margin

FINAL = re.compile(r"^FINAL (\S+) vs (\S+): \+(\d+) =(\d+) -(\d+)")


def main():
  wins = draws = losses = 0
  names = None
  for path in sys.argv[1:]:
    with open(path) as f:
      for line in f:
        m = FINAL.match(line)
        if m:
          names = names or (m[1], m[2])
          wins, draws, losses = wins + int(m[3]), draws + int(m[4]), losses + int(m[5])
  games = wins + draws + losses
  if not games:
    sys.exit("no finished shards")
  score = wins + 0.5 * draws
  bound, diff = elo_diff(score, games)
  print(
    f"{names[0]} vs {names[1]}: +{wins} ={draws} -{losses}  score {score}/{games} ({100 * score / games:.1f}%)  "
    f"Elo {bound}{diff:+.0f} +/- {elo_margin(wins, draws, losses):.0f}"
  )


if __name__ == "__main__":
  main()
