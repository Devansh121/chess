#!/usr/bin/env python3
"""Total the FINAL lines of match logs (one per shard) and print the combined Elo estimate.

usage: tools/summarize.py runs/NAME/*.log [--fail-below ELO]

--fail-below: exit 1 if the whole ~95% interval lies below ELO (a confident regression).
"""

import re
import sys

from match import elo_diff, elo_margin

FINAL = re.compile(r"^FINAL (\S+) vs (\S+): \+(\d+) =(\d+) -(\d+)")


def main():
  wins = draws = losses = 0
  names = None
  args = sys.argv[1:]
  fail_below = None
  if "--fail-below" in args:
    i = args.index("--fail-below")
    fail_below = float(args[i + 1])
    del args[i : i + 2]
  for path in args:
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
  margin = elo_margin(wins, draws, losses)
  print(
    f"{names[0]} vs {names[1]}: +{wins} ={draws} -{losses}  score {score}/{games} ({100 * score / games:.1f}%)  "
    f"Elo {bound}{diff:+.0f} +/- {margin:.0f}"
  )
  if fail_below is not None and diff + margin < fail_below:
    print(f"REGRESSION: upper bound {diff + margin:+.0f} is below {fail_below:+.0f}")
    sys.exit(1)


if __name__ == "__main__":
  main()
