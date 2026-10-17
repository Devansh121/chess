#!/usr/bin/env python3
"""Turn Lichess eval-database JSONL (stdin) into compact TSV rows (stdout).

Each output row: fen <TAB> cp <TAB> mate <TAB> depth, taken from the deepest
evaluation's first principal variation. Exactly one of cp / mate is set.

usage:
  curl -s https://database.lichess.org/lichess_db_eval.jsonl.zst \
    | zstd -dc | python3 scripts/fetch_evals.py 50000000 | zstd -q -o data/evals.tsv.zst
"""

import json
import sys


def main():
  limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
  out = sys.stdout
  n = 0
  for line in sys.stdin:
    rec = json.loads(line)
    best = max(rec["evals"], key=lambda e: e["depth"])
    pv = best["pvs"][0]
    cp = pv.get("cp", "")
    mate = pv.get("mate", "")
    out.write(f"{rec['fen']}\t{cp}\t{mate}\t{best['depth']}\n")
    n += 1
    if n % 1_000_000 == 0:
      print(f"{n:,} positions", file=sys.stderr, flush=True)
    if limit is not None and n >= limit:
      break


if __name__ == "__main__":
  try:
    main()
  except BrokenPipeError:
    pass
