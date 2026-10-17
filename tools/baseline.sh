#!/bin/bash
# Baseline matches for twitchchess, run in parallel shards. Results in runs/baseline/*.log
set -u
cd "$(dirname "$0")/.."
: "${TWITCHCHESS:?set TWITCHCHESS to a geohot/twitchchess checkout}"
OUT=runs/baseline; mkdir -p $OUT
run() { .venv/bin/python tools/match.py "$1" "$2" "$3" --pgn "$OUT/$4.pgn" > "$OUT/$4.log" 2>&1; }
run tc-nn1 sf1320 40 nn1_sf1320 &
run tc-nn1 random 40 nn1_random &
for i in 1 2 3 4 5 6; do run tc-classic sf1320 2 classic_sf1320_$i & done
for i in 1 2; do run tc-classic tc-nn1 2 classic_nn1_$i & done
wait
grep -h FINAL $OUT/*.log
