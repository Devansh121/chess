#!/bin/bash
# Run a match split across parallel shards, each starting at a different opening, then total it.
# usage: tools/run_match.sh NAME ENGINE_A ENGINE_B GAMES SHARDS [extra match.py args...]
# Logs and PGNs go to runs/NAME/. GAMES should be a multiple of 2 * SHARDS.
set -eu
cd "$(dirname "$0")/.."
name=$1 a=$2 b=$3 games=$4 shards=$5
shift 5
per=$((games / shards))
per=$((per - per % 2))
out=runs/$name
mkdir -p "$out"
pids=()
PYTHON=${PYTHON:-.venv/bin/python}
for ((k = 0; k < shards; k++)); do
  "$PYTHON" tools/match.py "$a" "$b" "$per" --start-game $((k * per)) --pgn "$out/shard$k.pgn" "$@" \
    > "$out/shard$k.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
"$PYTHON" tools/summarize.py "$out"/shard*.log | tee "$out/summary.txt"
