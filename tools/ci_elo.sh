#!/bin/bash
# Elo test of the C++ engine: this checkout vs a base git ref, node-limited so the result doesn't depend
# on machine speed or load. Prints a markdown report, also writes it to $ELO_REPORT (if set) and appends
# it to $GITHUB_STEP_SUMMARY (if set).
# usage: tools/ci_elo.sh BASE_REF [GAMES] [SHARDS] [NODES]
set -euo pipefail
cd "$(dirname "$0")/.."
base_ref=$1 games=${2:-1000} shards=${3:-4} nodes=${4:-20000}
report=$(mktemp)
publish() {
  cat "$report"
  if [ -n "${ELO_REPORT:-}" ]; then cp "$report" "$ELO_REPORT"; fi
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then cat "$report" >> "$GITHUB_STEP_SUMMARY"; fi
}

if ! git cat-file -e "$base_ref:cpp/src/main.cpp" 2>/dev/null; then
  echo "Base \`$base_ref\` has no C++ engine yet; nothing to compare." > "$report"
  publish
  exit 0
fi

work=$(mktemp -d)
git worktree add -q --detach "$work/base" "$base_ref"
trap 'git worktree remove --force "$work/base"; rm -rf "$work"' EXIT
for side in head base; do
  src=$([ $side = head ] && echo cpp || echo "$work/base/cpp")
  cmake -S "$src" -B "$work/build-$side" -DCMAKE_BUILD_TYPE=Release -DNATIVE=ON > /dev/null  # same runner builds and runs both: use its AVX2
  cmake --build "$work/build-$side" -j "$shards" --target chess-engine > /dev/null
  cp "$work/build-$side/chess-engine" "$work/engine-$side"
done

bench_head=$("$work/engine-head" bench 6)
bench_base=$("$work/engine-base" bench 6)
name=ci-elo-$(date +%s)
set +e
tools/run_match.sh "$name" "uci:$work/engine-head" "uci:$work/engine-base" "$games" "$shards" \
  --nodes "$nodes" --random-plies 4 --seed 1 > /dev/null &
match_pid=$!
# Live progress in the job log: games played and the running score (head's wins/draws/losses).
while kill -0 "$match_pid" 2> /dev/null; do
  sleep 30
  tail -qn1 runs/"$name"/shard*.log 2> /dev/null | grep -o '+[0-9]* =[0-9]* -[0-9]*' |
    awk '{w += substr($1, 2); d += substr($2, 2); l += substr($3, 2)}
         END {if (w + d + l) printf "progress: %d/'"$games"' games  head +%d =%d -%d\n", w + d + l, w, d, l}'
done
wait "$match_pid"
PYTHON=${PYTHON:-.venv/bin/python}
summary=$("$PYTHON" tools/summarize.py runs/"$name"/shard*.log --fail-below -10)
status=$?
set -e

{
  echo "## Elo: this PR vs \`$base_ref\`"
  echo
  echo "$games games, $nodes nodes per move, 40 openings + 4 seeded random plies, both colours."
  echo
  echo '```'
  echo "$summary" | sed "s#uci:$work/engine-##g"
  echo "bench head: $bench_head"
  echo "bench base: $bench_base"
  echo '```'
  echo
  echo "Fails only on a confident regression (the whole ~95% interval below -10 Elo)."
} > "$report"
publish
exit $status
