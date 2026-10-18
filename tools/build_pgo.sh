#!/usr/bin/env bash
# Profile-guided build of the engine: an instrumented build, a training workload, then an optimised
# rebuild from the recorded profile. The search is unchanged (bench node counts match a normal build).
#
#   tools/build_pgo.sh [BUILD_DIR]        default cpp/build-pgo; compiler from $CXX (g++ or clang++)
#
# The training workload is short self-play games, not the bench positions used to measure speed.
set -euo pipefail
cd "$(dirname "$0")/.."
build=${1:-cpp/build-pgo}
rm -rf "$build"
profiles=$(realpath -m "$build/pgo")
mkdir -p "$profiles"

cmake -S cpp -B "$build" -DPGO=GEN -DPGO_DIR="$profiles" > /dev/null
cmake --build "$build" -j "$(nproc)" --target chess-engine > /dev/null
echo "training profile (self-play games)..."
"$build/chess-engine" datagen 30 20000 7 /dev/null 8 > /dev/null 2>&1

if compgen -G "$profiles/*.profraw" > /dev/null; then  # Clang writes raw profiles that must be merged
  profdata=${LLVM_PROFDATA:-$(compgen -c llvm-profdata | sort -V | tail -1)}
  "$profdata" merge -o "$profiles/merged.profdata" "$profiles"/*.profraw
fi

cmake -S cpp -B "$build" -DPGO=USE > /dev/null
cmake --build "$build" -j "$(nproc)" --target chess-engine --clean-first > /dev/null
echo "built $build/chess-engine"
"$build/chess-engine" bench | tail -1
