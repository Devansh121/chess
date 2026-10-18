# chess-rs: Rust port

A port of the C++ engine in `cpp/` to compare languages on speed and, later, playing strength. Same
design: 16-bit moves, bitboards, magic numbers shared with `cpp/src/magics.inc`, the same Zobrist keys.

```
cargo test --release            # perft counts, hash consistency, FEN round trips, magic tables
cargo run --release --bin perft # benchmark on the six shared perft positions
```

## Speed vs C++

Perft over the six shared positions (16,046,250 nodes), same machine (Ryzen 7 7435HS), three rounds
each, millions of nodes per second:

| Build | Typical | Range |
|---|---|---|
| C++, `-march=native` | ~52 | 51.8 - 53.5 |
| C++, portable | ~52 | 51.8 |
| Rust, `target-cpu=native` (default here, see `.cargo/config.toml`) | ~43 (-17%) | 39.5 - 43.6 |
| Rust, generic x86-64 | ~36 (-31%) | 33.5 - 37.6 |

- The CPU target matters much more for Rust than for C++: generic x86-64 Rust lacks hardware `popcnt`.
- Two micro-optimizations made no measurable difference and were not kept: leaving the move list
  uninitialised instead of zeroing it, and unchecked indexing into the magic attack table.
- The remaining gap is likely bounds checks on square-indexed table lookups; to be profiled once the
  port has a search and evaluation.

## Search: Rust vs C++

With search and evaluation ported (step 2), mirroring the C++ engine's first search (#13):

- `bench 7`: **10,959,291 nodes in both engines**, so they search exactly the same tree; Rust ~10.1M
  nodes/s vs C++ ~9.9M. The move-generation gap above disappears once evaluation and search dominate.
- 80 games at 0.1 s/move, Rust vs C++: +28 =20 -32 (47.5%), -17 ± 67 Elo, equal within the margin.

## Search with pruning: Rust vs C++

Step 3 ports the C++ pruning set from #15-#22 (PVS, null-move pruning, late-move reductions, history,
check extension, aspiration windows, reverse futility and futility pruning). Bench node counts match the
C++ engine at #22 (`cpp-futility`) exactly: depth 7 819,565 and depth 9 2,422,516 in both. Speed on
bench 9: Rust ~5.4M nodes/s, C++ ~5.1M.
