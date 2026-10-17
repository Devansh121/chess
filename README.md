# chess

A small chess engine. Inspired by George Hotz's
[twitchchess](https://github.com/geohot/twitchchess) streams (2018), rebuilt from scratch.

## Baseline: twitchchess

Measured against Stockfish 17.1 at `UCI_Elo` 1320; details in [docs/baseline.md](docs/baseline.md).

| Engine | Search | Time per move | Estimated Elo |
|---|---|---|---|
| twitchchess Part 1 (value net) | 1 ply, no search | ~0.01 s | well under 1000 (0/40 vs Stockfish 1320) |
| twitchchess master (material + mobility) | 3 ply full + 2 ply beam(10), ~8.5k nodes/s | 6.6-28 s | about 1350 (6.5/12) |

## Plan (current direction, will change)

1. **Baseline harness** - play engines vs Stockfish 17.1 at fixed `UCI_Elo`, report score and Elo estimate.
2. **Data** - stream Lichess's evaluated-positions database, convert ~50M positions into compact arrays
   (piece indices + side to move), target = sigmoid(cp / 400).
3. **Network** - NNUE-style: 768 sparse inputs (12 pieces x 64 squares, side-to-move perspective)
   -> 256 -> 32 -> 1, with a held-out validation split. ~20-60 min on the RTX 4050.
4. **Search** - alpha-beta with iterative deepening, transposition table, move ordering
   (TT move, MVV-LVA, killers), quiescence search, time control.
5. **Speed (Python)** - batched evaluation, cheaper move generation, PyPy where torch isn't needed.
   Target ~1600-2000.
6. **Optional: C++ core** - bitboards, incrementally updated NNUE, null-move pruning, LMR. Target 2000-2500+.
