# Status and plan (paused 2026-10-23)

Goal: 3200 Elo against Stockfish 17.1 with `UCI_LimitStrength`, measured as in `docs/progress.md`:
0.1 s per move, 40 openings from both sides, single thread.

## Where things stand

**Rating: ~3007 ± 26** with #71: +79 =168 -73 over 320 games against Stockfish 3000. That is about 193 short
of 3200.

| Step | Rating vs Stockfish | Gain vs previous (800 games, 100k nodes) |
|---|---|---|
| #62 (start of this run) | ~2795 | |
| #67: PGO, king buckets, LMP/SEE, null move/razoring | ~2870 | +26, +58, +116, +37 |
| #68: SPSA-tuned search margins | ~2869 | +13 ± 15 |
| #70: net trained with own self-play data | ~2924 | +69 ± 22 |
| #71: second self-play round | **~3007** | +38 ± 22 |

### Open PRs (stacked, none merged)

#63 PGO → #64 king buckets → #65 LMP/SEE → #66 LMR tweaks (neutral) → #67 null move/razoring → #68 SPSA tuning
→ #69 resumable training → #70 self-play net → #71 second self-play net.

Branches without a PR, because they did not gain:
- `nnue-output-buckets`: arch 4, 8 output heads by piece count. Fully implemented and tested in the trainer,
  exporter and engine, but +0 ± 21.
- `search-capthist` (+3), `search-singular` (+1), `search-corrhist` (-15).

## The plateau after #71

Seven experiments, all within noise of #71's net (details in `docs/experiments.md`, round 8):

| Experiment | vs #71 |
|---|---|
| Round 3 self-play (16.7M positions) | +3 ± 21 |
| Half the Lichess data (~50% self-play) | -5 ± 21 |
| 512 wide | +10 ± 22 at equal nodes, ~15% slower |
| Rounds 1-4 of self-play | +9 ± 22 |
| SPSA retune for #71's net | +2 ± 21 |
| Output buckets (arch 4) | +0 ± 21 |
| Result weight 0.5 instead of 0.3 | -10 ± 22 |

The self-play validation sets still pick the best epoch within a run, but no longer predict gains between
runs.

## Plan, in order

1. **Deeper self-play labels.** This was in progress when paused.
   - All self-play so far was scored by 5k-node searches.
   - `data/selfplay8/` holds 5.4M positions at 20k nodes per move from #71's engine, compressed `.tsv.zst`.
   - To finish it, get to ~9M or more and build:
     `for f in data/selfplay8/sp*.tsv.zst; do zstd -dc $f; echo; done | .venv/bin/python scripts/build_dataset.py --input /dev/stdin --out data/processed-sp8 --wdl 0.3`
   - Then train #71's recipe with `--data data/processed-sp8:N` added. Keep every epoch and match the best
     one against #71's net.
2. **If deeper labels help, iterate rounds with them:** generate with the newest engine, retrain, then match.
   This loop gave +69 and +38 before it stalled.
3. **Combine the small positives** once a stronger base exists: 512 wide, output buckets, rounds 1-4.
   Each was +0 to +10 alone and might add up.
4. **Engine speed.** Ratings are at fixed time, so nodes/s is Elo.
   - About 1.7-1.9M nps on the PGO build.
   - `perf` is blocked (`kernel.perf_event_paranoid`); profiling needs that changed (sudo) or another
     profiler.
5. **Search ideas not yet retried on the new nets:** continuation history, singular extensions with a
   different margin, history-based pruning.
6. **Measurement:** at ~3000, also play Stockfish 3190 (its maximum `UCI_Elo`). At 50% against 3000, the
   3000 anchor alone says little.

## How to run things

- **Wrapper:** every job goes through `~/.local/bin/chess-run`, which runs in a systemd user scope pinned to
  threads 0-7, `nice 10`, with an 8 GB memory cap. Keep the total under ~8 threads; 4 cores and 6 GB stay free
  for other work.
- **Logout:** logging out kills all chess jobs. `loginctl enable-linger` would prevent that; it has not been
  enabled.
- **Disk:** the disk is shared and keeps filling up. `runs/selfplay*-finish.sh` watchers stop datagen at
  30M positions or 3 GB free, then build shards.
- **Matches:**
  - `runs/launch-match.sh NAME A B GAMES SHARDS` (100k nodes, `--random-plies 4`, `SEED` env).
  - `tools/summarize.py runs/NAME/*.log` for the result.
- **Rating:** `tools/run_match.sh NAME uci:ENGINE sf3000 320 4 --movetime 0.1 --our-time 0.1`
  (PGO build via `tools/build_pgo.sh`).
- **Training:** `scripts/train.py NAME --arch v3 --hidden1 384 --data DIR[:N] ... --val-extra DIR --epochs 10 --resume`.
  - `NAME.pt` keeps the Lichess-val best, which is usually not the best for play.
  - Save every epoch from `NAME.ckpt`; the best self-play-val epoch has been 9 in most runs.
- **Exporting:** `scripts/export_net.py nets/X.pt runs/bin/X.nnue` clamps first-layer outliers to the int16
  bound.
- **#71's recipe:** `--data data/processed --data data/processed2 --data data/processed3 --data data/processed-sp4:3 --data data/processed-sp5:4`.

## Data on disk

| Path | Contents |
|---|---|
| `data/processed{,2,3}` | ~297M Lichess positions |
| `data/processed-sp4..7` | Self-play rounds 1-4: 14.5M, 22.3M, 16.7M and 23.4M positions at 5k nodes, result weight 0.3 |
| `data/processed-sp4w5`, `-sp5w5` | Rounds 1-2 with result weight 0.5. Can be deleted (-10). |
| `data/selfplay4..8` | Raw datagen output, compressed |
| `runs/bin/` | Pinned engine binaries and exported nets. `chess-cpp-8ae407e-pgo` is #71's engine. |
