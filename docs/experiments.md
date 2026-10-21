# Training experiments

Every training run, its settings and results. Run logs with per-epoch losses are in `nets/NAME.json`;
weights are not committed (rebuild with the command below) unless a net becomes the engine default.

Data: first 50M positions of the Lichess evaluation database (`scripts/fetch_evals.py`), converted with
`scripts/build_dataset.py`: 49.49M train in 50 shards, 500k validation. Model: `nn/model.py`,
768 -> 256 -> 32 -> 1, 205,377 parameters. Hardware: RTX 4050 laptop GPU, ~1.5M positions/s.

Rebuild a run: `scripts/train.py NAME --shards K --epochs E --decided D --cp-scale S`, then
`scripts/export_net.py nets/NAME.pt`.

## Runs

| Name | Train positions | Epochs | Decided positions | Scale | Best val MSE | Time | vs hand-written eval (80 games) |
|---|---|---|---|---|---|---|---|
| size-5m | 5M (`--shards 5`) | 10 | keep | 400 | 0.03774 | 33 s | |
| size-10m | 10M | 10 | keep | 400 | 0.03492 | 67 s | |
| size-25m | 25M | 10 | keep | 400 | 0.03166 | 160 s | |
| size-all | 49.5M | 10 | keep | 400 | 0.02995 | 309 s | +34 =6 -40, -26 ± 75 |
| decided-weight | 49.5M | 10 | weight 0.1 | 400 | 0.03242 | 520 s | not played |
| decided-drop | 49.5M | 10 | drop | 400 | 0.03661 | 277 s | not played |
| scale1000 | 49.5M | 10 | keep | 1000 | 0.01531* | 311 s | +30 =10 -40, -44 ± 73 |
| epochs20 | 49.5M | 20 | keep | 400 | 0.02918 | 616 s | +43 =3 -34, +39 ± 76 |

\* Measured against its own (wider-scale) target, so not comparable with the scale-400 runs.

Matches: stage D search with the net vs the same search with the hand-written evaluation, 40 openings
from both sides, 0.5 s/move (`tools/run_match.sh`).

## Findings

- More data keeps helping up to 49.5M (no plateau yet); at full size train and validation loss are
  nearly equal, so the net is not overfitting.
- Decided positions (|cp| >= 2000 or mate, 19% of the data) must stay in; they teach what large
  advantages look like. Dropping them pushed a queen-up position from +746 to +273.
- Scale 400 compresses large advantages (queen-up only ~1.6x a knight-up). Scale 1000 fixes that
  but did not win more games.
- No net beats the hand-written evaluation clearly yet. Network evaluation costs 71 µs vs 35 µs in
  Python, but search depth was not the problem (2.92 vs 2.75 average depth in 0.5 s).
- Likely main limit: training positions (Lichess analysis/puzzle positions) differ from what the
  engine's search evaluates. Next: positions from our own engine, scored by Stockfish.

## Round 2: ~149M positions, v2 architecture, self-play

Data: Lichess positions 0-150M (`data/processed` + `data/processed2`, ~148.5M train), validation still
the original 500k set. Self-play: `chess-engine datagen` (#40), 25,200 games at 8k nodes/move from 8
random plies, 1.85M quiet positions; 593,518 of them scored by Stockfish at depth 10 before the run was
stopped for low memory.

| Name | Arch | Data | Epochs | Best val MSE | In engine |
|---|---|---|---|---|---|
| v2-150m (default since #39) | v2, 256 | ~148.5M Lichess | 10 | 0.02648 | +64 ± 34 vs hand-written (0.1 s/move, 400 games); +120 ± 23 in CI |
| v1-150m | v1, 256 | same | 7 of 10 (stopped) | 0.02926 | not played; v2 is clearly better at equal epochs (0.0282 vs 0.0301 at epoch 4) |
| v2h512-150m | v2, 512 | same | 3 of 10 (stopped, low memory) | 0.02696 | not played yet; v2-256 was at 0.0291 after 3 epochs |
| v2-sp1-ft | v2, 256 | fine-tuned from v2-150m: Lichess + self-play x10 (~11% of each epoch), lr 3e-4 | 3 | 0.02637 | **-22 ± 29** vs v2-150m (0.1 s/move, 400 games): no gain |

Self-play round 1 did not help. Likely causes, to test next: depth-10 labels are much noisier than the
Lichess labels (average depth ~31); 8 random opening plies give unusual positions; 10x repetition gave
the noisy labels too much weight.

## Round 3: network width and longer training (first layer 256 / 384 / 448)

All on the same ~149M Lichess positions, validation on the original 500k set. Matches: same engine
binary, 800 games at 0.1 s/move (400 from the 40 openings + 400 with 4 extra seeded random plies).

| Name | Width | Training | Best val MSE | nodes/s (bench 12) | Match |
|---|---|---|---|---|---|
| v2-150m-cont (#57) | 256 | v2-150m + 5 epochs, lr 3e-4 | 0.02623 | ~3.0M | +16 ± 21 vs v2-150m |
| v2-150m-cont2 | 256 | + 5 more epochs, lr 2e-4 | 0.02609 | ~3.0M | -0 ± 20 vs v2-150m-cont |
| **v2h384-150m (#58)** | **384** | 10 epochs from scratch | **0.02504** | ~2.75M | **+50 ± 22** vs v2-150m-cont |
| v2h448-150m | 448 | 10 epochs from scratch | 0.02456 | ~2.45M | -13 ± 21 vs v2h384 |

Longer training of the 256-wide net stopped paying after 5 epochs. Width paid off at 384, but at 448 the
slightly better loss (-1.9%) no longer made up for 11% less speed at this time control: 384 is the sweet
spot here. Next lever: more data (the size curve was still improving at every doubling).

## Round 4: king buckets (arch v3, #64)

v2 plus left-right mirroring on the own king's half and 4 king buckets (back rank a-b, back rank c-d,
2nd rank, beyond), with a factoriser folded in at export. Validation on Lichess (the original 500k set)
and on held-out self-play positions, which are closer to what the search actually evaluates. Matches:
same PGO binary, 800 games at 5s+0.05s unless noted.

| Name | Data | Lichess val | Self-play 2b val | Self-play 1 val | nodes/s | Match |
|---|---|---|---|---|---|---|
| v2h384-150m | ~149M Lichess | 0.02504 | 0.01576 | 0.01637 | 2.86M | |
| v3h384-150m | same | 0.02297 | 0.01514 | 0.01677 | 2.68M | -6 ± 20 vs v2h384-150m; -4 ± 21 at 100k nodes |
| v2h384-300m | ~297M Lichess | 0.02443 | 0.01430 | 0.01412 | 2.86M | |
| v2h384-sp2b-ft4 (#62 default) | v2h384-300m + self-play 2b x4 | 0.02395 | 0.01430 | 0.01533 | 2.86M | |
| **v3h384-300m** | ~297M Lichess | **0.02235** | **0.01304** | **0.01401** | 2.68M | **+58 ± 22 vs #62's net** |

At 150M positions the buckets mainly fit the Lichess positions better and did not help in play, at equal
time or equal nodes. With twice the data they also fit self-play positions 9% better than v2 on the same
data, and that shows up in games despite 6% fewer nodes/s. The Lichess-only validation loss was
misleading here: the self-play validation sets track match results much better.

## Round 5: more Lichess data (412M positions)

The Lichess eval dump had ~116M positions past the 300M already used (`data/processed4`). Same recipe as
v3h384-300m (v3, 384 wide, 10 epochs from scratch, cosine LR).

| Name | Training | Lichess val | Self-play 2b val | Self-play 3 val | Match |
|---|---|---|---|---|---|
| **v3h384-300m (default)** | ~297M Lichess | 0.02235 | **0.01304** | **0.01280** | |
| v3h384-415m-r | 7 epochs, killed by a reboot, then 3 more via `--init` (fresh Adam, new LR schedule) | 0.02219 | 0.01393 | 0.01375 | -33 ± 22 vs default (800 games, 100k nodes) |
| v3h384-415m-b | 10 epochs, resumed with `--resume` (exact continuation) | **0.02207** | 0.01321 (best epoch) | 0.01289 (best epoch) | not played |

The extra data improves the Lichess validation loss but not the self-play ones, which are what tracks
match results. Restarting the schedule with `--init` cost about 7% on self-play val; `--resume` (#69)
avoids that. More Lichess positions of this kind are not the next lever.
