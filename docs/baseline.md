# Baseline: twitchchess

The engine to beat: [geohot/twitchchess](https://github.com/geohot/twitchchess) at commit `d087c2d` (master), measured on 2026-10-17.

## Results

Opponent: Stockfish 17.1 with `UCI_LimitStrength` at `UCI_Elo` 1320 (its lowest calibrated level), 0.1 s per move. Colours alternate. Elo is the difference implied by the score, `-400 * log10(1/score - 1)`.

| Engine | Opponent | Games | +W =D -L | Score | Elo vs opponent | Estimate |
|---|---|---|---|---|---|---|
| Part 1 (value net, 1 ply) | Stockfish 1320 | 40 | +0 =0 -40 | 0% | below -750 | well under 1000 |
| Part 1 (value net, 1 ply) | random mover | 40 | +6 =32 -2 | 55% | +35 | |
| master (material + mobility, minimax) | Stockfish 1320 | 12 | +4 =5 -3 | 54% | +29 | **about 1350** |
| master | Part 1 | 4 | +4 =0 -0 | 100% | | |

Uncertainty is large: roughly ±100 Elo at 40 games and ±150 at 12 games. Stockfish's limiter is calibrated at longer time controls than 0.1 s per move, so treat the absolute numbers as approximate and the comparisons between engines as the useful part.

## What the numbers say

- **Part 1 cannot convert won positions.** 32 of 40 games against a random mover were draws: with no search it rarely finds mate and drifts into repetitions or the 50-move rule.
- **master is a real engine.** Material plus a mobility term and a 5-ply search (3 full + 2 beam of 10) scores about even with Stockfish 1320, but it spends 7-28 s per move to do it.

## Speed

CPython 3.10, one core, RTX 4050 laptop.

| Engine | Search | Time per move |
|---|---|---|
| Part 1 | 1 ply, one net eval per legal move | ~0.01 s |
| master | ~8.5k nodes/s, 57k nodes (start), 235k nodes (middlegame) | 6.6 s (start), 28.4 s (middlegame) |

master's search reports +4.00 from the start position while the static eval is 0.00, which suggests a bias in its search (not investigated).

## Reproduce

```
git clone https://github.com/geohot/twitchchess ../twitchchess
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt torch flask
TWITCHCHESS=../twitchchess tools/baseline.sh   # results in runs/baseline/*.log
```

Needs Stockfish at `~/.local/share/stockfish/stockfish-17.1` or `STOCKFISH=/path/to/stockfish`. The master engine is slow, so the full run takes about 1.5 hours with matches in parallel shards.
