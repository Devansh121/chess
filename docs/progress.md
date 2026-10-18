# Progress

Strength over time, with how each number was measured. Ratings marked "vs Stockfish" come from games
against Stockfish 17.1 with `UCI_LimitStrength` (0.1 s per move, 40 openings from both sides): the
opponent's `UCI_Elo` plus the Elo difference implied by the score. Stockfish's limiter is calibrated at
longer time controls, so treat absolute numbers as approximate and comparisons as the useful part.

## Rating history

| Engine | How measured | Result | Rating |
|---|---|---|---|
| twitchchess Part 1 (George Hotz, 2018) | vs Stockfish 1320, 40 games | +0 =0 -40 | well under 1000 |
| twitchchess master | vs Stockfish 1320, 12 games | +4 =5 -3 | ~1350 |
| Python stage A (alpha-beta) | vs Stockfish 1320, 40 games | +5 =12 -23 | ~1150 |
| Python stage D (+ quiescence, TT, ordering) | vs Stockfish 1320/1600/1900, 80 each | | ~1630 |
| C++ port of stage D | vs Python stage D, 80 games | +80 =0 -0 | |
| C++ port of stage D | vs Stockfish 1900 / 2200 / 2500, 80 each | +65 =6 -9 / +47 =13 -20 / +29 =16 -35 | ~2200 / ~2320 / ~2475 |
| C++ with search improvements (#15-#37) + v2 network | vs hand-written eval, same engine, 400 games | +228 =17 -155 (+64 ± 34) | |
| C++ with search improvements + v2 network | vs Stockfish 2500 / 2800 / 3000, 160 each | +94 =21 -45 (+110 ± 53) / +17 =54 -89 (-168 ± 46) / +4 =38 -118 (-310 ± 56) | ~2610 / ~2630 / ~2690, **~2640** |
| + int8 network kernel (#38) | vs int16 kernel, same net, 400 games at 0.1 s/move | +216 =120 -64 (+139 ± 30) | |
| + int8 network kernel (#38) | vs Stockfish 2800 / 3000, 160 each | +23 =57 -80 (-129 ± 45) / +5 =46 -109 (-269 ± 51) | ~2670 / ~2730, **~2700** |
| + counter moves (#44), eval cache (#53), neutral experiments reverted (#55) | vs Stockfish 2800 / 3000, 160 each | +42 =53 -65 (-50 ± 45) / +3 =45 -112 (-289 ± 52) | ~2750 / ~2710, **~2730** |

## Search changes (C++)

CI `elo` job: 1,000 games of each PR's engine vs its base branch, 100k nodes per move, hand-written
evaluation, 40 openings + 4 seeded random plies, both colours. Margins are ~95%.

| PR | Change | Elo vs previous |
|---|---|---|
| #14, #23 | CI tooling only (identical engines) | -12 ± 20, -0 ± 20 |
| #15 | principal variation search | -8 ± 20 |
| #16 | null-move pruning | +54 ± 20 |
| #17 | late-move reductions | +73 ± 20 |
| #18 | history heuristic | +18 ± 20 |
| #19 | check extension | +18 ± 20 |
| #20 | aspiration windows (±50) | +14 ± 20 |
| #21 | reverse futility pruning | +65 ± 20 |
| #22 | futility pruning | +60 ± 20 |
| #24 | SEE pruning in quiescence | +38 ± 20 |
| #25 | SEE move ordering | +4 ± 19 |
| #29 | TT in quiescence, entry aging | +9 ± 20 |
| #30 | internal iterative reductions | +25 ± 19 |
| #31 | "improving" heuristic | -11 ± 20 (needs a gentler variant) |
| #35 | staged lazy move picker | +5 ± 19 |
| #42 | "improving": drop the extra LMR ply | -8 ± 19 |
| #44 | counter-move heuristic | **+28 ± 19** |
| #46 | history-weighted LMR | -2 ± 19 (reverted in #55) |
| #47 | singular extensions | +1 ± 19 (reverted in #55) |
| #48 | continuation history | -1 ± 19 (reverted in #55) |

PRs that shouldn't change the hand-written search played exactly the same 1,000 games as their base
(#26/#27/#28: +406 =182 -412; #32/#33/#34: +395 =202 -403; #36/#37: +388 =231 -381), which confirms the
speed and network-only changes left the search untouched.

## Speed changes (C++)

| PR | Change | Effect |
|---|---|---|
| #12-#13 | C++ port | ~20k -> ~7.5M nodes/s (Python -> C++) |
| #28 | quantized int16 network with AVX2 | 0.13M -> 1.50M nodes/s with a network (~11x vs float) |
| #34 | TT prefetch | +7% nodes/s, identical nodes |
| #35 | lazy move picker | -7% time to depth |
| #36 | make each move once | +11% nodes/s, identical nodes |
| #37 | embedded magics | startup 0.33 s -> 0.056 s |
| #53 | network evaluation cache | +11% nodes/s, identical nodes |
| #38 | int8 hidden layer, 8-output kernel | +73% nodes/s with a network (idle machine); eval error 2.6 -> 5.5 cp mean; +139 ± 30 Elo vs int16 at 0.1 s/move |

## Networks

See `docs/experiments.md` for every training run. Highlights:

| Network | Data | Validation MSE | In engine |
|---|---|---|---|
| v1, Python engine era | 49.5M Lichess positions | 0.0299 | no gain over hand-written (Python) |
| v1 | ~149M positions | 0.0301 after 4 epochs | (comparison run) |
| **v2** (both perspectives, default since #39) | ~149M positions, 10 epochs | **0.0265** | +64 ± 34 vs hand-written (C++) |
