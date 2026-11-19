# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zakkysp/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zakkysp/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A program that forecasts every GTA VI PlayStation 5 trophy's earn rate on Exophase at a case's target time, as five quantiles (5/25/50/75/95) on the logit scale. It reads a case's history of Exophase page reads and writes one forecast file per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is the only entrypoint (Python 3.13, standard library only, no network, deterministic — two runs write byte-identical files in about 0.1 s per case).

## How it works

Exophase's page only moves when Exophase refreshes its counts, so a history is a staircase: long flat runs, then jumps at refreshes. When a refresh lands, the players-tracked count jumps (launch night: 638 → 3,868 in one refresh) and every rate drops, because the new cohort has earned little yet — then the rates climb back as those players play. A forecast that just persists the last rate, with a narrow random-walk spread, gets every trophy wrong at a refresh (that is exactly how the house baseline scores ~1.51 with ~0 coverage on the practice cases).

This program models the two counts separately, per trophy, on a log-time axis:

- It collapses each history to its distinct refresh snapshots (identical consecutive reads carry no information) and fits weighted least squares lines for `ln(players tracked)` and `ln(earned)` against `ln(1 + hours since release)`, with recency weights (a day-old snapshot counts half as much as the last one).
- The median is the ratio of the two extrapolations, anchored on the last snapshot so a slope error cannot move the level. Per-trophy slopes are shrunk toward a per-kind prior (story trophies climb fast, grind trophies barely move), the tracked-players slope is damped by 0.75 (the launch surge fades), and the median is capped by a per-kind ceiling so a runaway extrapolation cannot claim a week-one chapter-1 rate of 90%.
- The spread on the logit scale combines the fit's residual noise, the slope uncertainty times the extrapolation distance, and a refresh-jump term: the pooled spread of the rate jumps the history actually shows at refreshes, scaled by how many refreshes the horizon should hold. The total spread never goes below 0.40 logit units — a history that happens to look smooth is not evidence the future is.
- When the history is flat (every read identical, as on launch night) there is no slope to fit, so the program falls back to a refresh prior: if the page still sits at its launch-night count of 638 and the target is at or after the published first refresh (2026-11-19 04:30 UTC), it assumes that refresh lands before the target — denominator jumps to 3,868 while each trophy's earned count grows at its kind's prior rate. The trophy at 100% is exempt: it stays at 100%, as the rules state.

Calibration: the live Exophase page ~14 h after launch showed the pattern this model expects — the platinum down to 0.03% (its earned count flat while the denominator grew ~19×), Chapter 1 at 0.73% after dipping at the refresh (earned count up ~11×, denominator up ~19×), and most grind trophies at the 0.03–0.06% floor. Those numbers set the kind priors and the refresh-jump scale; the program itself fits whatever history a case gives it at runtime.

## What it forecasts

For each case file `/input/<id>.in` it writes `/output/<id>.out`: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, five finite numbers from 0 to 100, non-decreasing. On the practice cases its 90% bands average ~1.97 logit units wide (the starter's: ~0.41) — wide enough to cover a refresh-sized move — and its medians move the right way: Chapter 1 from 1.25% to ~0.31% at the 90-minute practice target, other trophies to ~0.2%.

Honest limits: the future refresh schedule is guessed, not known — the spread covers a refresh-sized miss but a refresh that lands just after the target instead of just before it (or a much bigger tracked-players jump) can still push the truth outside the band. The kind priors and ceilings are judgment calls from one day of live data, and a trophy whose history is flat gives the program nothing to fit — its median is then a prior, not a measurement.

---

<sub>Entered on [make.fun](https://make.fun/?ref=5twwhvj5&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
