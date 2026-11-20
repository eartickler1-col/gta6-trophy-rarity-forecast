# Vice Logit Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/eartickler1-col/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/eartickler1-col/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A Python program that forecasts every GTA VI trophy's Exophase earn rate at 2026-11-26 00:00 UTC as five quantiles (5%, 25%, 50%, 75%, 95%), from the history of reads in each case file. Standard library only, offline and deterministic.

How it forecasts: per trophy, a weighted logit-linear trend against log time (recent reads weighted most, reads before Exophase's first refresh dropped as a structural break), read at the target for the median, with a spread from the fit's prediction error plus a random walk that grows with the horizon. When the page has not refreshed for hours, the last read is stale, so the median shifts down for the dilution the next refresh brings and the spread widens. A trophy at 100% stays at 100%.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
