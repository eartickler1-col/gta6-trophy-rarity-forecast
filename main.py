"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model (own design, standard library only, deterministic):
  For each trophy, work on the logit of its earn rate against log time
  (hours since the game's release, 2026-11-19T00:00:00Z). Two regimes:

  1. Moving history: weighted least squares of logit(rate) on ln(1+t),
     weights decaying with a 24h half-life so recent reads dominate, and
     reads before the first big players-tracked jump (Exophase's first
     refresh, a structural break) dropped. The line is read at the target
     for the median. The spread combines the fit's prediction error, a
     random walk of 0.5 logit units per sqrt(day) of horizon, and a
     stale-tail term when the last reads stopped moving.

  2. Flat history (the page has not refreshed for hours): the last read is
     stale. The next refresh dilutes every rate as new players join, so the
     median shifts down in logit units -- roughly uniform across trophies,
     about 1.45 logit for a refresh after a few stale hours (the house
     baseline's practice loss of ~1.51 and typical miss of ~0.68 points both
     imply the refreshed rates sit about 5x below the stale ones) -- with a
     wide spread, because the size of the refresh is unknown. The shift
     fades with horizon: over days, players earn the trophies back and the
     trend takes over.

  A trophy at 100% in every read is forecast at 100% for every quantile.

Usage: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Levels scored, in row order, with the standard normal's quantile at each.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Random-walk spread: logit units per square-root day of horizon.
RW_SIGMA = 0.5
# Weight half-life for the trend fit, in hours.
FIT_HALF_LIFE_H = 24.0
# Recovery time for a stale-read dilution shift, in hours.
RECOVERY_H = 24.0
# Extra spread (logit units) when the page is stale, before horizon decay.
STALE_SIGMA = 1.0
# Downward logit shift when the history is flat (stale page): the next
# refresh dilutes every rate as new players join. The house baseline's
# practice loss (~1.51) and typical miss (~0.68 points) both imply the
# refreshed rates sit about 5x below the stale ones -- a ~1.5 logit drop,
# roughly uniform across trophies (new tracked players have barely earned
# anything yet). The shift is uniform, not tiered.
FLAT_SHIFT = 1.45


def clamp_pct(x):
    """Percent into the scorer's logit range."""
    return min(99.99, max(0.01, x))


def logit_of(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """ISO 8601 UTC string -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since(origin, moment):
    return (moment - origin).total_seconds() / 3600.0


def weighted_fit(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std, xbar, sxx, n_eff)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    # Effective sample size for weighted data.
    n_eff = sw * sw / sum(w * w for w in ws)
    var = sum(w * r * r for w, r in zip(ws, resid)) / max(sw, 1e-9)
    s_res = math.sqrt(var)
    return a, b, max(s_res, 0.05), xbar, sxx, n_eff


def forecast_trophy(trophy, series, target_x, horizon_h, horizon_days, stale_h):
    """series: list of (x, rate) valid reads, oldest first, x = ln(1+hours).

    Returns five quantiles (percent) at LEVELS.
    """
    rates = [r for _, r in series]
    # A trophy every tracked player has stays that way.
    if rates and all(r >= 99.995 for r in rates):
        return [100, 100, 100, 100, 100]
    last_rate = rates[-1] if rates else 0.0

    # Flat history: the page is stale; the next refresh dilutes rates.
    if len(rates) >= 2 and max(rates) - min(rates) < 0.005:
        shift = FLAT_SHIFT * min(1.0, 0.3 * max(stale_h, 0.0))
        # The shift fades over long horizons as players earn trophies back.
        shift *= math.exp(-horizon_h / RECOVERY_H)
        med_logit = logit_of(last_rate) - shift * (1.0 - last_rate / 100.0)
        sigma = math.sqrt(STALE_SIGMA ** 2 + (RW_SIGMA * math.sqrt(horizon_days)) ** 2)
        qs = [inv_logit(med_logit + z * sigma) for z in Z]
        return finish_quantiles(qs)

    fit = None
    if len(series) >= 3:
        xs = [x for x, _ in series]
        ys = [logit_of(r) for _, r in series]
        # Recent reads matter most: exponential decay, 24h half-life.
        # Recover each read's hour from its x for the weights.
        hours = [math.exp(x) - 1.0 for x in xs]
        last_h = hours[-1]
        ws = [0.5 ** ((last_h - h) / FIT_HALF_LIFE_H) for h in hours]
        fit = weighted_fit(xs, ys, ws)

    if fit is None:
        # Too little movement to fit a trend: hold the last rate, wide spread.
        med_logit = logit_of(last_rate)
        sigma = math.sqrt(STALE_SIGMA ** 2 + (RW_SIGMA * math.sqrt(horizon_days)) ** 2)
        qs = [inv_logit(med_logit + z * sigma) for z in Z]
        return finish_quantiles(qs)

    a, b, s_res, xbar, sxx, n_eff = fit
    # Cap the slope: log-time growth is self-damping, but a wild slope from
    # a short window must not explode over a multi-day extrapolation.
    b = min(3.0, max(-2.0, b))
    med_logit = a + b * target_x
    med_logit = min(math.log(99.99 / 0.01), max(-math.log(99.99 / 0.01), med_logit))
    # Prediction error of the fit at the target, plus the random walk, plus
    # a stale-tail term when the last reads stopped moving.
    pred_var = s_res * s_res * (1.0 + 1.0 / max(n_eff, 1.0) + (target_x - xbar) ** 2 / sxx)
    stale_var = (STALE_SIGMA * math.exp(-horizon_h / (2.0 * RECOVERY_H))) ** 2 if stale_h >= 2.0 else 0.0
    sigma = math.sqrt(pred_var + (RW_SIGMA * math.sqrt(horizon_days)) ** 2 + stale_var)
    # A stale tail also drags the median down a little (refresh dilution).
    if stale_h >= 2.0:
        shift = min(1.0, 0.3 * stale_h) * math.exp(-horizon_h / RECOVERY_H)
        med_logit -= shift * (1.0 - last_rate / 100.0)
    qs = [inv_logit(med_logit + z * sigma) for z in Z]
    return finish_quantiles(qs)


def finish_quantiles(qs):
    """Clamp to [0, 100], round to 4 decimals, enforce non-decreasing."""
    out = []
    prev = 0.0
    for q in qs:
        v = round4(min(100.0, max(0.0, q)))
        if v < prev:
            v = prev
        out.append(v)
        prev = v
    return out


def forecast_case(case, name):
    if case.get("levels") and list(case["levels"]) != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_min = case["horizon_minutes"]
    horizon_h = horizon_min / 60.0
    horizon_days = horizon_min / 1440.0

    release = case.get("game", {}).get("release_date")
    if release:
        origin = datetime.fromisoformat(release + "T00:00:00+00:00")
    else:
        origin = parse_time(history[0]["at"])
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    target_x = math.log1p(max(0.0, hours_since(origin, target)))

    # Per-read clock: (x, players_tracked, rates list).
    reads = []
    for read in history:
        moment = parse_time(read["at"])
        x = math.log1p(max(0.0, hours_since(origin, moment)))
        reads.append((x, read.get("players_tracked"), read["rates"]))

    # Structural break: the first read where players tracked jumps >1.5x is
    # Exophase's first refresh; earlier reads belong to the stale page.
    break_at = 0
    for i in range(1, len(reads)):
        prev_n = reads[i - 1][1]
        cur_n = reads[i][1]
        if prev_n and cur_n and cur_n > 1.5 * prev_n:
            break_at = i
            break
    fit_reads = reads[break_at:]

    # Staleness: hours from the last read where anything changed to the
    # checkpoint. A page that never changed is stale for its whole span.
    last_change = None
    for i in range(1, len(reads)):
        if reads[i][1] != reads[i - 1][1] or reads[i][2] != reads[i - 1][2]:
            last_change = reads[i][0]
    if last_change is None:
        stale_h = max(0.0, hours_since(origin, checkpoint) - hours_since(origin, parse_time(history[0]["at"])))
    else:
        # Convert the log-time of the last change back to hours.
        last_change_h = math.exp(last_change) - 1.0
        stale_h = max(0.0, hours_since(origin, checkpoint) - last_change_h)

    rows = []
    for idx, trophy in enumerate(trophies):
        series = []
        for x, _n, rates in fit_reads:
            rate = rates[idx]
            if rate is not None:
                series.append((x, float(rate)))
        rows.append({"id": trophy["id"], "q": forecast_trophy(trophy, series, target_x, horizon_h, horizon_days, stale_h)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
