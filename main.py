"""Forecast GTA VI trophy earn rates on Exophase at a case's target time.

The model, in one paragraph: Exophase's page only moves when Exophase
refreshes its counts, so a case's history is a staircase -- long flat runs
with jumps at refreshes. Between refreshes the number of players tracked is
frozen and the earned counts are frozen; at a refresh both jump to their true
values. So this program models the two counts separately, per trophy, on a
log-time axis: ln(players tracked) and ln(earned) each get a weighted least
squares line against ln(1 + hours since release), fitted on the history's
distinct refresh snapshots only (identical consecutive reads carry no new
information). The median forecast is the ratio of the two extrapolations,
anchored on the last snapshot so a slope error cannot move the level, and
capped by a per-kind ceiling (a week-one chapter-1 rate cannot be 90%).
The spread on the logit scale combines the fit's residual noise, the slope
uncertainty times the extrapolation distance, and a refresh-jump term: the
pooled spread of the rate jumps the history actually shows at refreshes,
scaled by how many refreshes the horizon should hold. When the history is
flat -- every read identical, as on launch night -- there is no slope to fit,
so the program falls back to a refresh prior: the published first refresh
took the page from 638 to 3,868 tracked players, and the earned counts grow
at a per-kind rate while the denominator jumps, which is what drags every
rate down at a refresh.

Everything is deterministic: no randomness, no network, standard library
only. One entrypoint, main.py, at the repository root.

Usage (as the grader runs it):
    python3 /bot/main.py            (reads /input, writes /output)
Locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The game's release, the axis the time trend is measured on. Taken from the
# case's game.release_date when present; the fallback is the release day.
DEFAULT_RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# The first refresh of the game's counts, as the competition's rules publish
# it: launch night's page sat at 638 tracked players, then refreshed to
# 3,868, every rate but the one at 100% lower. A flat history that still
# sits at 638 has not seen that refresh yet.
FIRST_REFRESH_AT = datetime(2026, 11, 19, 4, 30, tzinfo=timezone.utc)
FIRST_REFRESH_PLAYERS = 3868.0
LAUNCH_NIGHT_PLAYERS = 638.0

# Trophy kinds, by Exophase list position (the official 55). Story is the 12
# chapter and story-mission trophies, collectibles the 4 collection trophies,
# everything else activities; the platinum stands alone.
STORY_IDS = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
COLLECTIBLE_IDS = {28, 30, 31, 35}
PLATINUM_ID = 1

# How fast earned counts grow, in ln(earned) per ln(1 + hours since release),
# when a trophy has no usable history of its own. Story trophies climb fast
# as the first cohorts play through the chapters; grind trophies barely move.
PRIOR_SLOPE = {"platinum": 0.05, "story": 0.9, "collectibles": 0.25, "activities": 0.35}

# Ceilings for a week-one rate, in percent, by kind. These only ever bind a
# runaway extrapolation; the fitted trend decides the median below them.
CAP = {"platinum": 2.0, "story": 25.0, "collectibles": 5.0, "activities": 8.0}
CHAPTER_CAP = {1: 45.0, 2: 35.0, 3: 28.0, 4: 22.0}
FINAL_MISSION_CAP = 18.0
GRIND_WORDS = (
    "250", "100%", "gold medal", "every location", "each gang",
    "all of leonida", "all garage", "100 locations", "100 vehicles",
    "20 different", "50 challenges", "8 mission strands",
)
EASY_WORDS = (
    "arcade game", "cocktail", "bowling", "fishing challenge",
    "wanted level", "combat focus", "melee drive-by", "marked bills",
    "$100,000", "scooter bros", "vehicle showroom", "unscheduled departure",
    "rhinos", "jewelry store", "armored truck", "unique animal",
)
GRIND_CAP = 2.5
EASY_CAP = 12.0


def parse_time(text):
    """An ISO 8601 timestamp (with Z or an offset, seconds optional)."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def release_moment(case):
    game = case.get("game") or {}
    raw = game.get("release_date")
    if raw:
        raw = str(raw).strip()
        try:
            if "T" in raw:
                return parse_time(raw)
            return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return DEFAULT_RELEASE


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, halves away from zero, as a plain number.

    A whole result is returned as an int so the JSON writes 100, not 100.0.
    """
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def kind_of(trophy):
    tid = trophy.get("id")
    if tid == PLATINUM_ID:
        return "platinum"
    if tid in STORY_IDS:
        return "story"
    if tid in COLLECTIBLE_IDS:
        return "collectibles"
    return "activities"


def chapter_of(trophy):
    """The chapter number of a 'Complete Chapter N' trophy, else None."""
    desc = (trophy.get("description") or "").strip()
    if desc.startswith("Complete Chapter "):
        tail = desc[len("Complete Chapter "):].rstrip(".")
        if tail.isdigit():
            return int(tail)
    return None


def ceiling(trophy, kind):
    if kind == "platinum":
        return CAP["platinum"]
    if kind == "collectibles":
        return CAP["collectibles"]
    if kind == "story":
        chapter = chapter_of(trophy)
        if chapter is not None:
            return CHAPTER_CAP.get(chapter, 30.0)
        if "final mission" in (trophy.get("description") or "").lower():
            return FINAL_MISSION_CAP
        return CAP["story"]
    desc = (trophy.get("description") or "").lower()
    if any(word in desc for word in GRIND_WORDS):
        return GRIND_CAP
    if any(word in desc for word in EASY_WORDS):
        return EASY_CAP
    return CAP["activities"]


def prior_slope(trophy, kind):
    if kind != "story":
        return PRIOR_SLOPE[kind]
    chapter = chapter_of(trophy)
    if chapter is not None:
        return max(0.5, 0.9 - 0.1 * (chapter - 1))
    if "final mission" in (trophy.get("description") or "").lower():
        return 0.5
    return 0.7


def wls(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x.

    Returns (slope, intercept, slope_standard_error, residual_std). With
    fewer than two distinct x values the slope is 0 and the errors are None.
    """
    sw = sum(ws)
    if sw <= 0:
        return 0.0, (ys[0] if ys else 0.0), None, None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return 0.0, my, None, None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = my - slope * mx
    n = len(xs)
    if n > 2:
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        var = sum(w * r * r for r, w in zip(resid, ws)) / sw
        dof = max(1, n - 2)
        resid_std = math.sqrt(var * n / dof)
        slope_se = resid_std / math.sqrt(sxx) if sxx > 0 else None
    else:
        resid_std = 0.0
        slope_se = None
    return slope, intercept, slope_se, resid_std


def distinct_snapshots(case):
    """The history with identical consecutive reads collapsed to one.

    A read is identified by its players-tracked count and its whole rate
    vector; the page does not move between refreshes, so repeats add nothing.
    Returns a list of (moment, players, rates) oldest first.
    """
    out = []
    prev_key = None
    for read in case.get("history") or []:
        rates = tuple(read.get("rates") or [])
        key = (read.get("players_tracked"), rates)
        if key == prev_key:
            continue
        prev_key = key
        out.append((parse_time(read["at"]), read["players_tracked"], rates))
    return out


def median_gap_hours(snaps):
    if len(snaps) < 2:
        return 5.0
    gaps = sorted(
        (b[0] - a[0]).total_seconds() / 3600.0
        for a, b in zip(snaps, snaps[1:])
    )
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return 5.0
    return gaps[len(gaps) // 2]


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r" % (case.get("levels"), LEVELS))
    trophies = case["trophies"]
    release = release_moment(case)
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_hours = max(0.0, (target - checkpoint).total_seconds() / 3600.0)

    def x_of(moment):
        hours = (moment - release).total_seconds() / 3600.0
        return math.log1p(max(hours, -0.9))

    x_target = x_of(target)
    snaps = distinct_snapshots(case)
    last_read = (case.get("history") or [None])[-1]

    # The last rate each trophy was seen at, for the near-100 guard and for
    # histories too short to fit.
    last_rate = {}
    if last_read is not None:
        for i, trophy in enumerate(trophies):
            rates = last_read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last_rate[trophy["id"]] = float(rates[i])

    if len(snaps) >= 2:
        rows = forecast_from_fit(
            case, trophies, snaps, x_target, horizon_hours, last_rate, x_of
        )
    else:
        rows = forecast_from_prior(
            case, trophies, snaps, last_read, x_target, horizon_hours,
            last_rate, x_of, target,
        )
    return {"trophies": rows}


def forecast_from_fit(case, trophies, snaps, x_target, horizon_hours, last_rate, x_of):
    """Fit ln(earned) and ln(tracked) against ln(1 + hours) and extrapolate."""
    times = [s[0] for s in snaps]
    xs = [x_of(t) for t in times]
    last_t = times[-1]
    # Recency weights: a snapshot a day old counts half as much as the last.
    weights = [0.5 ** ((last_t - t).total_seconds() / 86400.0) for t in times]
    x_last = xs[-1]
    dx = max(0.0, x_target - x_last)

    # The tracked-players trend, shared by every trophy.
    n_ys = [math.log(max(1.0, float(s[1]))) for s in snaps]
    d_slope, _d_int, d_se, _d_resid = wls(xs, n_ys, weights)
    d_slope = min(1.5, max(0.0, d_slope))
    # The launch surge fades: the future growth runs slower than the
    # average slope the history shows.
    d_future = 0.75 * d_slope
    n_last = max(1.0, float(snaps[-1][1]))
    d_se = d_se if d_se is not None else 0.2

    gap = median_gap_hours(snaps)
    n_refresh = min(horizon_hours / gap if gap > 0 else 1.0, 4.0)

    # First pass: each trophy's median and its own sigma; collect the
    # residual spreads for the pooled refresh term below.
    prelim = []
    resid_sq_sum = 0.0
    for i, trophy in enumerate(trophies):
        kind = kind_of(trophy)
        cap = ceiling(trophy, kind)
        rate_last = last_rate.get(trophy["id"])
        if rate_last is not None and rate_last >= 99.0:
            prelim.append((min(99.99, rate_last), 0.08, trophy["id"], True))
            continue

        # Earned counts for this trophy at each snapshot (a printed 0.00%
        # means fewer than half a player, so 0.5 stands in for the log).
        ex, ey, ew = [], [], []
        for j, snap in enumerate(snaps):
            rates = snap[2]
            if i >= len(rates) or rates[i] is None:
                continue
            earned = float(rates[i]) / 100.0 * float(snap[1])
            ex.append(xs[j])
            ey.append(math.log(max(earned, 0.5)))
            ew.append(weights[j])
        b_prior = prior_slope(trophy, kind)
        if len(ex) >= 2:
            b_slope, _b_int, b_se, b_resid = wls(ex, ey, ew)
            b_slope = min(2.0, max(-0.5, b_slope))
            # Shrink a noisy per-trophy slope toward its kind's prior; with
            # many snapshots the data wins, with few the prior does.
            eff = len(ex)
            b_slope = (eff * b_slope + 2.0 * b_prior) / (eff + 2.0)
            resid = min(0.6, max(0.15, b_resid or 0.15))
            slope_se = b_se if b_se is not None else 0.25
        else:
            b_slope = b_prior
            resid = 0.25
            slope_se = 0.25
        resid_sq_sum += resid * resid

        # Anchor on the last observed earned count, not on the fit's level.
        if rate_last is not None:
            e_last = max(float(rate_last) / 100.0 * n_last, 0.5)
        else:
            e_last = 0.5
        e_target = e_last * math.exp(b_slope * dx)
        n_target = n_last * math.exp(d_future * dx)
        median = min(cap, clamp_rate(100.0 * e_target / n_target))
        sigma_slope = min(0.8, max(0.08, (slope_se + d_se) * dx))
        sigma = math.sqrt(resid ** 2 + sigma_slope ** 2)
        prelim.append((median, sigma, trophy["id"], False))

    # Second pass: the pooled refresh-jump term, from the average residual
    # spread across trophies, scaled by the refreshes the horizon holds.
    # The 0.5 damping is deliberate: the fitted trend already absorbs the
    # systematic part of each refresh, so only about half the jump spread
    # is surprise left to cover.
    pooled = math.sqrt(resid_sq_sum / max(1, len(trophies)))
    pooled = min(0.7, max(0.2, pooled))
    sigma_event = pooled * math.sqrt(n_refresh) * 0.5
    out = []
    for median, sigma, tid, frozen in prelim:
        if frozen:
            out.append(quantiles(median, sigma, tid))
        else:
            # Never claim more precision than 0.40 logit units: a history
            # that happens to look smooth is not evidence the future is.
            total = max(0.40, math.sqrt(sigma ** 2 + sigma_event ** 2))
            out.append(quantiles(median, total, tid))
    return out


def forecast_from_prior(case, trophies, snaps, last_read, x_target,
                        horizon_hours, last_rate, x_of, target):
    """No usable trend in the history: fall back to a refresh prior.

    With a single distinct snapshot there is nothing to fit. If the page
    still sits at its launch-night count and the target is at or after the
    published first refresh, assume that refresh lands before the target:
    the denominator jumps to 3,868 while each trophy's earned count grows
    at its kind's prior rate. Otherwise the rates sit where they are, with
    only the earned-count growth moving them.
    """
    if last_read is None:
        return [quantiles(prior_base(trophy), 1.2, trophy["id"])
                for trophy in trophies]
    n_last = float(last_read.get("players_tracked") or LAUNCH_NIGHT_PLAYERS)
    refresh = (
        n_last == LAUNCH_NIGHT_PLAYERS and target >= FIRST_REFRESH_AT
    )
    n_target = FIRST_REFRESH_PLAYERS if refresh else n_last
    x_last = x_of(parse_time(last_read["at"]))
    dx = max(0.0, x_target - x_last)
    n_ref = min(horizon_hours / 5.0 if horizon_hours > 0 else 0.0, 4.0)
    jump = 1.0 if refresh else 0.5
    sigma_event = jump * math.sqrt(n_ref) if n_ref > 0 else 0.3
    rows = []
    for trophy in trophies:
        kind = kind_of(trophy)
        cap = ceiling(trophy, kind)
        rate_last = last_rate.get(trophy["id"])
        if rate_last is not None and rate_last >= 99.0:
            rows.append(quantiles(min(99.99, rate_last), 0.08, trophy["id"]))
            continue
        if rate_last is None:
            rows.append(quantiles(prior_base(trophy), 1.2, trophy["id"]))
            continue
        e_last = max(float(rate_last) / 100.0 * n_last, 0.5)
        e_target = e_last * math.exp(prior_slope(trophy, kind) * dx)
        median = min(cap, clamp_rate(100.0 * e_target / n_target))
        sigma = math.sqrt(0.25 ** 2 + 0.10 ** 2 + sigma_event ** 2)
        rows.append(quantiles(median, sigma, trophy["id"]))
    return rows


def prior_base(trophy):
    """A week-one rate guess for a trophy never seen at all, in percent."""
    kind = kind_of(trophy)
    if kind == "platinum":
        return 0.05
    if kind == "story":
        chapter = chapter_of(trophy)
        if chapter is not None:
            return max(0.1, 1.0 - 0.2 * (chapter - 1))
        return 0.5
    if kind == "collectibles":
        return 0.1
    return 0.1


def quantiles(median_pct, sigma, tid):
    """Five quantiles of a logit-normal forecast, rounded to 4 decimals."""
    centre = logit(median_pct)
    qs = []
    prev = None
    for z in Z:
        value = round4(clamp_rate(inv_logit(centre + z * sigma)))
        if prev is not None and value < prev:
            value = prev
        qs.append(value)
        prev = value
    return {"id": tid, "q": qs}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
