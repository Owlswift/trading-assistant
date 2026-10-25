"""Level identification: candle closes on a timeframe that price is currently near.

Rule (from the strategy notes):
- Every completed candle close on the timeframe is a candidate level.
- EXCEPT the close of the candle immediately before the current running candle
  (on Tuesday, Monday's close is not a level; Friday's and older closes are).
- A level is only surfaced when price is close to it, to avoid cluttering the chart.
- A level that has been BROKEN is ignored: once any later candle on the same timeframe has
  CLOSED beyond it (above a level that sits above price, below a level that sits below price),
  it is gone for good. Wicks through a level do not break it; closes do.
"""
import os

import pandas as pd

from . import data

# Levels are only identified from 4H upward. 1H is the confirmation timeframe, not a level source.
LEVEL_TFS = ("1mo", "1w", "1d", "4h")

# LEVEL POLARITY. What a close is a level FOR depends on what price did RIGHT AFTER it, i.e. on the candle that FOLLOWED the level candle:
#     Bullish -> Bearish : SELLING level          Bearish -> Bullish : BUYING level
#     Bullish -> Bullish : BUYING  level          Bearish -> Bearish : SELLING level
# So: the next candle went up = a buying level (a buy can start there), it went down = a selling level (a sell can start there). The level
# candle's own colour does not matter. This is why the close just before the running candle is not a level yet: the candle after it has not
# finished, so its job is not known. A doji next candle serves both. Both timeframes are the level's own. A buy can only START at a buying
# level (the level price hits, and its confirmation level), a sell only at a selling level. Targets are, by default, the next valid level of
# any kind; LEVEL_POLARITY=on also restricts them (a buy aims at a selling level, a sell at a buying level).
# LEVEL_POLARITY = origin (default: where a trade starts) | on (start and target) | target (only where it aims) | off (every close alike)
_MODE = os.getenv("LEVEL_POLARITY", "origin").strip().lower()
POLARITY_ON = _MODE != "off"
POLARITY_ORIGIN = _MODE in ("on", "origin")
POLARITY_TARGET = _MODE in ("on", "target")


def _dir(c: dict | None) -> str:
    if c is None:
        return "none"
    if "dir" in c:   # set by data.tag_direction: close vs the PREVIOUS candle's close
        return c["dir"]
    return "bull" if c["close"] > c["open"] else "bear" if c["close"] < c["open"] else "doji"


def polarity(next_candle: dict | None) -> str:
    """What a close is a level FOR, from the candle that FOLLOWED it: "buy" (it went up), "sell" (it went down) or "both" (doji / not known yet)."""
    d = _dir(next_candle)
    return "buy" if d == "bull" else "sell" if d == "bear" else "both"


def formation(level_candle: dict, next_candle: dict | None) -> str:
    names = {"bull": "Bullish", "bear": "Bearish", "doji": "Doji", "none": "?"}
    return f"{names[_dir(level_candle)]}→{names[_dir(next_candle)]}"


def origin_ok(next_candle: dict | None, bull: bool) -> bool:
    """May a BUY (bull) / SELL start at the level whose following candle is `next_candle`?"""
    if not POLARITY_ORIGIN:
        return True
    p = polarity(next_candle)
    return p == "both" or p == ("buy" if bull else "sell")


def target_ok(next_candle: dict | None, bull: bool) -> bool:
    """May that level be the TARGET of a buy (bull) / sell? (Only enforced with LEVEL_POLARITY=on or target.)"""
    if not POLARITY_TARGET:
        return True
    p = polarity(next_candle)
    return p == "both" or p == ("sell" if bull else "buy")


def _atr14(candles: list[dict]) -> float | None:
    df = pd.DataFrame(candles[-60:])
    if len(df) < 15:
        return None
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return float(tr.rolling(14).mean().iloc[-1])


def intact_closes(candles: list[dict], price: float) -> list[dict]:
    """candles[-1] is the running candle, candles[-2] the immediate previous one (excluded).

    A close L at candle i is still intact if no later candle (including the running one) has
    closed beyond it on the side away from price: for L above price, no later close above L;
    for L below price, no later close below L.
    """
    closes = [c["close"] for c in candles]
    n = len(closes)
    # later_max[i] / later_min[i] = highest / lowest close among candles i+1 .. n-1
    later_max = [float("-inf")] * n
    later_min = [float("inf")] * n
    for i in range(n - 2, -1, -1):
        later_max[i] = max(later_max[i + 1], closes[i + 1])
        later_min[i] = min(later_min[i + 1], closes[i + 1])

    out = []
    for i in range(n - 2):  # excludes the immediate previous candle and the running one
        lv = closes[i]
        if lv > price and later_max[i] > lv:
            continue  # price already closed above it: broken
        if lv < price and later_min[i] < lv:
            continue  # price already closed below it: broken
        out.append({"price": lv, "time": candles[i]["time"], "polarity": polarity(candles[i + 1] if i + 1 < n else None),
                    "formation": formation(candles[i], candles[i + 1] if i + 1 < n else None)})
    return out


def nearby_levels(price: float, atr_mult: float = 0.5, tfs: tuple[str, ...] = LEVEL_TFS) -> list[dict]:
    out = []
    every: dict[str, dict[float, dict]] = {}     # every valid close of every timeframe, by price: a level near price on one timeframe is
    for tf in LEVEL_TFS:                         # also named after the other timeframes that have a level at the same price (any distance)
        cs = data.try_candles(tf)
        if cs:
            every[tf] = {round(lv["price"], 2): lv for lv in intact_closes(cs, price)}
    for tf in tfs:
        candles = data.try_candles(tf)          # a timeframe the feed cannot supply yet (monthly on an un-updated EA) is skipped
        if not candles:
            continue
        atr = _atr14(candles)
        if atr is None:
            continue
        reach = atr * atr_mult
        for lv in intact_closes(candles, price):
            dist = lv["price"] - price
            if abs(dist) <= reach:
                shared = [{"tf": o, "polarity": every[o][round(lv["price"], 2)]["polarity"], "time": every[o][round(lv["price"], 2)]["time"]}
                          for o in LEVEL_TFS if o != tf and o in every and round(lv["price"], 2) in every[o]]
                out.append({"tf": tf, "price": lv["price"], "time": lv["time"], "dist": round(dist, 2), "polarity": lv["polarity"], "also": shared})
    out += [c for c in custom_nearby(price, atr_mult, tfs) if c["tf"] in tfs]     # levels the user marked are always listed, near or far
    return sorted(out, key=lambda l: abs(l["dist"]))


# ---- custom levels: a level the USER says exists (e.g. a close on another provider's chart) ------------------------------------
# It acts exactly like an engine level of its timeframe ("1w" / "1d" / "4h"): price can hit it, a CPLOC can cross it, a setup can
# start from it and it can be a target. The user states what it is for (buy = support, sell = resistance). It exists from `since`
# and is broken once a completed candle of its timeframe closes beyond it, the same rule as for engine levels.
CUSTOM_TFS = ("1w", "1d", "4h")
_CUSTOM_CACHE: dict = {"at": 0.0, "rows": []}


def custom_levels(tf: str | None = None) -> list[dict]:
    """Stored custom levels: [{id, tf, price, polarity, since, label}] (cached for a couple of seconds: the engine asks very often)."""
    import time as _t
    if _t.time() - _CUSTOM_CACHE["at"] > 2:
        from . import store
        _CUSTOM_CACHE["rows"] = [
            {"id": r["id"], "tf": r["tf"], "price": float(r["price"]), "polarity": r["direction"], "since": int(r["time"] or 0), "label": r.get("label") or ""}
            for r in store.list_custom_levels() if r["tf"] in CUSTOM_TFS and r["direction"] in ("buy", "sell")]
        _CUSTOM_CACHE["at"] = _t.time()
    return [c for c in _CUSTOM_CACHE["rows"] if tf is None or c["tf"] == tf]


def forget_custom_cache() -> None:
    _CUSTOM_CACHE["at"] = 0.0


def custom_intact(cl: dict, candles: list[dict], idx: int, ref: float | None, time: int) -> bool:
    """A custom level is valid from `since` until the user deletes it. The engine's break rule does NOT apply: the user marks levels from
    other liquidity providers' charts, where the closes the MT5 feed shows may not have happened. (The arguments are kept so it can be
    called like the engine's own level checks.)"""
    return cl["since"] <= time


def custom_status(cl: dict, price: float) -> dict:
    """Custom levels stay valid until deleted. `closed_beyond` tells the user when this feed's candles closed beyond it (informational only)."""
    candles = data.try_candles(cl["tf"])
    out = {"intact": True, "sticky": True}
    if candles:
        lv = cl["price"]
        for c in candles[:-1]:
            if c["time"] > cl["since"] and ((lv > price and c["close"] > lv) or (lv < price and c["close"] < lv)):
                out["closed_beyond"] = f"on this feed the {cl['tf']} candle of {_fmt(c['time'])} closed at {c['close']}, {'above' if c['close'] > lv else 'below'} it; the engine still uses the level because you marked it"
                break
    return out


def _fmt(t: int) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(t, _dt.UTC).strftime("%m-%d %H:%M")


def custom_sides(cl: dict, candles: list[dict], idx: int, time: int) -> tuple[bool, bool]:
    """(valid as a resistance, valid as a support) as of `time`: both, from `since` on (custom levels are never broken by closes)."""
    ok = cl["since"] <= time
    return ok, ok


def custom_nearby(price: float, atr_mult: float = 0.5, tfs: tuple[str, ...] = CUSTOM_TFS) -> list[dict]:
    """Custom levels still intact now, in the same shape as engine levels (all of them, near or far: there are few)."""
    out = []
    for tf in tfs:
        candles = data.try_candles(tf)
        if not candles:
            continue
        now = candles[-1]["time"]
        for cl in custom_levels(tf):
            if custom_intact(cl, candles, len(candles) - 1, price, now + TF_SEC_ALL[tf]):
                out.append({"tf": tf, "price": cl["price"], "time": cl["since"] - TF_SEC_ALL[tf],      # so that the chart line starts at `since`
                            "dist": round(cl["price"] - price, 2), "polarity": cl["polarity"], "custom": True, "label": cl["label"], "id": cl["id"]})
    return out


TF_SEC_ALL = {"4h": 14400, "1d": 86400, "1w": 604800}


LEVEL_TF_SEC = {"4h": 14400, "1d": 86400, "1w": 604800, "5m": 300}


def next_boundary(tf: str, running_open: int) -> int:
    """When the next candle of `tf` opens, given the running candle's open time (months are not a fixed length)."""
    if tf != "1mo":
        return running_open + LEVEL_TF_SEC[tf]
    import datetime as dt
    d = dt.datetime.fromtimestamp(running_open, dt.UTC)
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return int(d.replace(year=y, month=m, day=1).timestamp())


def upcoming(tf: str, price: float) -> dict:
    """What changes at the next rollover of `tf` (the running candle closes and a new one opens).

    The close of the candle that is 'just before the running one' becomes a valid level when the new candle starts,
    and the candle that is closing now becomes the excluded one. Assumes the running candle closes at `price`
    and the next candle opens on schedule (weekends are not skipped).
    """
    candles = data.get_candles(tf)
    boundary = next_boundary(tf, candles[-1]["time"])
    now = intact_closes(candles, price)
    hypo = candles + [{"time": boundary, "open": price, "high": price, "low": price, "close": price}]
    after = intact_closes(hypo, price)
    now_t, after_t = {l["time"] for l in now}, {l["time"] for l in after}
    return {
        "rolls_at": boundary,
        "becomes_valid": sorted([l for l in after if l["time"] not in now_t], key=lambda l: abs(l["price"] - price)),
        "stops_being_valid": sorted([l for l in now if l["time"] not in after_t], key=lambda l: abs(l["price"] - price)),
    }
