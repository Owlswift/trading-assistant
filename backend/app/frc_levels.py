"""FRC (CPLOC) candles that cross a valid higher-timeframe level.

Timeframe relationships:
    1H  FRC -> Daily level
    4H  FRC -> Daily level
    15M FRC -> 4H level

A candle qualifies only if it CROSSES a valid level of the paired timeframe:
    bullish FRC: low <= level < close   (reaches the level, closes above it)
    bearish FRC: high >= level > close  (reaches the level, closes below it)

Levels are judged as of the FRC candle's own time (no hindsight):
- the HTF candle containing the FRC candle is the "running" one; its close and the close of the
  completed candle just before it are not levels;
- a level is broken if any completed HTF candle after it has closed beyond it on the side away
  from price (price = the FRC candle's open). Broken levels never qualify a candle.
"""
from bisect import bisect_left, bisect_right

from . import data, levels as levelsmod, patterns

LTF_TO_LEVEL_TF = {"1h": "1d", "4h": "1d", "15m": "4h"}


def valid_levels(htf_closes: list[float], idx: int, price: float) -> list[int]:
    """Indices k of HTF closes that are valid levels while HTF candle `idx` is running.

    Only completed candles (0..idx-1) are used. k <= idx-2: neither the running candle nor the
    completed one just before it is a level. A level is broken once a later completed candle has
    closed beyond it on the side away from `price`.
    """
    n = idx
    if n < 2:
        return []
    later_max = [float("-inf")] * n
    later_min = [float("inf")] * n
    for k in range(n - 2, -1, -1):
        later_max[k] = max(later_max[k + 1], htf_closes[k + 1])
        later_min[k] = min(later_min[k + 1], htf_closes[k + 1])
    out = []
    for k in range(n - 1):
        lv = htf_closes[k]
        if lv > price and later_max[k] > lv:
            continue
        if lv < price and later_min[k] < lv:
            continue
        out.append(k)
    return out


class LevelIndex:
    """valid_levels() made fast: for every HTF candle that is running (idx) the valid levels are built once (sorted by price) and
    then queried by price range, instead of rescanning every close for each lower-timeframe candle. Same answers."""

    def __init__(self, closes: list[float]):
        self.closes = closes
        self._state: dict[int, tuple] = {}
        self._by_price: dict[float, list[int]] = {}
        for k, v in enumerate(closes):
            self._by_price.setdefault(v, []).append(k)

    def _build(self, idx: int):
        if idx not in self._state:
            n, cl = idx, self.closes
            up, dn = [], []
            hi, lo = float("-inf"), float("inf")
            for k in range(n - 2, -1, -1):          # hi / lo = the highest / lowest completed close after k
                hi, lo = max(hi, cl[k + 1]), min(lo, cl[k + 1])
                if hi <= cl[k]:
                    up.append((cl[k], k))
                if lo >= cl[k]:
                    dn.append((cl[k], k))
            up.sort(); dn.sort()
            self._state[idx] = (up, [p for p, _ in up], dn, [p for p, _ in dn])
        return self._state[idx]

    def valid(self, idx: int, price: float, lo: float = float("-inf"), hi: float = float("inf")) -> list[int]:
        """Indices k of the valid levels with lo <= level <= hi (same rule as valid_levels, as seen from `price`)."""
        if idx < 2:
            return []
        up, upp, dn, dnp = self._build(idx)
        out = {k for _, k in up[bisect_left(upp, max(lo, price)):bisect_right(upp, hi)]}                 # at or above price: valid if nothing closed above later
        out |= {k for _, k in dn[bisect_left(dnp, lo):bisect_right(dnp, min(hi, price))]}              # at or below price: valid if nothing closed below later
        if lo <= price <= hi:
            out |= {k for k in self._by_price.get(price, []) if k <= idx - 2}
        return sorted(out, key=lambda k: self.closes[k])


def supported(timeframe: str) -> bool:
    return timeframe in LTF_TO_LEVEL_TF


def find(timeframe: str, hide_repeats: bool = False, level_tf: str | None = None) -> list[dict]:
    level_tf = level_tf or LTF_TO_LEVEL_TF[timeframe]          # level_tf: a candle of `timeframe` crossing a level of that timeframe (4H on Weekly, Daily on Monthly)
    ltf = data.get_candles(timeframe)
    htf = data.get_candles(level_tf)
    htf_times = [c["time"] for c in htf]
    htf_closes = [c["close"] for c in htf]
    index = LevelIndex(htf_closes)

    def qualify(i: int, direction: str) -> list[dict]:
        c = ltf[i]
        idx = bisect_right(htf_times, c["time"]) - 1  # HTF candle running when this candle opened
        if idx < 2:
            return []
        found = []
        lo, hi = (c["low"], c["close"]) if direction == "bull" else (c["close"], c["high"])
        # A level is judged on ITS OWN timeframe: which side of it price is on = where the last COMPLETED higher-timeframe candle closed, not
        # where this 1H / 15M candle happened to open (a candle that dips below a Daily support and closes back above it opened below the level,
        # but no Daily candle has closed below it, so the support is intact).
        for k in index.valid(idx, htf_closes[idx - 1], lo, hi):
            lv = htf_closes[k]
            crosses = (c["low"] <= lv < c["close"]) if direction == "bull" else (c["high"] >= lv > c["close"])
            if crosses and levelsmod.origin_ok(htf[k + 1] if k + 1 < len(htf) else None, direction == "bull"):   # a buy needs a buying level (the candle after it went up), a sell a selling one
                found.append({"tf": level_tf, "price": lv, "time": htf_times[k]})
        for cl in levelsmod.custom_levels(level_tf):          # levels the user marked count like engine levels (and carry the polarity they gave)
            lv = cl["price"]
            crosses = (c["low"] <= lv < c["close"]) if direction == "bull" else (c["high"] >= lv > c["close"])
            if (crosses and cl["polarity"] == ("buy" if direction == "bull" else "sell")
                    and levelsmod.custom_intact(cl, htf, idx, c["open"], c["time"])):
                found.append({"tf": level_tf, "price": lv, "time": cl["since"], "custom": True, "label": cl["label"], "id": cl["id"]})
        return found

    return patterns.find_frc(ltf, hide_repeats, qualify)
