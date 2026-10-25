"""Candle pattern identification.

FRC (also called CPLOC), as defined in the indicator:
- Bullish FRC: low <= min(prev open, prev close)  AND  close > max(prev open, prev close)
- Bearish FRC: high >= max(prev open, prev close) AND  close < min(prev open, prev close)
i.e. the candle reaches back to (or through) the previous candle's body on one side and closes
fully beyond the other side of that body. Only the previous candle's BODY matters, not its wicks.

Only completed candles are evaluated: a running candle can satisfy the shape and then lose it
before it closes, so the last candle is never reported.
"""


def find_frc(candles: list[dict], hide_repeats: bool = False, qualify=None) -> list[dict]:
    """Return [{"time", "direction": "bull"|"bear"}] for completed candles, oldest first.

    hide_repeats mirrors the indicator's suppression: after a bullish FRC, further bullish FRCs
    are hidden until a down candle closes (and the same for bearish after an up candle).

    qualify(index, "bull"|"bear") may add a condition: return a falsy value to reject the candle,
    or a list (e.g. the levels it crosses) that is attached to the result as "levels".
    Qualification happens before the repeat suppression, as in the indicator.
    """
    out = []
    bull_armed = bear_armed = True
    for i in range(1, len(candles) - 1):  # skip index 0 (no previous) and the running candle
        c, p = candles[i], candles[i - 1]
        body_hi, body_lo = max(p["open"], p["close"]), min(p["open"], p["close"])

        bull = c["low"] <= body_lo and c["close"] > body_hi
        bear = c["high"] >= body_hi and c["close"] < body_lo

        bull_extra = bear_extra = None
        if qualify and bull:
            bull_extra = qualify(i, "bull")
            bull = bool(bull_extra)
        if qualify and bear:
            bear_extra = qualify(i, "bear")
            bear = bool(bear_extra)

        if hide_repeats:
            if c["close"] > c["open"]:
                bear_armed = True
            if c["close"] < c["open"]:
                bull_armed = True
            bull, bear = bull and bull_armed, bear and bear_armed
            if bull:
                bull_armed = False
            if bear:
                bear_armed = False

        if bull:
            out.append({"time": c["time"], "direction": "bull", **({"levels": bull_extra} if bull_extra else {})})
        if bear:
            out.append({"time": c["time"], "direction": "bear", **({"levels": bear_extra} if bear_extra else {})})
    return out


def find_bos(candles: list[dict], swing: int = 5) -> list[dict]:
    """Break of Structure, as in the indicator (swing length `swing`, default 5).

    A swing high is a candle whose high is above the `swing` highs before it and at least the `swing`
    highs after it (confirmed `swing` candles later); swing lows mirror that. After each new swing,
    the first candle CLOSING beyond the latest swing high (bullish) / swing low (bearish) is a BOS;
    each swing can be broken once. Completed candles only. Returns [{"time", "direction"}].
    """
    out = []
    last_ph = last_pl = None
    up_done = down_done = False
    for j in range(len(candles) - 1):  # skip the running candle
        p = j - swing
        if p - swing >= 0:
            hi, lo = candles[p]["high"], candles[p]["low"]
            if all(hi > candles[p - k]["high"] for k in range(1, swing + 1)) and all(
                hi >= candles[p + k]["high"] for k in range(1, swing + 1)
            ):
                last_ph, up_done = hi, False
            if all(lo < candles[p - k]["low"] for k in range(1, swing + 1)) and all(
                lo <= candles[p + k]["low"] for k in range(1, swing + 1)
            ):
                last_pl, down_done = lo, False
        close = candles[j]["close"]
        if last_ph is not None and not up_done and close > last_ph:
            out.append({"time": candles[j]["time"], "direction": "bull"})
            up_done = True
        if last_pl is not None and not down_done and close < last_pl:
            out.append({"time": candles[j]["time"], "direction": "bear"})
            down_done = True
    return out
