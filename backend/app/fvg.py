"""Fair value gaps (FVG), exactly as the user's indicator.pine defines them, on the 1H, 4H and Daily timeframes.

Creation (three completed candles, the gap is between the first and the third):
    bullish:  low[0]  > high[2]   zone = [high[2], low[0]]
    bearish:  high[0] < low[2]    zone = [high[0], low[2]]
    It is known once the third candle has closed.
Consumed: after creation, a candle that touches the zone marks it "tapped"; once tapped, a bullish zone is consumed when a candle's
    low reaches its bottom, a bearish zone when a candle's high reaches its top (the zone has been filled right through).
"Caused a BOS": when a break of structure (the indicator's swing-5 BOS, patterns.find_bos) happens in the zone's direction while the
    zone is active, the zone is flagged. The indicator only trusts flagged zones for its FRC signals.

An FVG is a CONTINUATION zone: a bearish FVG means the trend is down and a rally into it is expected to be rejected; a bullish FVG
the opposite.
"""
from bisect import bisect_left, bisect_right

from . import data, patterns

TFS = ("1h", "4h", "1d")
SEC = {"1h": 3600, "4h": 14400, "1d": 86400}
_CACHE: dict = {}


def zones(tf: str) -> list[dict]:
    """Every FVG of `tf`, oldest first: {tf, bull, top, bottom, created, end, bos, formed}. Times are when each became known / was consumed /
    flagged (the close of the candle that did it). `end` and `bos` are None while not happened."""
    candles = data.try_candles(tf)
    if not candles or len(candles) < 5:
        return []
    key = (tf, len(candles), candles[-2]["time"])
    hit = _CACHE.get(tf)
    if hit and hit[0] == key:
        return hit[1]
    n = len(candles) - 1                                   # completed candles only (the last one is still running)
    out = []
    for k in range(2, n):
        a, b = candles[k - 2], candles[k]
        if b["low"] > a["high"]:
            z = {"bull": True, "top": b["low"], "bottom": a["high"]}
        elif b["high"] < a["low"]:
            z = {"bull": False, "top": a["low"], "bottom": b["high"]}
        else:
            continue
        z.update(tf=tf, created=b["time"] + SEC[tf], formed=candles[k - 1]["time"], end=None, bos=None, viol=[])
        tapped = False
        cur = z["top"] if z["bull"] else z["bottom"]          # the edge of the part of the gap price has NOT traded through yet
        for j in range(k + 1, n):
            x = candles[j]
            if not z["bull"] and x["high"] > cur:                # price went into the gap: that part is gone, what is left starts above the highest point reached
                cur = min(x["high"], z["top"])
                z["viol"].append((x["time"] + SEC[tf], cur))
            elif z["bull"] and x["low"] < cur:
                cur = max(x["low"], z["bottom"])
                z["viol"].append((x["time"] + SEC[tf], cur))
            if x["high"] >= z["bottom"] and x["low"] <= z["top"]:
                tapped = True
            # A gap is USED UP when a candle CLOSES through it (above a bearish gap's top, below a bullish gap's bottom). A wick that runs through the whole gap and
            # comes back to close outside does not use it up: that candle is the rejection.
            if (z["bull"] and x["close"] <= z["bottom"]) or (not z["bull"] and x["close"] >= z["top"]):
                z["end"] = x["time"] + SEC[tf]
                break
        out.append(z)
    for ev in patterns.find_bos(candles):
        tb = ev["time"] + SEC[tf]
        want = ev["direction"] == "bull"
        for z in out:
            if z["created"] > tb:
                break
            if z["bos"] is None and z["bull"] == want and (z["end"] is None or z["end"] > tb):
                z["bos"] = tb
    _CACHE[tf] = (key, out)
    return out


def bounds_at(z: dict, t: int) -> tuple[float, float]:
    """(bottom, top) of the part of the gap that is still UNFILLED at time t. Price that has traded into a gap has used that part up: for a bearish gap what is
    left starts at the highest point price reached inside it, for a bullish gap it ends at the lowest point reached."""
    bottom, top = z["bottom"], z["top"]
    for known, edge in z.get("viol", ()):
        if known > t:
            break
        if z["bull"]:
            top = edge
        else:
            bottom = edge
    return bottom, top


def active_at(t: int, tfs: tuple[str, ...] = TFS) -> list[dict]:
    """Zones that exist and are not consumed at time t."""
    return [z for tf in tfs for z in zones(tf) if z["created"] <= t and (z["end"] is None or z["end"] > t)]


def flagged_at(z: dict, t: int) -> bool:
    return z["bos"] is not None and z["bos"] <= t


def ahead(t: int, bull: bool, lo: float, hi: float) -> list[dict]:
    """Opposing zones (bearish for a buy, bullish for a sell) active at t that lie between `lo` and `hi` (the entry side and the target),
    nearest to the trade first. Each carries `edge` (the near edge: bottom for a buy, top for a sell) and `flagged`."""
    out = []
    for z in active_at(t):
        if z["bull"] == bull:
            continue
        zb, zt = bounds_at(z, t)                              # only the part price has not traded through yet
        if zt >= lo and zb <= hi:
            edge = zb if bull else zt
            if (edge > lo) if bull else (edge < hi):         # its near edge is still AHEAD of the trade (not already behind price)
                out.append({**z, "bottom": zb, "top": zt, "edge": edge, "flagged": flagged_at(z, t)})
    out.sort(key=lambda z: z["edge"] if bull else -z["edge"])
    return out


def as_chart_items(max_per_tf: int = 40) -> list[dict]:
    """Unconsumed zones for the chart (the newest few of each timeframe)."""
    items = []
    for tf in TFS:
        live = [z for z in zones(tf) if z["end"] is None][-max_per_tf:]
        for z in live:
            zb, zt = bounds_at(z, 10 ** 12)
            items.append({"tf": tf, "bull": z["bull"], "top": zt, "bottom": zb, "from": z["formed"],
                          "flagged": z["bos"] is not None})
    return items


_LEGS: dict = {}


def find_rejection(z: dict, candles: list[dict], k: int, last: int) -> int | None:
    """Does candle k touch the part of FVG `z` that is still unfilled, and is there a trigger? The trigger is the first candle (k itself, or one of the next 3) that
    has the trend's colour (red under a bearish gap, green over a bullish one) AND closes beyond what is left of the gap. The tap and the trigger can be the
    same candle or two: a candle can tap the gap and close inside it, and the next one close beyond it. It fails if a candle closes through the gap's far
    side, or closes back into the unfilled part after the tap. Returns the trigger's index."""
    bull = z["bull"]
    c = candles[k]
    zb, zt = bounds_at(z, c["time"])          # the unfilled part as it stood when the tap candle opened: its own wick does not count against it
    if zt - zb <= 0.05 or not ((c["low"] <= zt) if bull else (c["high"] >= zb)):
        return None                                               # nothing of the gap was left, or this candle does not reach what was left
    for j in range(k, min(k + 4, last)):
        x = candles[j]
        if z["end"] is not None and x["time"] >= z["end"]:
            return None                                           # a candle closed through the gap before this one: it is used up
        if (x["close"] <= z["bottom"]) if bull else (x["close"] >= z["top"]):
            return None                                           # closed through the gap
        coloured = (x["close"] > x["open"]) if bull else (x["close"] < x["open"])
        beyond = (x["close"] > zt) if bull else (x["close"] < zb)
        if coloured and beyond:
            return j
        if j > k and ((x["close"] <= zt) if bull else (x["close"] >= zb)):
            return None                                           # back into the unfilled part: no continuation
    return None


_BOS: dict = {}


def trend_start(z: dict, tap_time: int) -> int | None:
    """An FVG is a continuation: price was trending in the gap's direction and pulls back into it. On the gap's OWN timeframe, the latest break of structure
    (swing-5, as the indicator) completed before the tap must be in the gap's direction. Returns the opening time of that BOS candle (the trend move the
    pullback came from starts there), or None when the last BOS was the other way (the trend has turned: no setup)."""
    tf = z["tf"]
    candles = data.try_candles(tf)
    if not candles:
        return None
    key = (len(candles), candles[-2]["time"])
    hit = _BOS.get(tf)
    if not hit or hit[0] != key:
        ev = patterns.find_bos(candles)
        hit = _BOS[tf] = (key, ev, [e["time"] for e in ev])
    _, ev, times = hit
    i = bisect_right(times, tap_time - SEC[tf]) - 1                 # BOS candles that had closed by the time the tap candle opened
    if i < 0 or (ev[i]["direction"] == "bull") != z["bull"]:
        return None
    return ev[i]["time"]


def _tap_pending(z: dict, d: list[dict], p: int, ci: int) -> bool:
    """True when candle p did NOT tap what is left of FVG `z`, or the candles p..ci-1 already settled it (a trigger, a close through, a close back in);
    False when p tapped it and everything since is still waiting for a trigger (the candle at ci can still be it)."""
    bull = z["bull"]
    zb, zt = bounds_at(z, d[p]["time"])
    if zt - zb <= 0.05 or not ((d[p]["low"] <= zt) if bull else (d[p]["high"] >= zb)):
        return True
    for j in range(p, ci):
        x = d[j]
        if z["end"] is not None and x["time"] >= z["end"]:
            return True
        if (x["close"] <= z["bottom"]) if bull else (x["close"] >= z["top"]):
            return True
        if ((x["close"] > x["open"]) if bull else (x["close"] < x["open"])) and ((x["close"] > zt) if bull else (x["close"] < zb)):
            return True
        if j > p and ((x["close"] <= zt) if bull else (x["close"] >= zb)):
            return True
    return False


def legs(tf: str = "1d") -> list[dict]:
    """Trend legs signalled by the FVGs of one timeframe (1H, 4H or Daily). A candle of that timeframe taps into an unfilled FVG of that timeframe and closes
    outside it (below a bearish one, above a bullish one); the first candle of the trend's colour (red for a sell, green for a buy) from that tap on is the
    trigger. Only when the latest BOS of that timeframe before the tap is in the gap's direction (see trend_start). The leg then heads for the lowest low /
    highest high from that BOS up to the tap (the peak the pullback came from). Was: the lowest low / highest high of
    the move from the FVG's first candle up to the trigger. Its stop is the tap candle's high / low. It is alive until price reaches that target or a candle
    of that timeframe closes beyond the stop.
    Returns [{bull, start, stop, target, zone, tap_time, trigger_time, tf}] where `start` is when the trigger candle closed."""
    candles = data.try_candles(tf)
    if not candles or len(candles) < 6:
        return []
    key = (len(candles), candles[-2]["time"])
    if _LEGS.get(tf, {}).get("key") == key:
        return _LEGS[tf]["legs"]
    sec, n = SEC[tf], len(candles) - 1
    times = [c["time"] for c in candles]
    out = []
    for z in zones(tf):
        bull = z["bull"]
        k0 = bisect_left(times, z["formed"]) - 1               # the FVG's first candle
        if k0 < 0:
            continue
        used = -1                                               # candles up to this index belong to a rejection already handled
        for k in range(k0 + 3, n):
            c = candles[k]
            if z["end"] is not None and c["time"] >= z["end"]:
                break
            if k <= used:
                continue
            g = find_rejection(z, candles, k, n)
            if g is None:
                continue                                        # no rejection here; the unfilled part shrinks by the wick and later taps still count
            used = g
            span = candles[k:g + 1]
            stop = min(x["low"] for x in span) if bull else max(x["high"] for x in span)     # the extreme of the tap (and trigger) candle(s)
            b = trend_start(z, c["time"])
            seg = candles[bisect_left(times, b):k] if b is not None else []
            if not seg:
                continue                                        # the trend on this timeframe is not the gap's way: no leg
            target = max(x["high"] for x in seg) if bull else min(x["low"] for x in seg)      # the peak the pullback into the gap came from
            if (target > candles[g]["close"]) if bull else (target < candles[g]["close"]):
                zb, zt = bounds_at(z, c["time"])
                out.append({"bull": bull, "start": candles[g]["time"] + sec, "stop": stop, "target": target, "tf": tf,
                            "zone": {"top": zt, "bottom": zb, "formed": z["formed"]}, "tap_time": c["time"], "trigger_time": candles[g]["time"]})
    # When does each leg end? Worked out once here: the end of the first 1H candle that reached its target, or of the first candle of this timeframe that
    # closed beyond its stop. (alive_leg used to rescan every 1H candle for every setup, which took minutes.)
    h1 = data.get_candles("1h")
    h1_times = [x["time"] for x in h1]
    inf = float("inf")
    for lg in out:
        reached = inf
        for x in h1[bisect_left(h1_times, lg["start"]):]:
            if (x["high"] >= lg["target"]) if lg["bull"] else (x["low"] <= lg["target"]):
                reached = x["time"] + 3600
                break
        broken = inf
        for x in candles[bisect_left(times, lg["start"]):]:
            if (x["close"] < lg["stop"]) if lg["bull"] else (x["close"] > lg["stop"]):
                broken = x["time"] + sec
                break
        lg["ends"] = min(reached, broken)
    _LEGS[tf] = {"key": key, "legs": out}
    return out


def daily_legs() -> list[dict]:
    return legs("1d")


def alive_leg(t: int, bull: bool, price: float, tf: str = "1d") -> dict | None:
    """The `tf` FVG leg in this direction that is still running at time t (its target not reached, its stop not closed beyond) and whose target lies beyond
    `price`. If several, the nearest target."""
    best = None
    for lg in legs(tf):
        if lg["bull"] != bull or lg["start"] > t or t >= lg["ends"]:
            continue
        if (lg["target"] <= price) if bull else (lg["target"] >= price):
            continue
        if best is None or abs(lg["target"] - price) < abs(best["target"] - price):
            best = lg
    return best


_H1T: dict = {"key": None, "times": []}


def _h1_times(h1: list[dict]) -> list[int]:
    key = (len(h1), h1[-1]["time"])
    if _H1T["key"] != key:
        _H1T.update(key=key, times=[x["time"] for x in h1])
    return _H1T["times"]


def opposing_signal(t: int, bull: bool, price: float, above_sec: int) -> bool:
    """Is an FVG signal on a HIGHER timeframe (longer than `above_sec` seconds) against a trade in this direction in play at time t?
    alive:   an FVG leg of that timeframe the other way that has not reached its target or been stopped (see alive_leg).
    forming: the running 4H / Daily candle (judged from the 1H candles up to `t`, nothing later) has tapped into an unfilled FVG of its own timeframe, the
             other way, price is now outside it on the trend side, and the candle so far has the trend's colour. It has not closed yet, so the leg is not
             'alive' in the strict sense, but the signal is printing."""
    for tf in TFS:
        if SEC[tf] <= above_sec:
            continue
        if alive_leg(t, not bull, price, tf):
            return True
        if tf == "1h":
            continue
        d = data.try_candles(tf)
        if not d:
            continue
        cur = next((c for c in reversed(d) if c["time"] <= t), None)
        if cur is None:
            continue
        h1 = data.get_candles("1h")
        h1_times = _h1_times(h1)
        part = h1[bisect_left(h1_times, cur["time"]):bisect_right(h1_times, t - 3600)]       # the 1H candles of the running candle that had closed by t
        if not part:
            continue
        opened, hi, lo, now = part[0]["open"], max(x["high"] for x in part), min(x["low"] for x in part), part[-1]["close"]
        for z in zones(tf):
            if z["bull"] == bull or z["created"] > cur["time"] or (z["end"] is not None and z["end"] <= cur["time"]):
                continue
            zb, zt = bounds_at(z, cur["time"])
            if not z["bull"] and hi >= zb and now < zb and now < opened:
                return True
            if z["bull"] and lo <= zt and now > zt and now > opened:
                return True
        ci = bisect_left([c["time"] for c in d], cur["time"])
        for z in zones(tf):                                  # tapped on one of the 3 candles BEFORE the running one, no trigger yet: the running candle may be it
            if z["bull"] == bull:
                continue
            for p in range(max(0, ci - 3), ci):
                if z["created"] > d[p]["time"] or _tap_pending(z, d, p, ci):
                    continue
                zb, zt = bounds_at(z, d[p]["time"])
                if (not z["bull"] and now < zb and now < opened) or (z["bull"] and now > zt and now > opened):
                    return True
    return False
