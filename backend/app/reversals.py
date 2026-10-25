"""Reversal setups. Price moves from level to level; each tier is the same idea one step up.

DAILY tier - targets the nearest valid DAILY level beyond the trigger candle's close, in its direction
(judged at the time of the trigger, no hindsight):

  reversal      1H CPLOC (FRC) crossing a valid Daily level. The main signal.
  early         15M CPLOC crossing a valid 4H level, AFTER price has hit a Daily level, where the 4H level
                is at that Daily level or beyond it. An early warning of the reversal the 1H CPLOC would
                show, and a more refined entry.
  continuation  15M CPLOC crossing a valid 4H level, in the direction of a 1H reversal that is still
                pending, after a 1H Break of Structure (indicator definition) in that direction: price is
                pulling back to retest before reaching its Daily target.

WEEKLY tier - targets the nearest valid WEEKLY level beyond the trigger candle's close:

  reversal      4H CPLOC crossing a valid Daily level, AFTER price has hit a Weekly level, where the Daily
                level is at that Weekly level or beyond it.
  early         the same, but with a 1H CPLOC.

"Hit" = a wick reaching the level; it must be the most recent valid higher-tier level price touched.
Bearish: price arrived at that level from below and the lower-tier level is at/above it; bullish: price
arrived from above and the lower-tier level is at/below it. "Beyond" is limited to EARLY_REACH_ATR x the
ATR(14) of the lower tier's level timeframe, so the signal stays near the higher-tier level.

HTF CONFLICT (a warning, never a discard): while a Weekly setup is pending, a Daily-tier setup in the OPPOSITE
direction may have a target that passes the Weekly trade's ENTRY (or its STOP). That is flagged in `htf_conflict`
but the setup stays valid and pending: levels keep forming, so its target usually moves to a nearer level before it
matters (see dynamic targets). Only the Weekly trade's entry and stop are used, not the Weekly level it started from.

DYNAMIC TARGETS: levels keep being created while a trade runs (every completed higher-timeframe candle
close can become a level). So the target is re-evaluated on every candle, using only the levels that exist
at that moment (no hindsight): the nearest valid level beyond the current price, never behind the trigger's
close, its origin level or the entry. A newly created level between price and the old target replaces it;
one beyond the target changes nothing; one behind price is ignored. A setup / plan completes when price
touches the target AS IT STOOD at that candle. `initial_target` is the target at trigger time and
`target_history` records every change.

STOP / TRADE PLAN (the trigger candle is the ENTRY candle, on its own timeframe):
- stop:   bullish: the lowest low of the trigger candle and the two candles before it; bearish: the highest
          high of those three. (The trigger candle's own extreme alone gets hunted, so the extreme that
          created it is used.)
- range:  the full high-low (wicks included) of the two candles before the trigger candle.
- entry:  if the trigger closes INSIDE that range, enter at its close. If it closes beyond it (a bearish
          one below, a bullish one above), wait for price to retrace back to the range of the second
          candle before the trigger (its near edge), and enter there.
- target: the setup's (dynamic) target: next Daily level, or next Weekly level for the Weekly tier.
- An unfilled retrace entry is "missed" if price reaches the target before it fills.
- After the fill, a candle that reaches both stop and target counts as stopped (pessimistic); the fill
  candle itself is only checked for the stop, since the order inside it is unknown.

Status, checked candle by candle on the trigger's own timeframe after the trigger candle (running candle
included):
- failed:    price breaks the STOP first (bull: low < stop, bear: high > stop)
- completed: price touches the target (a wick is enough)
- pending:   neither yet
If one candle does both it counts as failed (order inside a candle is unknown: pessimistic reading).

A same-tier, same-kind, same-direction trigger with the same target while a setup is still unresolved is a
repeat of that setup and is not reported again.
"""
import json
import os
import threading
import time
from bisect import bisect_left, bisect_right

from . import data, frc_levels, fvg, patterns, store
from . import levels as levelsmod

RULES_VERSION = 44       # bump when the rules change: the stored results of the old rules are archived and rebuilt (see store.ensure_rules_version)
MIN_RR = 1.0   # a plan paying less than this is shown but not taken (not flagged as a trade, not counted in the results)
EARLY_REACH_ATR = 1.0   # how far beyond the higher-tier level the lower-tier level may sit, in ATRs
HOLD_WINDOWS = 3        # the three-timeframe hold: price must have come down/up to the higher level within this many confirmation-level candles
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1mo": 2678400}
LEVEL_TOL = 0.05        # price tolerance when comparing closes from different timeframes


def _atr_before(candles: list[dict], idx: int, n: int = 14) -> float:
    """ATR over the n completed candles before candle `idx` (the running one)."""
    lo = max(1, idx - n)
    trs = []
    for k in range(lo, idx):
        c, p = candles[k], candles[k - 1]
        trs.append(max(c["high"] - c["low"], abs(c["high"] - p["close"]), abs(c["low"] - p["close"])))
    return sum(trs) / len(trs) if trs else 0.0


_OVERRIDES: dict[str, dict] = {}   # setup id -> {"ignore_until", "note"}; loaded at the start of find()


class LevelBook:
    """Valid levels of one higher timeframe as they stood at any moment (no hindsight).

    While HTF candle `idx` is running, the candidates are the closes of candles 0..idx-2 (the running candle and
    the completed one just before it are not levels). A level is a valid TARGET for a bullish trade (above price)
    if no completed HTF candle after it has closed above it; for a bearish trade (below price) if none has
    closed below it.
    """

    def __init__(self, candles: list[dict], tf: str | None = None):
        self.custom = levelsmod.custom_levels(tf) if tf else []     # levels the user marked on this timeframe
        self.cpol = {cl["price"]: cl for cl in self.custom}
        self.times = [c["time"] for c in candles]
        self.closes = [c["close"] for c in candles]
        self.candles = candles
        self.idx_of = {c["close"]: k for k, c in enumerate(candles)}   # the (latest) candle behind each close; the one AFTER it decides what the level is for
        self._cache: dict[int, tuple[list[float], list[float]]] = {}

    def _next_candle(self, price: float) -> dict | None:
        k = self.idx_of.get(price)
        return self.candles[k + 1] if k is not None and k + 1 < len(self.candles) else None

    def info(self, price: float) -> dict | None:
        """The level's candle, the candle that followed it, and what the level is for (buy / sell / both)."""
        if price in self.cpol:
            cl = self.cpol[price]
            return {"time": cl["since"], "open": None, "close": price, "next_open": None, "next_close": None,
                    "formation": "Marked by you" + (f" ({cl['label']})" if cl["label"] else ""), "polarity": cl["polarity"], "custom": True}
        k = self.idx_of.get(price)
        if k is None:
            return None
        c, nxt = self.candles[k], self._next_candle(price)
        return {"time": c["time"], "open": c["open"], "close": c["close"],
                "next_open": None if nxt is None else nxt["open"], "next_close": None if nxt is None else nxt["close"],
                "formation": levelsmod.formation(c, nxt), "polarity": levelsmod.polarity(nxt)}

    def custom_at(self, price: float) -> dict | None:
        return self.cpol.get(price)

    def origin_ok(self, price: float, bull: bool) -> bool:
        if price in self.cpol:      # a level the user marked: it is for what they said it is for
            return self.cpol[price]["polarity"] == ("buy" if bull else "sell")
        return True if price not in self.idx_of else levelsmod.origin_ok(self._next_candle(price), bull)

    def _valid(self, idx: int) -> tuple[list[float], list[float]]:
        if idx not in self._cache:
            n, cl = idx, self.closes
            up, down = [], []   # valid targets above price (bullish) / below price (bearish)
            hi = float("-inf")
            lo = float("inf")
            for k in range(n - 2, -1, -1):  # walk back so hi / lo are the later closes k+1..n-1
                hi = max(hi, cl[k + 1])
                lo = min(lo, cl[k + 1])
                if hi <= cl[k]:
                    up.append(cl[k])
                if lo >= cl[k]:
                    down.append(cl[k])
            self._cache[idx] = (sorted(set(up)), sorted(set(down)))
        return self._cache[idx]

    def valid_lists(self, time: int) -> tuple[list[float], list[float]]:
        """(up, down) valid level prices as of `time`: `up` = no later close above them (resistances), `down` = none below (supports)."""
        idx = bisect_right(self.times, time) - 1
        if idx < 2:
            return [], []
        up, down = self._valid(idx)
        if self.custom:
            up, down = list(up), list(down)
            for cl in self.custom:
                is_up, is_dn = levelsmod.custom_sides(cl, self.candles, idx, time)
                if is_up:
                    up.append(cl["price"])
                if is_dn:
                    down.append(cl["price"])
            up, down = sorted(set(up)), sorted(set(down))
        return up, down

    def next(self, time: int, ref: float, bull: bool) -> float | None:
        """Nearest valid level strictly beyond `ref`, as of `time`."""
        idx = bisect_right(self.times, time) - 1
        if idx < 2:
            return None
        up, down = self.valid_lists(time)
        # a target is a level where the OTHER side acts: a buy aims at a selling level, a sell at a buying level
        def ok(p):
            if p in self.cpol:
                return (not levelsmod.POLARITY_TARGET) or self.cpol[p]["polarity"] == ("sell" if bull else "buy")
            return (p not in self.idx_of) or levelsmod.target_ok(self._next_candle(p), bull)
        if bull:
            for k in range(bisect_right(up, ref), len(up)):
                if ok(up[k]):
                    return up[k]
            return None
        for k in range(bisect_left(down, ref) - 1, -1, -1):
            if ok(down[k]):
                return down[k]
        return None


FVG_EXIT = os.getenv("FVG_EXIT", "bb").strip().lower()    # all (every trade) | bb (only trades on a breaker block, running past candle 2's close) | off
_BB_CACHE: dict = {}
_H1_END: dict = {"key": None, "map": {}}


def _h1_by_end() -> dict:
    h = data.get_candles("1h")
    key = (len(h), h[-1]["time"])
    if _H1_END["key"] != key:
        _H1_END.update(key=key, map={c["time"] + 3600: c for c in h})
    return _H1_END["map"]


def _bb_closes(origin: float, bull: bool, t: int, tf: str = "1d") -> tuple[set[float], float | None]:
    """The closes of a BREAKER BLOCK formation on the `tf` chart (Daily, Weekly or Monthly), when the level `origin` is one.

    A BB level is the close of candle 1. For a sell: candle 1 is bullish, candle 2 is a bearish CPLOC that closes below candle 1's body (it breaks the
    level), and candle 3 retests the level (its high reaches it) and closes back below it, leaving a wick. A buy is the mirror image. A setup taken on
    that formation must not target candle 2's close (or any other close inside the formation): price is expected to run through them to the next level
    beyond. Half the trade is closed at candle 2's close and the rest runs on. Judged as of `t` (when the trigger candle closed): no later knowledge.
    Returns (the closes of candle 1, 2, 3 and the candles between, candle 2's close), or (empty set, None) when `origin` is not a BB level."""
    d = data.try_candles(tf)
    if not d:
        return set(), None
    key = (tf, len(d), d[-1]["time"], origin, bull, t)
    if key in _BB_CACHE:
        return _BB_CACHE[key]

    def end_of(k: int) -> int:
        return d[k + 1]["time"] if k + 1 < len(d) else d[k]["time"] + TF_SECONDS[tf]

    out: tuple[set[float], float | None] = (set(), None)
    a = next((k for k in range(len(d) - 2, -1, -1) if abs(d[k]["close"] - origin) < 0.006 and end_of(k) <= t), None)
    if a is not None and a + 1 < len(d) and end_of(a + 1) <= t:
        c1, c2 = d[a], d[a + 1]
        top, bot = max(c1["open"], c1["close"]), min(c1["open"], c1["close"])
        if bull:      # a buy: candle 1 bearish, candle 2 a bullish CPLOC closing above its body
            shape = c1["dir"] == "bear" and c2["dir"] == "bull" and c2["low"] <= bot and c2["close"] > top
        else:
            shape = c1["dir"] == "bull" and c2["dir"] == "bear" and c2["high"] >= top and c2["close"] < bot
        if shape:
            closes, retest, broken = {c1["close"], c2["close"]}, False, False
            for k in range(a + 2, len(d)):
                c = d[k]
                if end_of(k) <= t:                                  # a completed candle after candle 2
                    if (c["close"] <= origin) if bull else (c["close"] >= origin):
                        broken = True
                        break
                    retest = retest or ((c["low"] <= origin) if bull else (c["high"] >= origin))
                    closes.add(c["close"])
                elif c["time"] <= t:                                # the candle the trigger is in: only what had happened by then
                    part = [x for x in data.get_candles("1h") if c["time"] <= x["time"] and x["time"] + 3600 <= t]
                    if part:
                        now = part[-1]["close"]
                        if (now <= origin) if bull else (now >= origin):
                            broken = True
                        retest = retest or ((min(x["low"] for x in part) <= origin) if bull else (max(x["high"] for x in part) >= origin))
                    break
            if retest and not broken:
                out = ({round(x, 2) for x in closes}, round(c2["close"], 2))
    if len(_BB_CACHE) > 20000:
        _BB_CACHE.clear()
    _BB_CACHE[key] = out
    return out


class _SkipBook:
    """A level book that never offers the given prices as a target (the closes of a breaker block formation)."""

    def __init__(self, base, skip: set[float]):
        self.base, self.skip = base, skip

    def __getattr__(self, name):
        return getattr(self.base, name)

    def next(self, time: int, ref: float, bull: bool) -> float | None:
        for _ in range(12):
            nxt = self.base.next(time, ref, bull)
            if nxt is None or round(nxt, 2) not in self.skip:
                return nxt
            ref = nxt
        return None


class _CappedBook:
    """A level book whose targets are also capped at the near edge of opposing FVGs (that caused a BOS) while those FVGs are unfilled:
    price is expected to reject there. Once an FVG is consumed (price filled it right through) its cap is gone."""

    def __init__(self, base: "LevelBook", caps: list[tuple[float, int | None]]):
        self.base, self.caps = base, caps

    def __getattr__(self, name):
        return getattr(self.base, name)

    def next(self, time: int, ref: float, bull: bool) -> float | None:
        best = self.base.next(time, ref, bull)
        for edge, end in self.caps:
            if (end is None or end > time) and ((edge > ref) if bull else (edge < ref)):
                if best is None or ((edge < best) if bull else (edge > best)):
                    best = edge
        return best


class _FixedBook:
    """For FVG continuation trades the target is a price extreme (the last swing high / low), not a level: it does not move."""

    def __init__(self, base: "LevelBook", target: float):
        self.base, self.target = base, target

    def __getattr__(self, name):
        return getattr(self.base, name)

    def next(self, time: int, ref: float, bull: bool) -> float | None:
        return self.target if ((self.target > ref) if bull else (self.target < ref)) else None


class _LazyPath:
    """The target in force during each candle after the trigger (index k -> path[k - i - 1]), worked out only as far as it is read:
    a setup that ended after a few candles no longer pays for the thousands of candles after it.

    For candle k it is the nearest valid level (as of that candle) beyond the price at the start of the candle, but never behind
    `floor` (the trigger close / origin / entry). Falls back to the previous target when no level qualifies."""

    def __init__(self, candles: list[dict], i: int, bull: bool, floor: float, target0: float, book: "LevelBook"):
        self.candles, self.i, self.bull, self.floor, self.book = candles, i, bull, floor, book
        self._vals: list[float] = []
        self._tgt = target0

    def _step(self) -> bool:
        k = self.i + 1 + len(self._vals)
        if k >= len(self.candles):
            return False
        prev = self.candles[k - 1]["close"]
        ref = max(prev, self.floor) if self.bull else min(prev, self.floor)
        new = self.book.next(self.candles[k]["time"], ref, self.bull)
        if new is not None:
            self._tgt = new
        self._vals.append(self._tgt)
        return True

    def _all(self) -> list[float]:
        while self._step():
            pass
        return self._vals

    def __iter__(self):
        n = 0
        while True:
            if n >= len(self._vals) and not self._step():
                return
            yield self._vals[n]
            n += 1

    def __len__(self) -> int:
        return len(self._all())

    def __bool__(self) -> bool:
        return self.i + 1 < len(self.candles)

    def __getitem__(self, x):
        return self._all()[x]


def _target_path(candles: list[dict], i: int, bull: bool, floor: float, target0: float, book: "LevelBook") -> _LazyPath:
    return _LazyPath(candles, i, bull, floor, target0, book)


def _history(candles: list[dict], i: int, target0: float, path: list[float], end_time: int | None) -> list[dict]:
    """Target changes: [{"time", "target"}], starting with the target at the trigger."""
    hist = [{"time": candles[i]["time"], "target": target0}]
    for k, t in enumerate(path, start=i + 1):
        if end_time is not None and candles[k]["time"] > end_time:
            break
        if t != hist[-1]["target"]:
            hist.append({"time": candles[k]["time"], "target": t})
    return hist


TF_SEC = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


def _resolve(candles: list[dict], i: int, bull: bool, stop: float, path: list[float],
             ignore_until: int = 0, close_stop: bool = False) -> tuple[str, int | None, float | None]:
    """(status, resolved_time, the target in force when it resolved). The target is None while still pending.
    Candles up to `ignore_until` are skipped (the user flagged them as a feed glitch)."""
    for k, tgt in enumerate(path, start=i + 1):
        later = candles[k]
        if later["time"] <= ignore_until:
            continue
        if close_stop:      # CPLOC signals: only a CLOSE beyond the stop breaks the setup; a wick through it that fails to close beyond it does not
            broke = later["close"] < stop if bull else later["close"] > stop
        else:
            broke = later["low"] < stop if bull else later["high"] > stop
        hit = later["high"] >= tgt if bull else later["low"] <= tgt
        if broke:
            return "failed", later["time"], tgt   # the target that was still in force when the stop broke
        if hit:
            return "completed", later["time"], tgt
    return "pending", None, None


def _dup(setups: list[dict], built: dict) -> bool:
    """The same trade again: a setup in the same direction, with the same FINAL target (after every adjustment), whose earlier twin is still running when this
    one triggers. _is_repeat is checked before a setup is built, with the target as first worked out, which the build often changes (a breaker block's skip,
    an FVG cap, a Daily FVG leg), so it missed these: the 21:00 Daily hold on Aug 13 was the 19:00 one again."""
    return any(s["direction"] == built["direction"] and s["initial_target"] == built["initial_target"] and s["tier"] == built["tier"] and s["kind"] == built["kind"]
               and s["trigger_tf"] == built["trigger_tf"] and (s["resolved_time"] is None or s["resolved_time"] > built["time"]) for s in setups)


def _is_repeat(setups: list[dict], direction: str, target: float, time: int) -> bool:
    return any(
        s["direction"] == direction and s["initial_target"] == target
        and (s["resolved_time"] is None or s["resolved_time"] > time)
        for s in setups
    )


def _plan(candles: list[dict], i: int, bull: bool, target0: float, stop: float, floor: float, tf_sec: int,
          book: LevelBook, entry_override: float | None = None, ignore_until: int = 0, partials: list[float] | None = None,
          exit_zones: list[dict] | None = None, h1_end: dict | None = None, force_immediate: bool = False,
          close_stop: bool = False) -> dict | None:
    """Entry / stop / target for the trigger candle at index i, simulated forward to its outcome.

    partials: prices where half of the trade is closed (candle 2's close of a breaker block formation). The nearest one between the entry and the target
    is used. The other half runs to the target. The stop is not moved, and a candle that touches both the stop and the partial counts as the stop.

    exit_zones: the 1H FVGs in the trade's direction (bearish for a sell, bullish for a buy). A zone printed AFTER the trade's partial exit (after the entry,
    when there is no partial exit) is an exit trigger: when price comes back into it and a 1H candle CLOSES beyond it against the trade (above it for a sell,
    below it for a buy), the trade is closed at that close. That means the level the trade was heading for has, for practical purposes, been reached (on another
    liquidity provider's chart it was hit). h1_end maps the end time of each 1H candle to that candle (so 15M trades can use the 1H close).

    entry_override is for the user's OWN setups: the entry is theirs. It counts as filled at the trigger candle's
    close if that candle traded through it, otherwise it waits for a touch.
    """
    c, p1, p2 = candles[i], candles[i - 1], candles[i - 2]
    rng_hi, rng_lo = max(p1["high"], p2["high"]), min(p1["low"], p2["low"])
    if entry_override is not None:
        entry = entry_override
        inside = c["low"] <= entry <= c["high"]
        mode = "manual"
    else:
        inside = rng_lo <= c["close"] <= rng_hi or force_immediate      # FVG trades enter at the trigger candle's close
        if inside:
            entry, mode = c["close"], "immediate"
        else:
            entry, mode = (p2["high"] if bull else p2["low"]), "retrace"

    risk = (entry - stop) if bull else (stop - entry)
    if (risk <= 0 or ((target0 - entry) if bull else (entry - target0)) <= 0) and entry_override is None and mode == "retrace":
        # The retrace entry makes no sense (it lies beyond the target or the wrong side of the stop: e.g. a sweep candle that closed on the other side of the
        # range, where "retrace to the second candle" would mean buying above the target). Price is already at a better price: enter at the close.
        e2 = c["close"]
        r2 = (e2 - stop) if bull else (stop - e2)
        if r2 > 0 and ((target0 - e2) if bull else (e2 - target0)) > 0:
            entry, mode, inside, risk = e2, "immediate", True, r2
    if risk <= 0 or ((target0 - entry) if bull else (entry - target0)) <= 0:
        return None  # stop on the wrong side of the entry, or the target is already behind it

    # the target never goes behind the entry either
    pfloor = max(floor, entry) if bull else min(floor, entry)
    path = _target_path(candles, i, bull, pfloor, target0, book)

    plan = {
        "mode": mode, "entry": entry, "stop": stop, "target": target0, "initial_target": target0,
        "rr": 0.0, "range_high": rng_hi, "range_low": rng_lo,
        "status": "open" if inside else "waiting",
        "fill_time": c["time"] + tf_sec if inside else None,  # an immediate entry happens as the candle closes
        "resolved_time": None, "outcome": None,
    }
    between = [x for x in (partials or []) if x is not None and ((entry < x < target0) if bull else (target0 < x < entry))]
    part = (min(between) if bull else max(between)) if between else None
    r1 = abs(part - entry) / risk if part is not None else 0.0
    p_hit = False
    filled = inside
    final = target0
    for k, tgt in enumerate(path, start=i + 1):
        later = candles[k]
        final = tgt
        if later["time"] <= ignore_until:
            continue          # flagged by the user as a feed glitch: this candle decides nothing
        if close_stop:      # CPLOC signals: the trade is only stopped out when a candle CLOSES beyond the stop, and it is closed at that close
            hit_stop = later["close"] <= stop if bull else later["close"] >= stop
        else:
            hit_stop = later["low"] <= stop if bull else later["high"] >= stop
        hit_target = later["high"] >= tgt if bull else later["low"] <= tgt
        if not filled:
            touched = later["low"] <= entry if bull else later["high"] >= entry
            if not touched:
                if hit_target:  # price ran to the target without ever retracing: the entry is gone
                    plan.update(status="missed", outcome="missed", resolved_time=later["time"])
                    break
                continue
            filled = True
            plan.update(status="open", fill_time=later["time"])
            if hit_stop:  # entry level is on the entry side of the stop, so the fill came first
                plan.update(status="stopped", outcome="stopped", resolved_time=later["time"])
                if close_stop:
                    plan["stop_close"] = later["close"]
                break
            continue  # target on the fill candle is not counted: its order inside the candle is unknown
        if hit_stop:
            plan.update(status="stopped", outcome="stopped", resolved_time=later["time"])
            if close_stop:
                plan["stop_close"] = later["close"]
            break
        if part is not None and not p_hit and ((later["high"] >= part) if bull else (later["low"] <= part)):
            p_hit = True
            plan["partial_time"] = later["time"]
        if hit_target:
            plan.update(status="target", outcome="target", resolved_time=later["time"])
            break
        anchor = plan.get("partial_time") if part is not None else plan["fill_time"]
        if exit_zones and anchor is not None and not (part is not None and not p_hit):
            later_end = later["time"] + tf_sec
            c1h = later if tf_sec == 3600 else ((h1_end or {}).get(later_end) if later_end % 3600 == 0 else None)
            if c1h is not None:
                for z in exit_zones:
                    if not (anchor < z["created"] <= c1h["time"]) or (z["end"] is not None and z["end"] < later_end):
                        continue
                    zb, zt = fvg.bounds_at(z, c1h["time"])
                    touched = (c1h["high"] >= zb) if not bull else (c1h["low"] <= zt)
                    beyond = (c1h["close"] > zt) if not bull else (c1h["close"] < zb)
                    if touched and beyond:
                        px = c1h["close"]
                        r_exit = ((px - entry) if bull else (entry - px)) / risk
                        total = (0.5 * r1 + 0.5 * r_exit) if (part is not None and p_hit) else r_exit
                        plan["exit"] = {"price": round(px, 2), "time": c1h["time"], "r": round(r_exit, 2), "fvg": [round(zb, 2), round(zt, 2)]}
                        plan["result_r"] = round(total, 2)
                        plan.update(status="target" if total > 0 else "stopped", outcome="target" if total > 0 else "stopped", resolved_time=later["time"])
                        break
                if plan.get("exit"):
                    break
    plan["target"] = final
    reward = (final - entry) if bull else (entry - final)
    rr_final = round(reward / risk, 2) if reward > 0 else 0.0
    plan["rr_final"] = rr_final
    plan["rr"] = rr_final
    if plan["outcome"] == "stopped" and plan.get("stop_close") is not None and part is None and "exit" not in plan:
        # closed at the close that went beyond the stop: that can be a little worse than 1R
        plan["result_r"] = round(-abs(plan["stop_close"] - entry) / risk, 2)
    if part is not None:
        plan["partial"] = {"price": round(part, 2), "r": round(r1, 2), "hit": p_hit}
        plan["rr"] = round(0.5 * r1 + 0.5 * rr_final, 2)            # what the trade is worth when both halves reach their prices
        if plan["outcome"] == "target":
            plan["result_r"] = round(0.5 * r1 + 0.5 * rr_final, 2)
        elif plan["outcome"] == "stopped":
            lost = abs(plan["stop_close"] - entry) / risk if plan.get("stop_close") is not None else 1.0
            plan["result_r"] = round((0.5 * r1 - 0.5 * lost) if p_hit else -lost, 2)
            if plan["result_r"] >= 0:                                # the first half paid for the second half's stop
                plan.update(status="target", outcome="target")
    return plan


KIND_RULE = {
    ("daily", "reversal"): "A 1H CPLOC (FRC) candle crossed a valid Daily level.",
    ("daily", "early"): "A 15M CPLOC crossed a valid 4H level right after price hit a Daily level, with the 4H level at or beyond it.",
    ("daily", "continuation"): "A 15M CPLOC crossed a valid 4H level during a pullback, after a 1H break of structure while a 1H reversal was pending.",
    ("weekly", "reversal"): "A 1H CPLOC crossed a valid Daily level right after price hit a Weekly level, with the Daily level at or beyond it.",
    ("weekly", "early"): "A 15M CPLOC crossed a valid 4H level right after price hit a Weekly level, with the 4H level at or beyond it.",
    ("monthly", "reversal"): "A 1H CPLOC crossed a valid Daily level right after price hit a Monthly level, with the Daily level at or beyond it.",
    ("monthly", "early"): "A 15M CPLOC crossed a valid 4H level right after price hit a Monthly level, with the 4H level at or beyond it.",
    ("weekly", "level-reversal"): "A 4H CPLOC crossed a valid Weekly level and closed beyond it: the Weekly level held and price reversed.",
    ("weekly", "level-hold"): "A Daily level lies inside the range of the Weekly candle that made a valid Weekly level, and a 4H candle crossed that Daily level and closed beyond it: the Weekly level held and price reversed.",
    ("monthly", "level-reversal"): "A Daily CPLOC crossed a valid Monthly level and closed beyond it: the Monthly level held and price reversed.",
    ("monthly", "level-hold"): "A Weekly level lies inside the range of the Monthly candle that made a valid Monthly level, and a Daily candle crossed that Weekly level and closed beyond it: the Monthly level held and price reversed.",
    ("daily", "hold"): "Price hit a valid Daily level; a 4H level sits at or beyond it; then a 1H candle closed over that 4H level in the reversal direction (it did not have to be a CPLOC).",
    ("weekly", "hold"): "Weekly level hold, trimmed down: price reached a valid Weekly level, a Daily level sits at or beyond it, and the Daily / 4H / 1H condition was met there: a 4H level at or beyond the Daily level, and a 1H candle closed beyond it (it did not have to be a CPLOC). The 1H candle is the entry.",
    ("monthly", "hold"): "Monthly level hold, trimmed down: price reached a valid Monthly level, a Daily level sits at or beyond it, and the Daily / 4H / 1H condition was met there: a 4H level at or beyond the Daily level, and a 1H candle closed beyond it. The 1H candle is the entry and the Monthly level is the target.",
}
TF_NAME = {"15m": "15M", "1h": "1H", "4h": "4H", "1d": "Daily", "1w": "Weekly", "1mo": "Monthly"}


def _ohlc(c: dict) -> dict:
    return {"time": c["time"], "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"]}


def _why(tier: str, kind: str, trigger_tf: str, level_tf: str, bull: bool, c: dict, p1: dict, p2: dict,
         stop: float, target0: float, origin: float, ev: dict, plan: dict | None) -> list[str]:
    """The reasons this setup was identified, in plain language with the actual numbers."""
    tfn = TF_NAME[trigger_tf]
    if tier == "manual":
        out = [f"This is YOUR OWN setup, entered by you (or by the assistant at your request); the engine did not find it by its rules.",
               ev.get("note") or "No reason was recorded."]
        if ev.get("user_origin") is not None:
            out.append(f"You based it on the level {ev['user_origin']} (it may come from another provider and not exist in this feed).")
        out.append(f"Entry {plan['entry'] if plan else '?'}, stop {stop} and initial target {target0} are as you set them. "
                   f"The target is dynamic: it moves to a nearer valid {TF_NAME[level_tf]} level when one is created. "
                   "The engine tracks the trade candle by candle like any other.")
        return out
    if tier == "fvg":
        z = ev["fvg_zone"]
        tap = ev.get("tap") or {"time": c["time"], "high": c["high"], "low": c["low"], "close": c["close"]}
        outside = (tap["close"] > z["top"]) if z["bull"] else (tap["close"] < z["bottom"])
        out = [f"A {tfn} candle tapped into a {TF_NAME[z['tf']]} {'bullish' if z['bull'] else 'bearish'} FVG {z['bottom']}-{z['top']} (a fair value gap that has not been "
               f"filled) and closed {'outside it' if outside else 'inside it'} ({tap['close']}). An FVG is a continuation zone: it marks the direction of the trend, and a pullback "
               "into it is expected to be rejected, so the trend continues."]
        if tap["time"] != c["time"]:
            out.append(f"That tap candle was not {'green' if bull else 'red'}, so the trigger is the next {'green' if bull else 'red'} candle ({_ft(c['time'])}), which joins the trend. "
                       f"Entry at its close {c['close']}.")
        else:
            out.append(f"The tap candle is itself {'green' if bull else 'red'}, the colour of the trend, so it is the trigger. Entry at its close {c['close']}.")
        out.append("The zone " + ("caused a break of structure (the indicator's valid FVG)." if z["flagged"] else "did not cause a break of structure (it is not flagged as valid by the indicator)."))
        out.append(f"Stop {stop}: the {'low' if bull else 'high'} of the candle that tapped into the FVG and was rejected.")
        out.append(f"Target {target0}: the {'high' if bull else 'low'} of the move that made the FVG, the {'high' if bull else 'low'} the trend is expected to take out. It does not move.")
        return out
    body_hi, body_lo = max(p1["open"], p1["close"]), min(p1["open"], p1["close"])
    out = [KIND_RULE[(tier, kind)]] + list(ev.get("chain_text") or [])
    if ev.get("candle_rule") == "reversal_candle":
        lv = ", ".join(str(l["price"]) for l in (ev.get("levels_crossed") or []))
        out.append(f"Reversal candle ({tfn}, {'bullish' if bull else 'bearish'}): opened {c['open']} and closed {c['close']}, "
                   f"{'above' if bull else 'below'} the {lv} level; the previous {tfn} candle had closed {p1['close']} on the other side of it. "
                   "A CPLOC shape is not required for this kind.")
    elif bull:
        out.append(f"CPLOC shape (bullish): low {c['low']} <= previous {tfn} candle's body bottom {body_lo} and close {c['close']} > its body top {body_hi}.")
    else:
        out.append(f"CPLOC shape (bearish): high {c['high']} >= previous {tfn} candle's body top {body_hi} and close {c['close']} < its body bottom {body_lo}.")
    crossed = ev.get("levels_crossed") or []
    if crossed:
        out.append("It crossed the valid " + ", ".join(f"{TF_NAME[l['tf']]} level {l['price']}" for l in crossed)
                   + " (a level = an older completed candle close that no later close has broken).")
    if ev.get("rule") == "gated":
        out.append(
            f"The {TF_NAME[ev['lower_level']['tf']]} level {ev['lower_level']['price']} lies inside the range of the {TF_NAME[ev['hit_level_tf']]} candle that made the "
            f"{ev['hit_level_tf'].upper() if ev['hit_level_tf'] != '1d' else 'Daily'} level {ev['hit_level']}: from {ev['hit_level']} out to {ev['range_extreme']} "
            f"({abs(ev['distance_beyond'])} {'beyond' if abs(ev['distance_beyond']) > 0.05 else 'at'} the level). Price had to pass through the {TF_NAME[ev['hit_level_tf']]} level to reach it.")
    if ev.get("rule") == "continuation":
        out.append("A 1H break of structure in the same direction had formed, and the earlier 1H reversal setup was still pending.")
    oc, tc = ev.get("origin_candle"), ev.get("target_candle")
    word = {"buy": "buying", "sell": "selling", "both": "either"}
    if oc and ev.get("polarity_origin"):
        out.append(f"The {TF_NAME[level_tf]} level {origin} is a {word[oc['polarity']]} level: it was a {oc['formation'].replace('→', ' candle followed by a ').lower()} candle"
                   f" (the candle after it opened {oc['next_open']} and closed {oc['next_close']}), "
                   f"and a {'buy' if bull else 'sell'} can start from a {'buying' if bull else 'selling'} level.")
    tgt_note = ""
    if tc and ev.get("polarity_target"):
        tgt_note = f" That level is a {word[tc['polarity']]} level, where {'sellers' if bull else 'buyers'} act, which is where a {'buy' if bull else 'sell'} aims."
    if ev.get("target_cap"):
        out.append(f"Initial target {target0}: CAPPED at the near edge of an unfilled FVG that caused a break of structure. The nearest valid {TF_NAME[level_tf]} level "
                   f"beyond is {ev['target_cap']['level_target']}, the target once that FVG is filled.")
    else:
        out.append(f"Initial target {target0}: the nearest valid {TF_NAME[level_tf]} level beyond the CPLOC close and the origin level {origin}.{tgt_note} "
                   "Targets update if a nearer valid level is created while the trade runs.")
    out.append(f"Stop {stop}: the {'lowest low' if bull else 'highest high'} of the trigger candle and the two before it.")
    if plan:
        if plan["mode"] == "immediate":
            out.append(f"Entry {plan['entry']}: the CPLOC closed inside the range ({plan['range_low']} - {plan['range_high']}) of the two candles before it, so entry is at its close.")
        else:
            out.append(f"Entry {plan['entry']}: the CPLOC closed outside the range ({plan['range_low']} - {plan['range_high']}) of the two candles before it, "
                       "so entry waits for a retrace to the second candle before it.")
    return out


def _build(tier: str, kind: str, trigger_tf: str, level_tf: str, candles: list[dict], i: int,
           direction: str, origin: float, target0: float, book: LevelBook, evidence: dict | None = None,
           manual: dict | None = None, stop_override: float | None = None) -> dict | None:
    if i < 2:
        return None  # the entry check needs two candles before the trigger
    bull = direction == "bull"
    c, p1, p2 = candles[i], candles[i - 1], candles[i - 2]
    if kind == "level-hold" and not manual and stop_override is None:
        stop_override = c["low"] if bull else c["high"]             # the rejection candle serves as the entry (its close) and the stop (its extreme)
    stop = manual["stop"] if manual else (stop_override if stop_override is not None else
                                           (min(c["low"], p1["low"], p2["low"]) if bull else max(c["high"], p1["high"], p2["high"])))
    floor = max(c["close"], origin) if bull else min(c["close"], origin)
    if manual:
        floor = max(floor, manual["entry"]) if bull else min(floor, manual["entry"])
    setup_id = f"manual-{manual['id']}" if manual else f"{tier}-{kind}-{trigger_tf}-{c['time']}-{direction}"
    bb, bb_target, c2_closes = set(), None, []
    if not manual and tier != "fvg":
        known_t = c["time"] + TF_SEC[trigger_tf]
        up_ev = (evidence or {}).get("upgraded_from") or {}
        checks = [(origin, level_tf if kind.startswith("level-") else "1d")] + ([(up_ev["higher_level"], up_ev["higher_tf"])] if up_ev.get("higher_level") else [])
        for lvl, ftf in checks:                                   # the Daily level, and the Weekly / Monthly level that gave the target
            closes, c2 = _bb_closes(lvl, bull, known_t, ftf)
            if closes:
                bb |= closes
                c2_closes.append(c2)
        if bb:
            book = _SkipBook(book, bb)
            target0 = book.next(c["time"], max(c["close"], origin) if bull else min(c["close"], origin), bull)
            if target0 is None:
                return None                                     # no level beyond the formation to aim for
            bb_target = target0
    leg = None
    if not manual and tier == "daily" and not bb:
        # "Always look in terms of Daily": when the DAILY chart has already given an FVG continuation signal in this direction (a Daily candle tapped a Daily
        # FVG and closed outside it) and that trend has not reached its Daily low / high yet, the target is that Daily low / high.
        leg = fvg.alive_leg(c["time"] + TF_SEC[trigger_tf], bull, c["close"])
        if leg:
            target0 = leg["target"]
            book = _FixedBook(book, target0)
    if not manual and tier != "fvg" and fvg.opposing_signal(c["time"] + TF_SEC[trigger_tf], bull, c["close"], TF_SEC[trigger_tf]):
        return None            # a bounce against a higher-timeframe FVG rejection and its trend: valid as a bounce, never a trade of this plan
    fvg_info = None
    if not manual and tier != "fvg":
        known = c["time"] + TF_SEC[trigger_tf]                    # the trigger candle's close: what the trader knows at entry
        ahead = fvg.ahead(known, bull, c["close"], target0) if bull else fvg.ahead(known, bull, target0, c["close"])
        if ahead:
            caps = [(z["edge"], z["end"]) for z in ahead if z["flagged"]]
            cap_applied = False
            uncapped = target0
            if caps:
                book = _CappedBook(book, caps)
                nearest = caps[0][0]
                if (nearest < target0) if bull else (nearest > target0):
                    target0, cap_applied = nearest, True
            z0 = ahead[0]
            fvg_info = {"cap_applied": cap_applied, "cap": caps[0][0] if caps else None, "level_target": uncapped,
                        "zones": [{"tf": z["tf"], "bull": z["bull"], "top": z["top"], "bottom": z["bottom"], "formed": z["formed"], "flagged": z["flagged"]} for z in ahead[:4]],
                        "warning": (f"{TF_NAME[z0['tf']]} {'bullish' if z0['bull'] else 'bearish'} FVG {z0['bottom']}-{z0['top']} is in place between the entry and the target: "
                                    f"a continuation zone AGAINST this trade, where price may be rejected."
                                    + (f" The target is capped at its near edge {z0['edge']} because it caused a break of structure." if cap_applied else
                                       (" No cap: it did not cause a break of structure (capping at those made the history worse)." if not z0["flagged"] else "")))}
    override = _OVERRIDES.get(setup_id)
    ignore_until = override["ignore_until"] if override else 0
    path = _target_path(candles, i, bull, floor, target0, book)
    close_stop = (not manual) and (kind in ("reversal", "early", "continuation", "level-reversal") or (tier == "fvg" and kind == "15m"))     # the CPLOC signals
    status, resolved, target_then = _resolve(candles, i, bull, stop, path, ignore_until, close_stop)
    # the target as it stood when the setup ended (or is now, while it is still pending)
    final = target_then if target_then is not None else (path[-1] if path else target0)
    exit_zones, h1_end = None, None
    if not manual and (FVG_EXIT == "all" or (FVG_EXIT == "bb" and bb)):
        exit_zones = [z for z in fvg.zones("1h") if z["bull"] == bull and z["created"] > c["time"]]
        h1_end = _h1_by_end()
    plan = _plan(candles, i, bull, target0, stop, floor, TF_SEC[trigger_tf], book, manual["entry"] if manual else None, ignore_until, c2_closes, exit_zones, h1_end, force_immediate=(tier == "fvg" or kind == "level-hold"), close_stop=close_stop)
    if plan and plan.get("exit") and (resolved is None or plan["resolved_time"] < resolved):      # closed on the FVG exit before its target or stop
        status, resolved = ("completed" if plan["outcome"] == "target" else "failed"), plan["resolved_time"]
    if plan and not manual and plan["rr"] < MIN_RR:
        # Shown, but let go: a plan that pays less than it risks is not flagged as a trade. It stays in History (with what would have happened).
        plan.update(would_be=plan["outcome"] or "pending", skipped=f"R:R {plan['rr']} is below {MIN_RR:g}", status="skipped", outcome="skipped")
    ev = {**(evidence or {}), "trigger": _ohlc(c), "prev1": _ohlc(p1), "prev2": _ohlc(p2),
          "origin_candle": book.info(origin), "target_candle": book.info(target0), "polarity_origin": levelsmod.POLARITY_ORIGIN, "polarity_target": levelsmod.POLARITY_TARGET}
    ev["target_cap"] = {"level_target": fvg_info["level_target"], "edge": target0} if fvg_info and fvg_info["cap_applied"] else None
    ev["why"] = _why(tier, kind, trigger_tf, level_tf, bull, c, p1, p2, stop, target0, origin, ev, plan)
    if bb:
        ev["why"].insert(1, f"The Daily level {origin} is a BREAKER BLOCK level: the next Daily candle broke it with a CPLOC and a later candle retested it and closed back "
                            f"{'above' if bull else 'below'} it. The closes inside that formation ({', '.join(str(x) for x in sorted(bb))}) are skipped as targets; "
                            f"the target is the next Daily level beyond them: {bb_target}.")
        ev["breaker_block"] = sorted(bb)
        if plan and plan.get("partial"):
            ev["why"].insert(2, f"Half of the trade is closed at {plan['partial']['price']}, the close of the formation's candle 2 ({plan['partial']['r']}R), and the rest runs "
                                f"to {bb_target} or beyond. The stop stays where it is.")
    if leg:
        ev["daily_fvg_leg"] = {"target": leg["target"], "stop": leg["stop"], "tap_time": leg["tap_time"], "zone": leg["zone"]}
        ev["why"].insert(1, f"The DAILY chart already gave an FVG continuation signal in this direction: on {_ft(leg['tap_time'])[:5]} a Daily candle tapped the Daily FVG "
                            f"{leg['zone']['bottom']}-{leg['zone']['top']} and closed outside it. That trend has not reached the Daily {'high' if bull else 'low'} "
                            f"{leg['target']} (the swing {'high' if bull else 'low'} price printed before it came back into the FVG) yet, so that is the target, not the next Daily level.")
    if close_stop:
        ev["why"].append(f"The stop {stop} only counts when a {TF_NAME[trigger_tf]} candle CLOSES beyond it: price touching it and failing to close beyond it leaves the setup valid.")
    if fvg_info:
        ev["why"].append("WARNING: " + fvg_info["warning"])
    cust = book.custom_at(origin) or next((book.custom_at(l["price"]) for l in (ev.get("levels_crossed") or []) if l.get("custom") and book.custom_at(l["price"])), None)
    if not cust:      # the level it formed on may be the lower-timeframe level the trigger crossed (hold / early kinds)
        cust = next(({"id": l["id"], "label": l.get("label", ""), "price": l["price"], "polarity": "buy" if bull else "sell"} for l in (ev.get("levels_crossed") or []) if l.get("custom")), None)
    if cust:
        ev["why"].insert(0, f"The level {cust['price']} is one YOU marked" + (f" ({cust['label']})" if cust.get("label") else "")
                         + f" as a {'buying' if cust.get('polarity', 'buy') == 'buy' else 'selling'} level; the engine treated it like any other valid level.")
    if override:
        import datetime as _dt
        ev["why"].append("Correction by you: candles up to " + _dt.datetime.fromtimestamp(ignore_until, _dt.UTC).strftime("%m-%d %H:%M")
                         + " are ignored for this setup (marked as a feed glitch" + (": " + override["note"] if override["note"] and override["note"].lower() != "feed glitch" else "") + ").")
    return {
        "id": setup_id,
        "override": override,
        "tier": tier, "kind": kind, "trigger_tf": trigger_tf, "level_tf": level_tf,
        "time": c["time"], "direction": direction, "origin": origin,
        "open": c["open"], "close": c["close"], "high": c["high"], "low": c["low"],
        "extreme": stop,  # the level whose break fails the setup (kept under its old name for the chart)
        "stop": stop,
        "target": final, "initial_target": target0,
        "target_history": _history(candles, i, target0, path, resolved),
        "status": status, "resolved_time": resolved, "invalid": None, "htf_conflict": None,
        "plan": plan,
        "fvg": fvg_info,
        "custom_level": ({"id": cust["id"], "label": cust.get("label", ""), "price": cust["price"]} if cust else None),
        "evidence": ev,
    }


def _find_level_reversals(tier: str, c_tf: str, level_tf: str) -> list[dict]:
    """Weekly / Monthly level reversal, signal 1: a CPLOC on `c_tf` (4H for Weekly, Daily for Monthly) crosses a valid `level_tf` level and closes beyond it.
    Its own trade: the CPLOC rules for entry and stop, the target the next valid level of `level_tf`. No early signal at this tier."""
    ltf = data.try_candles(c_tf)
    htf = data.try_candles(level_tf)
    if not ltf or not htf:
        return []
    book = LevelBook(htf, level_tf)
    index_of = {c["time"]: i for i, c in enumerate(ltf)}
    out: list[dict] = []
    for f in frc_levels.find(c_tf, level_tf=level_tf):
        i = index_of.get(f["time"])
        if i is None:
            continue
        c, bull = ltf[i], f["direction"] == "bull"
        crossed = [l["price"] for l in f["levels"]]
        origin = max(crossed) if bull else min(crossed)
        target = book.next(c["time"], max(c["close"], origin) if bull else min(c["close"], origin), bull)
        if target is None or _is_repeat(out, f["direction"], target, f["time"]):
            continue
        built = _build(tier, "level-reversal", c_tf, level_tf, ltf, i, f["direction"], origin, target, book, {"rule": "reversal", "levels_crossed": f["levels"]})
        if built and not _dup(out, built):
            out.append(built)
    return out


def _find_level_holds(tier: str, c_tf: str, l2_tf: str, hit_tf: str) -> list[dict]:
    """Weekly / Monthly level reversal, signal 2: a `l2_tf` level (Daily for Weekly, Weekly for Monthly) lies inside the range of the `hit_tf` candle that made the
    level, and a `c_tf` candle (4H, Daily) crosses it and closes beyond it. Not a CPLOC: the rejection candle is the entry (its close) and its extreme the stop."""
    if not data.try_candles(hit_tf) or not data.try_candles(l2_tf):
        return []
    return _gated_core(tier, "level-hold", c_tf, hit_tf, l2_tf, _hold_candidates(c_tf, l2_tf), {"candle_rule": "reversal_candle"})


def _find_reversals() -> list[dict]:
    """Daily tier: 1H CPLOC on a Daily level."""
    ltf = data.get_candles("1h")
    book = LevelBook(data.get_candles("1d"), "1d")
    index_of = {c["time"]: i for i, c in enumerate(ltf)}

    out: list[dict] = []
    for f in frc_levels.find("1h"):
        i = index_of[f["time"]]
        c = ltf[i]
        bull = f["direction"] == "bull"
        crossed = [l["price"] for l in f["levels"]]
        origin = max(crossed) if bull else min(crossed)
        target = book.next(c["time"], max(c["close"], origin) if bull else min(c["close"], origin), bull)
        if target is None or _is_repeat(out, f["direction"], target, f["time"]):
            continue
        built = _build("daily", "reversal", "1h", "1d", ltf, i, f["direction"], origin, target, book,
                       {"rule": "reversal", "levels_crossed": f["levels"]})
        if built and not _dup(out, built):
            out.append(built)
    return out


def _find_gated(tier: str, kind: str, trigger_tf: str, hit_tf: str) -> list[dict]:
    """CPLOC on a lower-tier level, after price hit a `hit_tf` level, with the lower-tier level at/beyond it.

    Daily early:   trigger 15m (on 4H levels), hit_tf 1d.
    Weekly:        trigger 4h / 1h (on Daily levels), hit_tf 1w.
    """
    reach_tf = frc_levels.LTF_TO_LEVEL_TF[trigger_tf]  # timeframe of the levels the trigger crosses
    return _gated_core(tier, kind, trigger_tf, hit_tf, reach_tf, frc_levels.find(trigger_tf), {"candle_rule": "cploc"})


def _hold_candidates(c_tf: str, l2_tf: str) -> list[dict]:
    """Reversal candles on `c_tf`: a candle that CROSSES a valid `l2_tf` level and CLOSES beyond it in the reversal direction.

    Crossing means the candle went through the level: either the previous candle closed on the near side of it (a plain close beyond), or the candle's
    own wick reached the level (price swept it and was rejected: a low at or below a buying level with the close above it, a high at or above a selling
    level with the close below it), however small the poke. A CPLOC shape is not needed, but the colour is: a buy needs a green candle (close above
    its open), a sell a red one (close below its open)."""
    m = data.get_candles(c_tf)
    l2 = data.get_candles(l2_tf)
    book2 = LevelBook(l2, l2_tf)
    time_of = {c["close"]: c["time"] for c in l2}
    out = []
    for i in range(3, len(m) - 1):                       # completed candles only; the plan needs two candles before
        c, prev = m[i], m[i - 1]
        up, down = book2.valid_lists(c["time"])
        # a buy: a support (valid while no later close is below it) at or above the candle's low (or the previous close), under the candle's close
        bull_lv = [lv for lv in down[bisect_left(down, min(c["low"], prev["close"])):bisect_left(down, c["close"])] if book2.origin_ok(lv, True)]
        # a sell: a resistance (valid while no later close is above it) at or below the candle's high (or the previous close), over the candle's close
        bear_lv = [lv for lv in up[bisect_right(up, c["close"]):bisect_right(up, max(c["high"], prev["close"]))] if book2.origin_ok(lv, False)]
        green, red = c["close"] > c["open"], c["close"] < c["open"]
        # the candle's colour decides: a buy needs a green candle (close above its open), a sell a red one
        for direction, levels in (("bull", bull_lv if green else []), ("bear", bear_lv if red else [])):
            if levels:
                out.append({"time": c["time"], "direction": direction,
                            "levels": [{"tf": l2_tf, "price": lv, "time": time_of.get(lv)} for lv in levels]})
    return out


def _find_hold(tier: str, l1_tf: str, l2_tf: str, c_tf: str) -> list[dict]:
    """The three-timeframe level hold. Level on `l1_tf` -> confirmation level on `l2_tf` at/beyond it -> reversal candle on `c_tf`.
    Monthly: M / W / D.  Weekly: W / D / 4H.  Daily: D / 4H / 1H."""
    return _gated_core(tier, "hold", c_tf, l1_tf, l2_tf, _hold_candidates(c_tf, l2_tf), {"candle_rule": "reversal_candle"}, arrival="approach")


def _ft(t: int) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(t, _dt.UTC).strftime("%m-%d %H:%M")


_CHAIN_EVIDENCE = ("rule", "candle_rule", "hit_level", "hit_level_tf", "touched_at", "arrived_from", "candle_before_arrival", "lower_level",
                   "distance_beyond", "max_reach", "levels_crossed")


UPGRADE_TAP_HOURS = 72   # "price tapped into the Weekly / Monthly level": within this many hours before the trigger candle


def _upgrade_targets(daily: list[dict]) -> list[dict]:
    """Weekly and Monthly levels are TARGETS, not setups. A Daily setup (1H CPLOC on a Daily level / Daily + 4H level + 1H close / the early 15M
    CPLOC) keeps being exactly that. When it happens where a higher-timeframe level is involved, it is upgraded: same entry, same stop, but the target
    is the next valid Weekly (or Monthly) level instead of the next Daily one.

    Involved means: a valid Monthly / Weekly level of the right kind for the trade (a buying level for a buy) that
      - IS the Daily level (same price), or
      - price tapped into recently (within UPGRADE_TAP_HOURS before the trigger candle), with the Daily level at or beyond it.
    In both cases the Daily level must lie between the level and the extreme of the candle that made it (its low for a buying level, its high for a
    selling level), and price must not have broken that extreme since: once it has, the level no longer holds.
    Monthly wins over Weekly."""
    h1 = data.get_candles("1h")
    h1_times = [c["time"] for c in h1]
    dcs = data.get_candles("1d")
    d_times = [c["time"] for c in dcs]
    by_tf = {"1h": h1, "15m": data.get_candles("15m")}
    index_of = {tf: {c["time"]: i for i, c in enumerate(cs)} for tf, cs in by_tf.items()}
    books = {}
    for tf in ("1mo", "1w"):
        cs = data.try_candles(tf)
        if cs:
            books[tf] = LevelBook(cs, tf)
    out: list[dict] = []
    for st in daily:
        if st["tier"] != "daily" or st["kind"] not in ("reversal", "early", "hold") or st["trigger_tf"] not in by_tf:
            continue
        bull = st["direction"] == "bull"
        t = st["time"]
        reach = EARLY_REACH_ATR * _atr_before(dcs, bisect_right(d_times, t) - 1)
        D = st["origin"]                                        # the Daily level the setup is on
        lo_t = t - UPGRADE_TAP_HOURS * 3600
        a, b = bisect_left(h1_times, lo_t), bisect_right(h1_times, t)
        recent = h1[a:b]
        chosen = None
        for tf in ("1mo", "1w"):
            book = books.get(tf)
            if not book:
                continue
            up, down = book.valid_lists(t)
            best = None
            for L in sorted(set(up) | set(down)):
                if not book.origin_ok(L, bull):
                    continue
                k = book.idx_of.get(L)
                if k is not None and L not in book.cpol:
                    # The higher-timeframe candle that made the level has an extreme: its LOW for a buying level, its HIGH for a selling level. The
                    # Daily level must be at the level or beyond it but not beyond that extreme, and price must not have broken the extreme since.
                    cand = book.candles[k]
                    E = cand["low"] if bull else cand["high"]
                    if not ((E - LEVEL_TOL <= D <= L + LEVEL_TOL) if bull else (L - LEVEL_TOL <= D <= E + LEVEL_TOL)):
                        continue
                    end_t = book.candles[k + 1]["time"] if k + 1 < len(book.candles) else cand["time"] + TF_SECONDS[tf]       # when that candle closed
                    a2 = bisect_left(h1_times, end_t)
                    since = h1[a2:b]
                    if since and ((min(c["low"] for c in since) < E - LEVEL_TOL) if bull else (max(c["high"] for c in since) > E + LEVEL_TOL)):
                        continue                                  # the candle's low / high has been breached: the level no longer holds
                else:
                    E = None
                    if not ((-reach <= D - L <= LEVEL_TOL) if bull else (-LEVEL_TOL <= D - L <= reach)):
                        continue
                same = abs(D - L) <= LEVEL_TOL
                tapped = any(c["low"] <= L <= c["high"] for c in recent)
                if (same or tapped) and (best is None or abs(D - L) < abs(D - best[0])):
                    best = (L, E)
            if best is not None:
                chosen = (tf, best[0], abs(D - best[0]) <= LEVEL_TOL, best[1])
                break
        if not chosen:
            continue
        tf, L, same, E = chosen
        cs = by_tf[st["trigger_tf"]]
        i = index_of[st["trigger_tf"]].get(t)
        if i is None:
            continue
        book, c = books[tf], cs[i]
        tier = "monthly" if tf == "1mo" else "weekly"
        target = book.next(c["time"], max(c["close"], L) if bull else min(c["close"], L), bull)
        if target is None or _is_repeat(out, st["direction"], target, t):
            continue
        ev = {k: v for k, v in st["evidence"].items() if k not in ("why", "trigger", "prev1", "prev2", "origin_candle", "target_candle", "target_cap",
                                                                   "polarity_origin", "polarity_target")}
        ev["upgraded_from"] = {"id": st["id"], "daily_level": D, "higher_level": L, "higher_tf": tf, "same_level": same}
        built = _build(tier, st["kind"], st["trigger_tf"], tf, cs, i, st["direction"], D, target, book, ev)
        if not built or _dup(out, built):
            continue
        name = TF_NAME[tf]
        built["evidence"]["why"].insert(1, (f"The Daily level {D} is also a {name} level, " if same else
                                            f"Price tapped the {name} level {L} and the Daily level {D} sits at or beyond it, ")
                                        + (f"and that {name} candle's {'low' if bull else 'high'} {E} has not been breached, " if E is not None else "")
                                        + f"so this same Daily setup has the {name} level as its target: the next valid {name} level {target}, not the next Daily one.")
        out.append(built)
    return out


def _gated_core(tier: str, kind: str, trigger_tf: str, hit_tf: str, reach_tf: str, candidates: list[dict], extra: dict,
                arrival: str = "episode") -> list[dict]:
    m = data.get_candles(trigger_tf)
    hit = data.get_candles(hit_tf)
    reach_c = data.get_candles(reach_tf)
    hit_times = [c["time"] for c in hit]
    hit_closes = [c["close"] for c in hit]
    hit_index = frc_levels.LevelIndex(hit_closes)
    reach_times = [c["time"] for c in reach_c]
    book = LevelBook(hit, hit_tf)
    index_of = {c["time"]: i for i, c in enumerate(m)}

    out: list[dict] = []
    for f in candidates:
        i = index_of[f["time"]]
        c = m[i]
        bull = f["direction"] == "bull"
        idx_h = bisect_right(hit_times, c["time"]) - 1
        ks = [k for k in hit_index.valid(idx_h, hit_closes[idx_h - 1] if idx_h >= 1 else c["close"])
              if levelsmod.origin_ok(hit[k + 1] if k + 1 < len(hit) else None, bull)]       # a buy starts at a buying level (the candle after it went up), a sell at a selling one
        kof = {hit_closes[k]: k for k in ks}
        levels = set(kof)
        for cl in book.custom:      # a level the user marked on the hit timeframe is a valid hit level too (for the direction it serves)
            if cl["polarity"] == ("buy" if bull else "sell") and levelsmod.custom_intact(cl, hit, idx_h, c["close"], c["time"]):
                levels.add(cl["price"])
        if not levels:
            continue

        # The 4H level must lie INSIDE the range of the Daily candle that made the level: from its close (the level) out to its low (a buy) / high (a sell).
        # A 4H level outside that range is ignored. (A level the user marked has no candle: one Daily ATR stands in for the range.)
        best = None  # (distance, hit-tier level, lower-tier level, details)
        fallback = EARLY_REACH_ATR * _atr_before(hit, idx_h)
        for hl in sorted(levels):
            k = kof.get(hl)
            ext = (hit[k]["low"] if bull else hit[k]["high"]) if k is not None else ((hl - fallback) if bull else (hl + fallback))
            for lv in f["levels"]:
                p = lv["price"]
                ok = (ext - LEVEL_TOL <= p <= hl + LEVEL_TOL) if bull else (hl - LEVEL_TOL <= p <= ext + LEVEL_TOL)
                if ok and (best is None or abs(p - hl) < best[0]):
                    best = (abs(p - hl), hl, p, {
                        "hit_level": hl, "hit_level_tf": hit_tf, "touched_at": c["time"], "arrived_from": "above" if bull else "below",
                        "lower_level": {"tf": reach_tf, "price": p, "time": lv["time"]},
                        "distance_beyond": round(p - hl, 2), "range_extreme": round(ext, 2),
                    })
        if best is None:
            continue

        # The target must lie beyond BOTH the CPLOC close and the higher-tier level that was hit: if the
        # CPLOC closes while still past that level, the nearest level beyond the close would be the level
        # itself (a reversal "targeting" its own origin).
        ref = min(c["close"], best[1]) if not bull else max(c["close"], best[1])
        target = book.next(c["time"], ref, bull)
        if target is None or _is_repeat(out, f["direction"], target, f["time"]):
            continue
        built = _build(tier, kind, trigger_tf, hit_tf, m, i, f["direction"], best[1], target, book,
                       {"rule": "gated", **best[3], "levels_crossed": f["levels"], **extra})
        if built and not _dup(out, built):
            out.append(built)
    return out


def _find_continuations(reversals: list[dict]) -> list[dict]:
    """Daily tier: 15M CPLOC on a 4H level while a 1H reversal is pending, after a 1H BOS in its direction."""
    h1 = data.get_candles("1h")
    m15 = data.get_candles("15m")
    index_of = {c["time"]: i for i, c in enumerate(m15)}
    bos = patterns.find_bos(h1)
    frcs = frc_levels.find("15m")
    book = LevelBook(data.get_candles("1d"), "1d")

    out: list[dict] = []
    for s in reversals:
        end = s["resolved_time"]  # None while pending
        known = next(
            (b["time"] + 3600 for b in bos
             if b["direction"] == s["direction"] and b["time"] > s["time"] and (end is None or b["time"] < end)),
            None,
        )  # the BOS is known when its 1H candle closes
        if known is None:
            continue
        bull = s["direction"] == "bull"
        for f in frcs:
            if f["direction"] != s["direction"] or f["time"] < known or (end is not None and f["time"] >= end):
                continue
            i = index_of[f["time"]]
            c = m15[i]
            ref = max(c["close"], s["origin"]) if bull else min(c["close"], s["origin"])
            target = book.next(c["time"], ref, bull)
            if target is None or _is_repeat(out, f["direction"], target, f["time"]):
                continue
            built = _build("daily", "continuation", "15m", "1d", m15, i, f["direction"], s["origin"], target, book,
                           {"rule": "continuation", "parent_id": s["id"], "parent_time": s["time"],
                            "bos_candle_time": known - 3600, "bos_known_at": known, "levels_crossed": f["levels"]})
            if built and not _dup(out, built):
                out.append(built)
    return out


TIER_RANK = {"daily": 1, "weekly": 2, "monthly": 3, "fvg": 0.5}
LEVEL_RANK = {"1d": 1, "1w": 2, "1mo": 3}


def _rank(s: dict) -> int:
    """How high a timeframe a setup lives on. Your own setups rank by the level timeframe their target follows."""
    return LEVEL_RANK.get(s["level_tf"], 1) if s["tier"] == "manual" else TIER_RANK[s["tier"]]


def _apply_weekly_conflicts(setups: list[dict]) -> None:
    """Flag (never invalidate) setups whose target passes the entry / stop of a pending OPPOSITE setup on a higher timeframe."""
    higher = [s for s in setups if s["tier"] != "manual" and s["plan"]]
    for s in setups:
        bull = s["direction"] == "bull"
        for w in higher:
            if _rank(w) <= _rank(s):
                continue
            pending_then = w["time"] <= s["time"] and (w["resolved_time"] is None or w["resolved_time"] > s["time"])
            if not pending_then or w["direction"] == s["direction"]:
                continue

            def passes(level: float, target: float) -> bool:   # a bullish setup's target above the level; bearish below it
                return target > level if bull else target < level

            w_entry, w_stop = w["plan"]["entry"], w["plan"]["stop"]
            at_creation = s["initial_target"]
            if passes(w_entry, at_creation):
                names = {"weekly": "Weekly", "monthly": "Monthly"}.get(w["tier"], "higher-timeframe") + (" buy" if w["direction"] == "bull" else " sell")
                s["htf_conflict"] = {
                    "weekly_id": w["id"], "weekly_direction": w["direction"], "weekly_entry": w_entry, "weekly_stop": w_stop,
                    "passes": "stop" if passes(w_stop, at_creation) else "entry",
                    "now_passes": passes(w_entry, s["target"]),   # False once the target has moved to a nearer level
                    "text": f"Target {at_creation} passes the {names} entry {w_entry}"
                            + (f" and its stop {w_stop}" if passes(w_stop, at_creation) else "")
                            + ". Kept pending: a nearer level usually forms and the target moves.",
                }
                break


def expected_shift(s: dict, price: float) -> dict | None:
    """Where a LIVE setup's target moves at the next rollover of its level timeframe, if it moves.

    Targets are dynamic, and the close of the candle just before the running one becomes a valid level when the next
    candle opens. This looks one rollover ahead (assuming price stays where it is) so the user is not surprised.
    """
    plan = s.get("plan")
    if not plan or plan.get("outcome") is not None or s["tier"] == "fvg" or s["level_tf"] not in ("1d", "1w", "1mo"):
        return None
    candles = data.try_candles(s["level_tf"])
    if not candles:
        return None
    from .levels import next_boundary
    boundary = next_boundary(s["level_tf"], candles[-1]["time"])
    hypo = candles + [{"time": boundary, "open": price, "high": price, "low": price, "close": price}]
    bull = s["direction"] == "bull"
    floor = max(s["close"], s["origin"], plan["entry"]) if bull else min(s["close"], s["origin"], plan["entry"])
    ref = max(price, floor) if bull else min(price, floor)
    new = LevelBook(hypo, s["level_tf"]).next(boundary, ref, bull)
    if new is None or abs(new - plan["target"]) < 1e-9:
        return None
    out = {"target": new, "at": boundary, "level_tf": s["level_tf"], "stays": plan["target"]}
    # The new target is the close of the candle just before the running one. What that level is FOR is decided by the running candle,
    # which has not closed: a sell's target must be a buying level (the running candle closes ABOVE it), a buy's a selling level (BELOW).
    if abs(new - candles[-2]["close"]) < 1e-9:
        out["conditional"] = {"level": new, "needs_close": "below" if bull else "above", "running_close": candles[-1]["close"]}
    return out


def _find_manual() -> list[dict]:
    """The user's own setups, tracked exactly like the engine's (dynamic target, stop, outcome).

    Only setups created on the feed that is active now are shown: their trigger time is on that feed's clock.
    """
    out = []
    active = data.active_source()
    for ms in store.list_manual_setups():
        if ms["source"] != active:
            continue
        candles = data.get_candles(ms["trigger_tf"])
        times = [c["time"] for c in candles]
        i = bisect_right(times, ms["trigger_time"]) - 1 if ms["trigger_time"] else len(candles) - 2
        i = min(i, len(candles) - 2)            # the running candle cannot be a trigger
        if i < 2:
            continue
        bull = ms["direction"] == "bull"
        book = LevelBook(data.get_candles(ms["level_tf"]), ms["level_tf"])
        c = candles[i]
        origin = ms["origin"] if ms["origin"] is not None else ms["entry"]
        floor = max(c["close"], origin, ms["entry"]) if bull else min(c["close"], origin, ms["entry"])
        target0 = ms["target"] if ms["target"] is not None else book.next(c["time"], floor, bull)
        if target0 is None:
            continue
        built = _build("manual", "manual", ms["trigger_tf"], ms["level_tf"], candles, i, ms["direction"], origin, target0, book,
                       {"rule": "manual", "note": ms["note"] or ms["label"], "user_origin": ms["origin"]}, manual=ms)
        if built:
            built["label_text"] = ms["label"]
            out.append(built)
    return out


def _swings(c: list[dict], n: int = 5) -> tuple[list, list]:
    """Swing highs / lows (the indicator's swing length 5) as (index when known, price)."""
    sh, sl = [], []
    for p in range(n, len(c) - n):
        if all(c[p]["high"] > c[p - k]["high"] for k in range(1, n + 1)) and all(c[p]["high"] >= c[p + k]["high"] for k in range(1, n + 1)):
            sh.append((p + n, c[p]["high"]))
        if all(c[p]["low"] < c[p - k]["low"] for k in range(1, n + 1)) and all(c[p]["low"] <= c[p + k]["low"] for k in range(1, n + 1)):
            sl.append((p + n, c[p]["low"]))
    return sh, sl


def _fvg_target(h1: list[dict], h1_times: list[int], z: dict, end: int, bull: bool, close: float, tap_time: int) -> float | None:
    """Where an FVG rejection is heading, on the gap's OWN timeframe: the highest high (buy) / lowest low (sell) from the latest break of structure of that
    timeframe (it must be in the gap's direction: see fvg.trend_start) up to the tap, i.e. the peak the pullback into the gap came from. The 1H candles
    h1[..end) are used (a 4H / Daily high is the highest of its 1H highs). None when the trend is not the gap's way, or the peak is not beyond the close
    (price already ran past it): then there is no trade."""
    b = fvg.trend_start(z, tap_time)
    if b is None:
        return None
    seg = h1[bisect_left(h1_times, b):end]
    if not seg:
        return None
    ext = max(c["high"] for c in seg) if bull else min(c["low"] for c in seg)
    return ext if ((ext > close) if bull else (ext < close)) else None


def _find_fvg_continuations() -> list[dict]:
    """A 1H candle taps into an unfilled FVG (1H, 4H or Daily) and CLOSES OUTSIDE it (below a bearish one, above a bullish one), the candle before having
    closed outside too: the TAP candle. The trend of the FVG continues. The TRIGGER is the first candle of the trend's colour (red for a sell, green for a
    buy): the tap candle itself when it has that colour, otherwise the next such candle within 3 candles, as long as no candle closes back into the zone in
    between. Entry at the trigger's close. The STOP is the extreme of the tap candle (its high for a sell, its low for a buy). The target is the low / high
    of the move that made the FVG. Every tap into the unfilled part that rejects is a setup."""
    h1 = data.get_candles("1h")
    times = [c["time"] for c in h1]
    book0 = LevelBook(data.get_candles("1d"), "1d")
    out: list[dict] = []
    last = len(h1) - 1                       # the running candle cannot be a trigger
    for tf in fvg.TFS:
        for z in fvg.zones(tf):
            bull = z["bull"]
            used = -1                                              # candles up to this index belong to a rejection already handled
            for k in range(max(bisect_left(times, z["created"]), 3), last):
                c = h1[k]
                if z["end"] is not None and c["time"] >= z["end"]:
                    break
                if k <= used:
                    continue
                g = fvg.find_rejection(z, h1, k, last)             # tap (k) and trigger (k..k+3): see fvg.find_rejection
                if g is None:
                    continue                                       # no rejection here; the unfilled part shrinks by the wick and later taps still count
                used = g
                zb, zt = fvg.bounds_at(z, c["time"])               # the unfilled part as it stood when the tap candle opened
                if g is not None and fvg.opposing_signal(h1[g]["time"] + 3600, bull, h1[g]["close"], fvg.SEC[tf]):
                    g = None                                       # a HIGHER-timeframe FVG signal the other way is in play: not a valid continuation
                if g is not None:
                    trig = h1[g]
                    span = h1[k:g + 1]
                    stop = min(x["low"] for x in span) if bull else max(x["high"] for x in span)       # the extreme of the tap (and trigger) candle(s)
                    target = _fvg_target(h1, times, z, k, bull, trig["close"], c["time"])
                    if target is not None and not _is_repeat(out, "bull" if bull else "bear", target, trig["time"]):
                        built = _build("fvg", tf, "1h", tf, h1, g, "bull" if bull else "bear", zt if bull else zb, target,
                                       _FixedBook(book0, target),
                                       {"rule": "fvg", "tap": {"time": c["time"], "high": c["high"], "low": c["low"], "close": c["close"]},
                                        "fvg_zone": {"tf": tf, "bull": bull, "top": zt, "bottom": zb, "formed": z["formed"],
                                                     "original": [z["bottom"], z["top"]], "flagged": fvg.flagged_at(z, trig["time"] + 3600)}}, stop_override=stop)
                        if built and not _dup(out, built):
                            out.append(built)
    return out


def _find_fvg_cplocs() -> list[dict]:
    """The 15M version of a continuation: a 15M CPLOC rejects a valid 4H LEVEL while an unfilled FVG (1H, 4H or Daily) of the same direction is in place
    where the candle went: a bearish FVG for a sell, a bullish one for a buy. The CPLOC candle pushed into the zone and closed back OUTSIDE it, so the
    trend (the FVG's direction) continues. The entry is the 15M candle, the stop and entry are the usual rules, the target is the FVG target (see _fvg_target)."""
    h1 = data.get_candles("1h")
    h1_times = [c["time"] for c in h1]
    m15 = data.get_candles("15m")
    index_of = {c["time"]: i for i, c in enumerate(m15)}
    book0 = LevelBook(data.get_candles("1d"), "1d")
    out: list[dict] = []
    for f in frc_levels.find("15m"):
        i = index_of.get(f["time"])
        if i is None:
            continue
        c, bull = m15[i], f["direction"] == "bull"
        known = c["time"] + 900
        zones = []
        for z in fvg.active_at(known):
            if z["bull"] != bull:
                continue
            zb, zt = fvg.bounds_at(z, c["time"])               # only the part of the gap price has not already traded through counts
            if c["high"] >= zb and c["low"] <= zt and ((c["close"] > zt) if bull else (c["close"] < zb)):
                zones.append({**z, "bottom": zb, "top": zt})
        if not zones:
            continue
        if fvg.opposing_signal(known, bull, c["close"], 900):
            continue                                                            # a higher-timeframe (1H / 4H / Daily) FVG signal the other way is in play
        z = max(zones, key=lambda x: TF_SECONDS[x["tf"]])                       # the biggest timeframe among those the candle rejected
        r = bisect_right(h1_times, c["time"]) - 2                               # the last COMPLETED 1H candle at that moment
        target = _fvg_target(h1, h1_times, z, r + 1, bull, c["close"], c["time"])
        if target is None or _is_repeat(out, f["direction"], target, c["time"]):
            continue
        lv = f["levels"][0]
        built = _build("fvg", "15m", "15m", "4h", m15, i, f["direction"], z["top"] if bull else z["bottom"], target, _FixedBook(book0, target),
                       {"rule": "fvg", "levels_crossed": f["levels"],
                        "fvg_zone": {"tf": z["tf"], "bull": bull, "top": z["top"], "bottom": z["bottom"], "formed": z["formed"], "flagged": fvg.flagged_at(z, known)}},
                       stop_override=(c["low"] if bull else c["high"]))        # the CPLOC candle is the tap candle: its own extreme is the stop
        if built:
            built["evidence"]["why"].insert(1, f"A 15M CPLOC crossed the valid 4H level {lv['price']} (a {'buying' if bull else 'selling'} level) at the same place: that is a continuation signal, "
                                               "not a reversal, because an unfilled FVG of the same direction was there.")
            out.append(built)
    return out


def _soft(fn) -> list[dict]:
    """Run an optional finder; if its data is not available yet (monthly candles on an EA that has not been updated), it finds nothing."""
    try:
        return fn()
    except RuntimeError:
        return []


_FIND_CACHE: dict = {"at": 0.0, "key": None, "out": []}
_FIND_LOCK = threading.Lock()


def find() -> list[dict]:
    """All setups. Several endpoints (cards, chart, history sync, chat) ask at the same moment: they share one calculation, reused for
    2 seconds while the data and the user's levels / corrections are unchanged."""
    import copy
    with _FIND_LOCK:
        key = (data.active_source(), tuple(((data.try_candles(tf) or [{}])[-1].get("time"), len(data.try_candles(tf) or [])) for tf in ("15m", "1h", "4h", "1d")),
               json.dumps(levelsmod.custom_levels(), sort_keys=True), json.dumps(store.get_overrides(), sort_keys=True), levelsmod._MODE)
        if key == _FIND_CACHE["key"] and time.time() - _FIND_CACHE["at"] < 2:
            return copy.deepcopy(_FIND_CACHE["out"])
        out = _find_uncached()
        _FIND_CACHE.update(at=time.time(), key=key, out=out)
        return copy.deepcopy(out)


_CPLOC_CACHE: dict = {}


def _cploc_set(tf: str) -> set[tuple[int, str]]:
    """Every CPLOC candle of this timeframe: {(candle time, "bull" | "bear")}."""
    cs = data.get_candles(tf)
    key = (tf, len(cs), cs[-1]["time"], cs[-1]["close"])
    if _CPLOC_CACHE.get("key") != key or tf not in _CPLOC_CACHE:
        _CPLOC_CACHE.clear()
        _CPLOC_CACHE["key"] = key
    if tf not in _CPLOC_CACHE:
        _CPLOC_CACHE[tf] = {(f["time"], f["direction"]) for f in patterns.find_frc(cs, False, lambda i, d: [{"x": 1}])}
    return _CPLOC_CACHE[tf]


def _apply_busted(setups: list[dict]) -> list[dict]:
    """A BUSTED level takes no more trades. When a setup on a level is stopped out by a candle that is a CPLOC the OTHER way (a bearish CPLOC for a buy), the level
    has been busted: from the end of that candle on, no setup of any kind or direction is taken on that level any more. Setups are walked in time order, and
    only the ones that are actually taken can bust a level. FVG trades and your own setups are not tied to a level, so they are left alone."""
    busted: dict[float, int] = {}
    out = []
    for s in sorted(setups, key=lambda x: x["time"]):
        if s["tier"] in ("fvg", "manual") or s.get("origin") is None:
            out.append(s)
            continue
        key = round(s["origin"], 2)
        if key in busted and s["time"] >= busted[key]:
            continue
        out.append(s)
        p = s.get("plan")
        if p and p.get("outcome") == "stopped" and p.get("resolved_time"):
            opposite = "bear" if s["direction"] == "bull" else "bull"
            if (p["resolved_time"], opposite) in _cploc_set(s["trigger_tf"]):
                busted[key] = min(busted.get(key, 10 ** 12), p["resolved_time"] + TF_SEC[s["trigger_tf"]])
    return out


def _one_trade_per_candle(setups: list[dict]) -> list[dict]:
    """Weekly and Monthly levels are TARGETS, not different setups. When the same entry candle gives a Daily-tier and a Weekly- or Monthly-tier
    setup (a Daily / 4H / 1H setup that happens where a higher-timeframe level was reached), it is ONE trade: it keeps the higher tier, i.e. the
    higher level as its target. With no higher level involved the Daily target stays."""
    rank = {"monthly": 3, "weekly": 2, "daily": 1}
    best: dict[tuple, dict] = {}
    for s in setups:
        if s["tier"] in rank:
            k = (s["trigger_tf"], s["time"], s["direction"])
            if k not in best or rank[s["tier"]] > rank[best[k]["tier"]]:
                best[k] = s
    return [s for s in setups if s["tier"] not in rank or best[(s["trigger_tf"], s["time"], s["direction"])] is s]


def _find_uncached() -> list[dict]:
    global _OVERRIDES
    _OVERRIDES = store.get_overrides()
    reversal = _find_reversals()
    # The setup is always the Daily one: 1H CPLOC on a Daily level, Daily level + 4H level + a 1H close beyond it, or the early 15M CPLOC on the 4H level.
    # Weekly and Monthly levels only decide the TARGET: where one is involved the same setup is upgraded to it (see _upgrade_targets).
    daily = reversal + _find_gated("daily", "early", "15m", "1d") + _find_hold("daily", "1d", "4h", "1h")
    out = (
        daily
        + _soft(lambda: _upgrade_targets(daily))
        + _find_continuations(reversal)
        + _find_fvg_continuations()
        + _find_fvg_cplocs()
        + _find_level_reversals("weekly", "4h", "1w") + _find_level_holds("weekly", "4h", "1d", "1w")           # the Weekly level held: a 4H candle shows it
        + _soft(lambda: _find_level_reversals("monthly", "1d", "1mo")) + _soft(lambda: _find_level_holds("monthly", "1d", "1w", "1mo"))   # the Monthly level held: a Daily candle shows it
        + _find_manual()
    )
    out = _apply_busted(out)
    out = _one_trade_per_candle(out)
    out.sort(key=lambda s: s["time"])
    _apply_weekly_conflicts(out)
    return out

