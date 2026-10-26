"""Deep-dive studies of one rule at a time (shown in the app's Deep-dive tab).

monthly: the Monthly level hold. A Weekly level lies inside the range of the Monthly candle that made a valid Monthly level, and a Daily candle crosses that
Weekly level and closes beyond it. The trades are split by HOW the Daily candle did it: a CPLOC across the Weekly level, or a plain close beyond it.
Every trade gets an explanation written by the engine from its own numbers (no LLM).
"""
import datetime as dt
from bisect import bisect_right

from . import data, patterns, reversals

SINCE = int(dt.datetime(2021, 1, 1, tzinfo=dt.UTC).timestamp())
TF_NAME = {"1d": "Daily", "1w": "Weekly", "1mo": "Monthly", "4h": "4H", "1h": "1H"}


def _d(t: int | None) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%a %d %b %Y") if t else "–"


def _month(t: int) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%B %Y")


def _result_r(p: dict) -> float | None:
    if p.get("status") == "skipped":
        return None
    if p.get("outcome") == "target":
        return p.get("result_r", p.get("rr_final", p["rr"]))
    if p.get("outcome") == "stopped":
        return p.get("result_r", -1.0)
    return None


def _candle_at(cs: list[dict], times: list[int], t: int) -> int | None:
    i = bisect_right(times, t) - 1
    return i if i >= 0 else None


def _move_line(s: dict, m: list[dict], m_times: list[int], d: list[dict], d_times: list[int]) -> str:
    bull = s["direction"] == "bull"
    i = _candle_at(d, d_times, s["time"])
    book = reversals.LevelBook(m, "1mo")
    up, down = book.valid_lists(s["time"])
    kind = [lv for lv in (down if bull else up) if book.origin_ok(lv, bull)]
    far, lo, hi = reversals._furthest_level(d, i, bull, kind, m, m_times)
    crossed = sorted([lv for lv in kind if lo <= lv <= hi], reverse=bull)
    return (f"The move: price ran {'down from ' + str(hi) + ' to ' + str(lo) if bull else 'up from ' + str(lo) + ' to ' + str(hi)} through the valid Monthly "
            f"{'buying' if bull else 'selling'} level(s) {crossed}. The setup is taken from the furthest one, {far}.")


def _explain(s: dict, m: list[dict], m_times: list[int], w: list[dict], d: list[dict], d_times: list[int], cploc: bool) -> list[str]:
    bull = s["direction"] == "bull"
    ev, p = s["evidence"], s["plan"]
    lines = []
    signal1 = s["kind"] == "level-reversal"
    # 1. the Monthly level
    L = ev.get("hit_level", s["origin"])
    im = _candle_at(m, m_times, s["time"])
    k = next((j for j in range(im - 1, -1, -1) if abs(m[j]["close"] - L) < 0.006), None) if im is not None else None
    if k is not None:
        mc, nx = m[k], (m[k + 1] if k + 1 < len(m) else None)
        lines.append(f"Monthly level {L}: the close of the {_month(mc['time'])} candle (O {mc['open']} H {mc['high']} L {mc['low']} C {mc['close']}). "
                     + (f"The next Monthly candle ({_month(nx['time'])}) closed {'up' if nx['dir'] == 'bull' else 'down'} at {nx['close']}, so it is a "
                        f"{'buying' if nx['dir'] == 'bull' else 'selling'} level." if nx else "")
                     + f" No Monthly close {'below' if bull else 'above'} it since, so it was still valid on {_d(s['time'])}.")
        rng = f"from its close {L} down to its low {mc['low']}" if bull else f"from its close {L} up to its high {mc['high']}"
    else:
        lines.append(f"Monthly level {L} (a level you marked, or a close the engine could not find again).")
        rng = f"within one Daily ATR of {L} (no candle to take a range from)"
    lines.append(_move_line(s, m, m_times, d, d_times))
    i = _candle_at(d, d_times, s["time"])
    c, prev = d[i], d[i - 1]
    body_lo, body_hi = min(prev["open"], prev["close"]), max(prev["open"], prev["close"])
    if signal1:
        lines.append(f"Signal 1, Daily candle {_d(c['time'])} (O {c['open']} H {c['high']} L {c['low']} C {c['close']}): a "
                     + (f"bullish CPLOC across the Monthly level: its low {c['low']} reached the previous candle's body low {body_lo} and it closed at {c['close']}, "
                        f"above that body's high {body_hi} and above {L}." if bull else
                        f"bearish CPLOC across the Monthly level: its high {c['high']} reached the previous candle's body high {body_hi} and it closed at {c['close']}, "
                        f"below that body's low {body_lo} and below {L}."))
        lines.append(f"Plan: entry {p['entry']} ({'at the CPLOC close' if p['mode'] == 'immediate' else 'on the retrace'}), stop {p['stop']} (the "
                     f"{'lowest low' if bull else 'highest high'} of the CPLOC and the two candles before it; only a Daily CLOSE beyond it stops the trade), "
                     f"target: {_target_text(s, bull)}.")
        return lines
    # 2. the Weekly level
    ll = ev.get("lower_level") or {}
    W = ll.get("price")
    lines.append(f"Weekly level {W}: the close of the week of {_d(ll.get('time'))}. It lies inside the Monthly candle's range, {rng}.")
    # 3. the Daily candle
    colour = "green (close above its open)" if c["close"] > c["open"] else "red (close below its open)"
    if cploc:
        how = (f"a bullish CPLOC: its low {c['low']} reached the previous candle's body low {body_lo} and it closed at {c['close']}, above that body's high {body_hi}"
               if bull else
               f"a bearish CPLOC: its high {c['high']} reached the previous candle's body high {body_hi} and it closed at {c['close']}, below that body's low {body_lo}")
    else:
        via = (f"the previous Daily close {prev['close']} was {'below' if bull else 'above'} it" if ((prev["close"] <= W) if bull else (prev["close"] >= W))
               else f"its wick reached it ({'low' if bull else 'high'} {c['low'] if bull else c['high']})")
        how = f"not a CPLOC: it closed {'above' if bull else 'below'} the Weekly level at {c['close']}; it crossed it because {via}"
    lines.append(f"Daily candle {_d(c['time'])} (O {c['open']} H {c['high']} L {c['low']} C {c['close']}), {colour}: {how}.")
    # 4. the plan
    tgt = _target_text(s, bull)
    lines.append(f"Plan: entry {p['entry']} (the Daily close), stop {p['stop']} (the Daily candle's {'low' if bull else 'high'}), target: {tgt}.")
    return lines


def _target_text(s: dict, bull: bool) -> str:
    return f"the next valid Monthly {'selling' if bull else 'buying'} level {_monthly_target(s)}"


def _monthly_target(s: dict) -> float:
    f = s.get("fvg") or {}
    return f["level_target"] if f.get("cap_applied") and f.get("level_target") is not None else s["initial_target"]


def _run(d: list[dict], j0: int, bull: bool, entry: float, stop: float, target: float, close_stop: bool, zones: list[dict], fvg_exit: bool, what: str,
         cplocs: set | None = None) -> dict:
    """Walk the Daily candles from j0 until the stop, the target or (with fvg_exit) a Daily FVG rejection against the trade. In one candle the stop counts
    first, then the target, then the exit (it happens at the close). R:R is judged once, at the start: below 1 the trade is let go."""
    from . import fvg
    risk = (entry - stop) if bull else (stop - entry)
    rr = round(((target - entry) if bull else (entry - target)) / risk, 2) if risk > 0 else 0
    out = {"entry": entry, "stop": stop, "target": target, "rr": rr}
    if risk <= 0 or rr < 1:
        return {**out, "result": "let go", "r": None, "line": f"Let go: at the start the R:R to {what} {target} was {rr}, below 1."}
    opp = [z for z in zones if z["bull"] != bull]
    last = len(d) - 1
    for j in range(j0, last):
        c = d[j]
        opp_cploc = cplocs is not None and (c["time"], "bear" if bull else "bull") in cplocs and ((c["low"] < stop) if bull else (c["high"] > stop))
        if opp_cploc:                    # an opposite CPLOC whose wick takes the stop: stopped, even with a close-based stop
            return {**out, "result": "lost", "r": -1.0, "end": c["time"], "outcome": "stopped", "by_cploc": True,
                    "line": f"Lost: the {'bearish' if bull else 'bullish'} CPLOC of {_d(c['time'])} took the stop {stop} ({'low' if bull else 'high'} "
                            f"{c['low'] if bull else c['high']}) (−1R)."}
        if (c["close"] < stop if bull else c["close"] > stop) if close_stop else (c["low"] < stop if bull else c["high"] > stop):
            return {**out, "result": "lost", "r": -1.0, "end": c["time"], "outcome": "stopped",
                    "line": f"Lost: the stop {stop} was {'closed through' if close_stop else 'hit'} on {_d(c['time'])} (−1R)."}
        if (c["high"] >= target) if bull else (c["low"] <= target):
            return {**out, "result": "won", "r": rr, "end": c["time"], "outcome": "target", "line": f"Won: {what} {target} was reached on {_d(c['time'])} (+{rr}R)."}
        if not fvg_exit:
            continue
        for z in opp:
            if z["created"] > c["time"] or (z["end"] is not None and z["end"] <= c["time"]):
                continue
            for k in range(max(j0, j - 3), j + 1):
                if z["created"] <= d[k]["time"] and fvg.find_rejection(z, d, k, last) == j:
                    zb, zt = fvg.bounds_at(z, d[k]["time"])
                    r = round(((c["close"] - entry) if bull else (entry - c["close"])) / risk, 2)
                    return {**out, "result": "won" if r > 0 else "lost", "exit": True, "r": r, "g": j, "k": k, "zone": z, "end": c["time"], "outcome": "exited",
                            "exit_price": c["close"],
                            "line": (f"Exited on {_d(c['time'])} at {c['close']} ({r:+.2f}R, the stop was not hit): price got into the Daily "
                                     f"{'bearish' if bull else 'bullish'} FVG {zb}-{zt} on {_d(d[k]['time'])} and the Daily candle of {_d(c['time'])} rejected it, "
                                     f"closing {'red below' if bull else 'green above'} it.")}
    return {**out, "result": "open", "r": None, "line": "Still open."}


def _result_line(p: dict) -> str:
    if p.get("status") == "skipped":
        return f"Let go: R:R {p['rr']} is below 1 (it would have {p.get('would_be') or 'still been open'})."
    if p.get("outcome") == "target":
        return f"Won: the target was reached on {_d(p.get('resolved_time'))} ({_result_r(p):+.2f}R)."
    if p.get("outcome") == "stopped":
        return f"Lost: the stop was hit on {_d(p.get('resolved_time'))} (−1R)."
    return "Still open."


def _cap_zone(f: dict, bull: bool) -> dict | None:
    """The FVG whose near edge is the cap (the bottom of a bearish gap for a buy, the top of a bullish one for a sell)."""
    return next((z for z in f.get("zones") or [] if abs((z["bottom"] if bull else z["top"]) - f["cap"]) < 0.006), None)


# ---- every version's setups, each with its own id and explanation (the walkthrough explains them like any setup: see record()) ----
_RECORDS: dict[str, dict] = {}
GROUP = {"signal1": "Signal 1: Daily CPLOC across the Monthly level", "cploc": "Signal 2: Daily CPLOC across a Weekly level at or beyond it",
         "close": "Signal 2: Daily close beyond a Weekly level at or beyond it (not a CPLOC)", "other": "The trade the other way, from the Daily FVG rejection that cut a Monthly trade short",
         "follow": "Follow the trend: the Monthly level was busted by an opposite Daily CPLOC"}
VERSION = {"a": "A: today's engine", "b": "B: the full Monthly target, exit on a Daily FVG rejection",
           "c": "C: B + the trade the other way, aiming at the next Monthly level", "d": "D: B + the trade the other way, aiming at the FVG target"}


def record(setup_id: str) -> dict | None:
    """A Deep-dive setup in the shape of a stored one ({"record": ...}), so the chart walkthrough can explain it."""
    if setup_id not in _RECORDS:
        try:
            monthly()
        except Exception:
            return None
    r = _RECORDS.get(setup_id)
    return {"record": r, "id": setup_id, "source": data.active_source()} if r else None


def _ohlc(c: dict) -> dict:
    return {"time": c["time"], "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"]}


def _row(sid: str, version: str, group: str, rec: dict, res: str, r: float | None, steps: dict, end: int | None) -> dict:
    p = rec["plan"]
    return {"id": sid, "version": version, "group": group, "time": rec["time"], "date": _d(rec["time"]), "year": dt.datetime.fromtimestamp(rec["time"], dt.UTC).year,
            "direction": "BUY" if rec["direction"] == "bull" else "SELL", "level": rec.get("origin"), "entry": p["entry"], "stop": p["stop"], "target": p["target"],
            "rr": p["rr"], "result": res, "result_r": r, "end": _d(end) if end else None, "explain": steps["level"] + steps["trigger"] + steps["plan"]}


def _register(sid: str, base: dict, plan: dict, steps: dict, decision: str, reason: str, **extra) -> dict:
    ev = {**(base.get("evidence") or {}), **extra.pop("evidence", {}), "why": steps["level"] + steps["trigger"] + steps["plan"], "study_steps": steps}
    rec = {**base, **extra, "id": sid, "plan": plan, "target": plan["target"], "initial_target": plan["initial_target"], "evidence": ev,
           "resolved_time": plan.get("resolved_time"), "status": plan.get("status", "done"), "invalid": None, "fvg": None,
           "study_decision": {"decision": decision, "reason": reason}, "target_history": [{"time": base["time"], "target": plan["target"]}]}
    _RECORDS[sid] = rec
    return rec


def _version_plan(p: dict, res: dict, mode: str) -> dict:
    return {"mode": mode, "entry": res["entry"], "stop": res["stop"], "target": res["target"], "initial_target": res["target"], "rr": res["rr"],
            "fill_time": p.get("fill_time"), "resolved_time": res.get("end"), "outcome": None if res["result"] in ("open",) else
            ("skipped" if res["result"] == "let go" else res.get("outcome")), "status": "skipped" if res["result"] == "let go" else "done",
            "range_high": p.get("range_high"), "range_low": p.get("range_low"), "exit_price": res.get("exit_price")}


def _other_way(s: dict, b: dict, d: list[dict], d_times: list[int], zones: list[dict], m: list[dict], v: str) -> tuple[dict, dict] | None:
    """C / D: the trade the other way from the Daily FVG rejection that closed the B trade."""
    bull = s["direction"] == "bull"
    g, k, z = b["g"], b["k"], b["zone"]
    rb = not bull
    entry = d[g]["close"]
    span = d[k:g + 1]
    stop = max(x["high"] for x in span) if bull else min(x["low"] for x in span)
    from . import fvg
    zb, zt = fvg.bounds_at(z, d[k]["time"])
    if v == "c":
        tgt = reversals.LevelBook(m, "1mo").next(d[g]["time"], entry, rb)
        what, how = "the next Monthly level", "the next valid Monthly level"
        res = _run(d, g + 1, rb, entry, stop, tgt, False, zones, True, what) if tgt is not None else None
    else:
        tgt = reversals._fvg_target(d, d_times, z, k, rb, entry, d[k]["time"])
        what, how = "the FVG target", "the FVG target (the extreme on the Daily chart from its last break of structure up to the tap)"
        res = _run(d, g + 1, rb, entry, stop, tgt, False, zones, False, what) if tgt is not None else None
    side = "SELL" if bull else "BUY"
    mb = reversals.LevelBook(m, "1mo")
    up, down = mb.valid_lists(d[g]["time"])
    near = ([lv for lv in up if mb.origin_ok(lv, False) and lv >= d[g]["close"]][:1] if bull else
            [lv for lv in down if mb.origin_ok(lv, True) and lv <= d[g]["close"]][-1:])
    src = {c["close"]: _month(c["time"]) for c in m}
    hi_lo = max(x["high"] for x in span) if bull else min(x["low"] for x in span)
    # why it is in the Monthly deep-dive: this rejection cut short a Monthly trade that was in action
    mline = (f"This rejection interrupted the Monthly {'BUY' if bull else 'SELL'} of {_d(s['time'])} on the Monthly level "
             f"{s['evidence'].get('hit_level', s['origin'])}: that trade was exited here (B), and this {side} is taken from the same candle."
             + (f" The rejection was also at the Monthly {'selling' if bull else 'buying'} level {near[0]} ({src.get(near[0], '')})."
                if near and (near[0] <= hi_lo if bull else near[0] >= hi_lo) else ""))
    steps = {
        "level": [f"The Daily {'bearish' if bull else 'bullish'} FVG {zb}-{zt} (formed {_d(z['formed'])}): a {'downtrend' if bull else 'uptrend'} gap. "
                  f"On {_d(d[k]['time'])} price got into it.", mline],
        "trigger": [f"The Daily candle of {_d(d[g]['time'])} (O {d[g]['open']} H {d[g]['high']} L {d[g]['low']} C {d[g]['close']}) rejected it: it closed "
                    f"{'red below' if bull else 'green above'} the gap. That closed the {'buy' if bull else 'sell'} (B) and is the {side} here."],
        "plan": [f"Plan: {side}, entry {entry} (the rejection candle's close), stop {stop} (the {'high' if bull else 'low'} of the tap and rejection candles), "
                 + (f"target {how} {tgt}." if tgt is not None else
                    ("no target: no valid Monthly level beyond the entry." if v == "c" else "no FVG target: the Daily trend was not the gap's way, or the swing was not beyond the entry."))],
    }
    if res is None:
        return None
    steps["plan"].append(f"R:R at the start {res['rr']}. " + res["line"])
    sid = f"study-monthly-{v}-other-{s['id']}"
    plan = _version_plan({}, res, "immediate")
    plan["fill_time"] = d[g]["time"] + 86400
    rec = _register(sid, {"time": d[g]["time"], "direction": "bear" if bull else "bull", "tier": "fvg", "kind": "1d", "trigger_tf": "1d", "level_tf": "1d",
                          "origin": zb if bull else zt, "low": d[g]["low"], "high": d[g]["high"], "evidence": {}}, plan, steps,
                    "let go" if res["result"] == "let go" else "taken", res["line"],
                    evidence={"rule": "fvg", "fvg_zone": {"tf": "1d", "bull": z["bull"], "top": zt, "bottom": zb, "formed": z["formed"], "flagged": False},
                              "tap": _ohlc(d[k]), "trigger": _ohlc(d[g]), "prev1": _ohlc(d[g - 1]), "prev2": _ohlc(d[g - 2])})
    return rec, res


def _follow(s: dict, b: dict, L: float, cp: int, je: int, d: list[dict], d_times: list[int], zones: list[dict], m: list[dict], close_stop: bool, cpl: set | None = None):
    """The level failed: an opposite Daily CPLOC crossed back through it while the trade was open, and the trade was then stopped. Price is expected to go
    to the next level, so follow it: a trade the other way, entered when the stop-out is KNOWN (the close that broke a close-based stop, or the stop price
    for a wick stop), stop by the CPLOC rules (the extreme of that CPLOC and the two Daily candles before it, only a Daily close beyond it), target the next
    valid Monthly level that way. The stop is the extreme of the CPLOC candle that busted the level. R:R judged at the start; the same Daily FVG exit as B."""
    bull = s["direction"] == "bull"
    rb = not bull
    if je + 1 >= len(d) - 1:
        return None
    entry = d[je + 1]["open"]                                   # the open of the candle after the one that stopped the trade (and busted the level)
    stop = d[cp]["high"] if bull else d[cp]["low"]              # the extreme of the CPLOC candle that busted the level
    tgt = reversals.LevelBook(m, "1mo").next(d[je]["time"], entry, rb)
    c = d[cp]
    side = "SELL" if bull else "BUY"
    steps = {"level": [f"The Monthly {'buying' if bull else 'selling'} level {L} failed: the {'BUY' if bull else 'SELL'} of {_d(s['time'])} on it was open when the "
                       f"Daily candle of {_d(c['time'])} (O {c['open']} H {c['high']} L {c['low']} C {c['close']}), a {'bearish' if bull else 'bullish'} CPLOC, crossed "
                       f"back {'below' if bull else 'above'} it; " + ("that same candle took its stop " + str(s['plan']['stop']) if je == cp else "that trade was then stopped on " + _d(d[je]['time'])) + ". The level is busted: no more trades on it, and price is "
                       f"expected to go to the next level."],
             "trigger": [f"{side} at the open of {_d(d[je + 1]['time'])}, {entry}: the candle after the one that stopped the {'BUY' if bull else 'SELL'} and busted the level."]}
    if tgt is None:
        return None
    res = _run(d, je + 1, rb, entry, stop, tgt, True, zones, True, "the next Monthly level", cpl)
    steps["plan"] = [f"Plan: {side}, entry {entry}, stop {stop} (the {'high' if bull else 'low'} of the CPLOC candle of {_d(c['time'])} that busted the level; only a "
                     f"Daily close beyond it stops the trade), target the next valid Monthly {'buying' if bull else 'selling'} level {tgt}.",
                     f"R:R at the start {res['rr']}. " + res["line"]]
    plan = _version_plan({}, res, "immediate")
    plan["fill_time"] = d[je + 1]["time"]
    base = {"time": d[je]["time"], "direction": "bear" if bull else "bull", "tier": "monthly", "kind": "level-reversal", "trigger_tf": "1d", "level_tf": "1mo",
            "origin": L, "low": c["low"], "high": c["high"]}
    ev = {"rule": "follow", "hit_level": L, "hit_level_tf": "1mo", "trigger": _ohlc(d[je]), "prev1": _ohlc(d[je - 1]), "prev2": _ohlc(d[je - 2]),
          "levels_crossed": [{"tf": "1mo", "price": L}]}
    return {"base": base, "plan": plan, "steps": steps, "evidence": ev}, res


def monthly() -> dict:
    m, w, d = data.get_candles("1mo"), data.get_candles("1w"), data.get_candles("1d")
    m_times, d_times = [c["time"] for c in m], [c["time"] for c in d]
    cplocs = {(f["time"], f["direction"]) for f in patterns.find_frc(d)}
    from . import fvg
    dz = fvg.zones("1d")
    rows: dict[str, list[dict]] = {v: [] for v in "abcd"}
    busted: dict[float, tuple[int, str]] = {}             # B / C / D: Monthly level -> (busted from, by which trade)
    for s in reversals.find():
        if s["tier"] != "monthly" or s["kind"] not in ("level-hold", "level-reversal") or not s.get("plan") or s["time"] < SINCE:
            continue
        p, bull = s["plan"], s["direction"] == "bull"
        grp = "signal1" if s["kind"] == "level-reversal" else "cploc" if (s["time"], s["direction"]) in cplocs else "close"
        lines = _explain(s, m, m_times, w, d, d_times, grp == "cploc")
        base_steps = {"level": lines[:2], "trigger": lines[2:-1]}
        plan_line = lines[-1]
        # A: today's engine, as it is
        f = s.get("fvg") or {}
        a_res = "let go" if p.get("status") == "skipped" else "won" if p.get("outcome") == "target" else "lost" if p.get("outcome") == "stopped" else "open"
        a_plan = plan_line.replace(f"target: {_target_text(s, bull)}.", f"target {s['initial_target']}" + (
            f" (today's engine cut it from the Monthly level {f['level_target']} because an FVG was in the way)." if f.get("cap_applied") else "."))
        a_steps = {**base_steps, "plan": [a_plan, f"R:R {p['rr']}. " + _result_line(p)]}
        sid = f"study-monthly-a-{s['id']}"
        _register(sid, s, {**p}, a_steps, "let go" if a_res == "let go" else "taken", a_steps["plan"][-1])
        rows["a"].append(_row(sid, "a", grp, _RECORDS[sid], a_res, _result_r(p), a_steps, p.get("resolved_time")))
        # B: the full Monthly target, exit on a Daily FVG rejection
        if not p.get("fill_time"):
            continue
        L = s["evidence"].get("hit_level", s["origin"])
        key = round(L, 2)
        if key in busted and s["time"] >= busted[key][0]:          # the level was busted by an opposite CPLOC: no more trades on it
            for v in "bcd":
                rows[v].append({**_row(f"study-monthly-{v}-{s['id']}", v, grp, {**s, "plan": p}, "dropped", None, {**base_steps, "plan": [
                    f"Not taken: the Monthly level {L} was busted on {_d(busted[key][0] - 86400)}, when {busted[key][1]} was stopped out after an opposite Daily CPLOC "
                    f"crossed back through it. No more trades on that level."]}, None)})
            continue
        i = bisect_right(d_times, s["time"]) - 1
        j0 = bisect_right(d_times, p["fill_time"] - 1) if p["mode"] != "immediate" else i + 1
        b = _run(d, j0, bull, p["entry"], p["stop"], _monthly_target(s), s["kind"] == "level-reversal", dz, True, "the Monthly target", cplocs)
        b_steps = {**base_steps, "plan": [plan_line, f"R:R at the start {b['rr']}. " + b["line"]]}
        b_plan = _version_plan(p, b, p["mode"])
        follow = None
        if b.get("outcome") == "stopped":
            je = bisect_right(d_times, b["end"]) - 1
            opp = "bear" if bull else "bull"
            cp = next((j for j in range(j0, je + 1) if (d_times[j], opp) in cplocs
                       and ((d[j]["high"] >= L > d[j]["close"]) if bull else (d[j]["low"] <= L < d[j]["close"]))), None)
            if cp is not None:
                again = key in busted and busted[key][0] == d_times[je] + 86400      # another trade on the same level stopped by the same candle: one bust, one follow
                if not again:
                    busted[key] = (d_times[je] + 86400, s["id"])
                    follow = _follow(s, b, L, cp, je, d, d_times, dz, m, s["kind"] == "level-reversal", cplocs)
        for v in "bcd":                      # C and D keep the same B trade, plus the trade the other way
            sid = f"study-monthly-{v}-{s['id']}"
            _register(sid, s, dict(b_plan), b_steps, "let go" if b["result"] == "let go" else "taken", b["line"])
            rows[v].append(_row(sid, v, grp, _RECORDS[sid], b["result"], b["r"], b_steps, b.get("end")))
            if follow:
                fid = f"study-monthly-{v}-follow-{s['id']}"
                frec, fres = follow
                rec2 = _register(fid, frec["base"], dict(frec["plan"]), frec["steps"], "let go" if fres["result"] == "let go" else "taken", fres["line"], evidence=frec["evidence"])
                rows[v].append(_row(fid, v, "follow", rec2, fres["result"], fres["r"], frec["steps"], fres.get("end")))
            if v in "cd" and b.get("exit"):
                got = _other_way(s, b, d, d_times, dz, m, v)
                if got:
                    rec, res = got
                    rows[v].append(_row(rec["id"], v, "other", rec, res["result"], res["r"], rec["evidence"]["study_steps"], res.get("end")))
    versions = []
    for v in "abcd":
        groups = [{"key": k, "title": GROUP[k], **_summary(sorted([r for r in rows[v] if r["group"] == k], key=lambda r: r["time"])),
                   "trades": sorted([r for r in rows[v] if r["group"] == k], key=lambda r: r["time"])} for k in GROUP if any(r["group"] == k for r in rows[v])]
        versions.append({"key": v, "title": VERSION[v], "total": _summary(rows[v])["total"], "groups": groups})
    return {
        "title": "Monthly level hold",
        "rule": "A Monthly level holds when a Daily candle does a CPLOC across it (signal 1), or a Daily candle does a CPLOC across, or closes beyond, a Weekly "
                "level at or beyond it (signal 2). When price runs through several Monthly levels, the furthest one is used (the lowest for a buy, the highest "
                "for a sell).",
        "how_the_engine_applies_it": [
            "The move: from the start of the previous Monthly candle up to the Daily trigger, the lowest low (buy) / highest high (sell) and the opposite extreme "
            "before it. Of the valid Monthly levels of the trade's kind inside that move, the furthest one is the level; the signals are only checked there.",
            "Monthly level: a valid Monthly close (not the one just before the running month, not broken by a later Monthly close); its kind comes from the next Monthly candle (up = buying level, down = selling level).",
            "Signal 1: a Daily CPLOC across that Monthly level. Entry and stop by the CPLOC rules (stop = the extreme of the CPLOC and the two candles before it, "
            "only a Daily close beyond it stops the trade).",
            "Signal 2: a Weekly level inside the Monthly level candle's range (from its close out to its low for a buy, high for a sell), and a Daily candle that "
            "crosses it and closes beyond it, green for a buy, red for a sell. Entry at the Daily close, stop at its low (buy) / high (sell).",
            "R:R is judged once, at the start: below 1 the trade is let go, otherwise it is taken.",
            "A (today's engine): an opposing FVG between the entry and the target cuts the target to its near edge.",
            "B: the full Monthly target. When price gets into an opposing Daily FVG and a Daily candle rejects it (the first candle of the opposite colour closing "
            "beyond the gap: the tap candle or one of the next 3), the trade is exited at that close, in profit or at a loss, even though the stop was not hit.",
            "Only Daily FVGs that interfere with a Monthly trade in action are tracked here: the rejection that exits it (B), and the trade the other way taken "
            "from that rejection (C, D). A Daily FVG setup that does not interrupt a Monthly trade is not part of this deep-dive.",
            "C: B, plus the trade the other way from that rejection candle: entry its close, stop the extreme of the tap and rejection candles, target the next "
            "valid Monthly level, the same Daily FVG exit.",
            "D: as C, but the trade the other way aims at the FVG target and has no FVG exit (as every FVG trade).",
            "Busted level (B, C, D): when an opposite Daily CPLOC crosses back through the Monthly level while a trade on it is open, and that trade is then "
            "stopped, the level is busted: no more trades on it. An opposite CPLOC whose wick takes the stop stops the trade, even a close-based one. Follow the "
            "trend: a trade the other way at the open of the next candle, stop at the extreme of the CPLOC candle that busted the level, target the next valid "
            "Monthly level, the same Daily FVG exit.",
        ],
        "since": "2021-01-01",
        "versions": versions,
    }


def _summary(trades: list[dict]) -> dict:
    years: dict[int, dict] = {}
    for t in trades:
        y = years.setdefault(t["year"], {"year": t["year"], "won": 0, "lost": 0, "exited": 0, "let_go": 0, "open": 0, "net_r": 0.0})
        res = t["result"]
        if res in ("won", "lost", "let go", "open"):
            y[res.replace("let go", "let_go")] += 1
        if "Exited on" in " ".join(t["explain"][-1:]):
            y["exited"] += 1
        y["net_r"] += t["result_r"] or 0.0
    rows = sorted(years.values(), key=lambda r: r["year"])
    tot = {"year": "Total", **{k: sum(r[k] for r in rows) for k in ("won", "lost", "exited", "let_go", "open", "net_r")}}
    for r in rows + [tot]:
        n = r["won"] + r["lost"]
        r["trades"] = n
        r["win_rate"] = round(r["won"] / n, 3) if n else None
        r["net_r"] = round(r["net_r"], 2)
    return {"years": rows, "total": tot}


STUDIES = {"monthly": monthly}
