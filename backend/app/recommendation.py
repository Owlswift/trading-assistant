"""BUY / SELL / NO TRADE, decided by the engine (never by the LLM), plus the list of every live setup.

A trade is LISTED when its setup is not invalidated by a pending Weekly setup and its plan is
- "waiting" (the retrace entry has not filled yet) or "open" (filled, stop and target not hit), or
- finished (target hit / stopped / missed) less than an hour ago. It then drops off the list: the chart keeps
  drawing the setup, but the list is for what still matters now.

The HEADLINE is the primary live trade: the highest tier first (Weekly before Daily), then the most recent.
If other live trades point the other way they stay in the list and the headline says so.
NO TRADE when nothing is live: no CPLOC trigger has a plan that is still waiting or open.

Each plan comes from a trigger candle (the entry candle): a 1H CPLOC at a Daily level, a 15M CPLOC at a 4H
level (early / continuation), or the Weekly-tier triggers. Entry, stop and target are in reversals._plan.
"""
import datetime as dt
from bisect import bisect_left

KEEP_FINISHED_SECONDS = 3600
TIER_RANK = {"monthly": 3, "weekly": 2, "daily": 1, "fvg": 0.5}      # your own setups rank 0: they never lead the headline
LEVEL_NAME = {"1d": "Daily", "1w": "Weekly", "1mo": "Monthly"}
FEED_TF_SEC = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

KIND_LABEL = {
    ("daily", "reversal"): "1H CPLOC on a Daily level",
    ("daily", "early"): "15M CPLOC on a 4H level (early signal)",
    ("daily", "continuation"): "15M CPLOC on a 4H level (continuation after a 1H BOS)",
    ("weekly", "reversal"): "1H CPLOC on a Daily level (Weekly reversal)",
    ("monthly", "reversal"): "1H CPLOC on a Daily level (Monthly reversal)",
    ("monthly", "early"): "15M CPLOC on a 4H level (Monthly early signal)",
    ("weekly", "early"): "15M CPLOC on a 4H level (Weekly early signal)",
    ("manual", "manual"): "Your own setup",
    ("weekly", "level-reversal"): "4H CPLOC on a Weekly level (Weekly reversal)",
    ("weekly", "level-hold"): "4H candle closed beyond a Daily level inside a Weekly level (Weekly reversal)",
    ("monthly", "level-reversal"): "Daily CPLOC on a Monthly level (Monthly reversal)",
    ("monthly", "level-hold"): "Daily candle closed beyond a Weekly level inside a Monthly level (Monthly reversal)",
    ("fvg", "1h"): "1H candle rejected from a 1H FVG (trend continuation)",
    ("fvg", "4h"): "4H candle rejected from a 4H FVG (trend continuation)",
    ("fvg", "1d"): "Daily candle rejected from a Daily FVG (trend continuation)",
    ("fvg", "15m"): "15M CPLOC on a 4H level at an FVG (trend continuation)",
    ("daily", "hold"): "1H reversal candle over a 4H level (Daily level hold)",
    ("weekly", "hold"): "4H reversal confirmed, 1H entry (Weekly level hold)",
    ("monthly", "hold"): "Daily reversal confirmed, 1H entry (Monthly level hold)",
}
SHORT_LABEL = {
    ("daily", "reversal"): "1H CPLOC",
    ("daily", "early"): "15M CPLOC · early",
    ("daily", "continuation"): "15M CPLOC · cont.",
    ("weekly", "reversal"): "1H CPLOC · Weekly",
    ("monthly", "reversal"): "1H CPLOC · Monthly",
    ("monthly", "early"): "15M CPLOC · Monthly early",
    ("weekly", "early"): "15M CPLOC · Weekly early",
    ("manual", "manual"): "Your setup",
    ("weekly", "level-reversal"): "4H CPLOC · Weekly level",
    ("weekly", "level-hold"): "4H candle · Weekly level",
    ("monthly", "level-reversal"): "Daily CPLOC · Monthly level",
    ("monthly", "level-hold"): "Daily candle · Monthly level",
    ("fvg", "1h"): "1H FVG rejection",
    ("fvg", "4h"): "4H FVG rejection",
    ("fvg", "1d"): "Daily FVG rejection",
    ("fvg", "15m"): "15M CPLOC at an FVG",
    ("daily", "hold"): "1H candle · Daily hold",
    ("weekly", "hold"): "1H candle · Weekly hold",
    ("monthly", "hold"): "1H candle · Monthly hold",
}


def _fmt(t: int | None) -> str | None:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%m-%d %H:%M") if t else None


def _finish_time(s: dict, m5: list[dict], m5_times: list[int]) -> int:
    """When the plan actually finished: the first 5m candle inside the resolving candle that touched the level."""
    p, bull = s["plan"], s["direction"] == "bull"
    t0 = p["resolved_time"]
    tf = FEED_TF_SEC[s["trigger_tf"]]
    for c in m5[bisect_left(m5_times, t0):]:
        if c["time"] >= t0 + tf:
            break
        if p["outcome"] == "stopped":
            touched = c["low"] <= p["stop"] if bull else c["high"] >= p["stop"]
        else:  # target hit, or the target ran away before the retrace entry (missed)
            touched = c["high"] >= p["target"] if bull else c["low"] <= p["target"]
        if touched:
            return c["time"]
    return t0 + tf   # 5m data does not reach back that far: the end of the resolving candle


def _trade(s: dict, state: str, finished_at: int | None = None) -> dict:
    p = s["plan"]
    return {
        "id": s["id"],
        "direction": "BUY" if s["direction"] == "bull" else "SELL",
        "state": state,   # waiting | open | target | stopped | missed
        "tier": s["tier"], "kind": s["kind"],
        "label": KIND_LABEL[(s["tier"], s["kind"])], "short": SHORT_LABEL[(s["tier"], s["kind"])],
        "custom_label": s.get("label_text") or None, "custom_level": s.get("custom_level"), "fvg": s.get("fvg"), "partial": (s.get("plan") or {}).get("partial"), "fvg_exit": (s.get("plan") or {}).get("exit"), "manual": s["tier"] == "manual",
        "trigger_tf": s["trigger_tf"], "trigger_time": s["time"], "trigger_time_text": _fmt(s["time"]),
        "origin_level": s["origin"], "level_tf": s["level_tf"],
        "entry": p["entry"], "stop": p["stop"], "target": p["target"], "rr": p["rr"],
        "entry_mode": p["mode"], "fill_time": p["fill_time"], "fill_time_text": _fmt(p["fill_time"]),
        "finished_at": finished_at, "finished_at_text": _fmt(finished_at),
        "htf_conflict": s.get("htf_conflict") if (s.get("htf_conflict") or {}).get("now_passes") else None,
        "next_target": None,   # filled in by the caller: where the target moves at the next rollover
    }


def recommend(setups: list[dict], price: float, now: int, m5: list[dict]) -> dict:
    """`now` and the candles share the feed's clock (MT5 broker time or UTC)."""
    m5_times = [c["time"] for c in m5]
    live, finished = [], []
    for s in setups:
        p = s["plan"]
        if s["invalid"] or not p:
            continue
        if p["status"] in ("waiting", "open"):
            live.append(s)
        elif p["outcome"] in ("target", "stopped", "missed") and p["resolved_time"]:
            done_at = _finish_time(s, m5, m5_times)
            if now - done_at < KEEP_FINISHED_SECONDS:
                finished.append((s, done_at))

    # highest timeframe first (Monthly, Weekly, Daily), your own setups last; newest first within each
    live.sort(key=lambda s: (TIER_RANK.get(s["tier"], 0), s["time"]), reverse=True)
    engine_live = [s for s in live if s["tier"] != "manual"]   # the headline follows the ENGINE; your setups are listed
    trades = [_trade(s, s["plan"]["status"]) for s in live]
    trades += [_trade(s, s["plan"]["outcome"], t) for s, t in sorted(finished, key=lambda x: x[1], reverse=True)]

    if not engine_live:
        reasons = ["No CPLOC trigger has a live trade plan right now."]
        if live:
            reasons.append("Your own setup(s) below are live, but they are not engine triggers.")
        for s in list(reversed(setups))[:3]:
            p = s["plan"]
            why = (
                f"invalid ({s['invalid']['reason']})" if s["invalid"]
                else "no valid plan (target already behind the entry, or fewer than two candles before it)" if not p
                else f"plan {p['outcome']}" if p["outcome"] else s["status"]
            )
            reasons.append(f"Latest {KIND_LABEL[(s['tier'], s['kind'])]} at {_fmt(s['time'])} ({s['direction']}): {why}.")
        return {"direction": "NO TRADE", "price": price, "reasons": reasons, "trades": trades, "mixed": False}

    best = engine_live[0]
    p = best["plan"]
    bull = best["direction"] == "bull"
    mixed = any(s["direction"] != best["direction"] for s in live)   # your own setups count: they may go against it
    if p["status"] == "waiting":
        state = "waiting"
        note = f"Wait for price to retrace to {p['entry']:.2f} (the range of the second candle before the trigger), then enter."
    else:
        state = "open"
        move = (price - p["entry"]) if bull else (p["entry"] - price)
        note = (
            f"Entry filled at {p['entry']:.2f} ({'as the trigger candle closed' if p['mode'] == 'immediate' else 'on the retrace'}, "
            f"{_fmt(p['fill_time'])}). Price is {move:+.2f} from entry."
        )
    reasons = [
        f"{KIND_LABEL[(best['tier'], best['kind'])]} at {_fmt(best['time'])} formed on the {best['origin']:.2f} level.",
        note,
        f"Stop {p['stop']:.2f} = the {'lowest low' if bull else 'highest high'} of "
        + ("the tap candle (and the trigger candle if it is a different one). " if best["tier"] == "fvg" else "the trigger candle and the two before it. ")
        + (f"Target {p['target']:.2f} is the {'highest high' if bull else 'lowest low'} on the FVG's own timeframe from its last break of structure up to the tap (the peak the pullback into the FVG came from)."
           if best["tier"] == "fvg" else
           f"Target {p['target']:.2f} is capped at the near edge of an unfilled FVG; the next valid {LEVEL_NAME.get(best['level_tf'], 'Daily')} level beyond is {best['fvg']['level_target']:.2f}."
         if (best.get("fvg") or {}).get("cap_applied") else f"Target {p['target']:.2f} is the next valid {LEVEL_NAME.get(best['level_tf'], 'Daily')} level."),
    ]
    if mixed:
        reasons.append("Other live setups point the other way (counter-trend pullbacks): see the list.")
    return {
        "direction": "BUY" if bull else "SELL", "state": state, "price": price,
        "entry": p["entry"], "stop": p["stop"], "target": p["target"], "rr": p["rr"],
        "entry_mode": p["mode"], "reasons": reasons, "trades": trades, "mixed": mixed,
    }
