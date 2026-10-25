"""What the assistant can DO beyond chatting: look things up in the stored history, keep notes, write observations.

Learning here means memory, not retraining: the model's weights never change. It learns through
- lessons the user teaches (always followed once confirmed),
- observations it writes from statistics on the stored history (hypotheses until the user confirms them),
and both are put in front of it in every conversation (memory_block).
"""
import datetime as dt
import json
import os
import re

from anthropic import AsyncAnthropic

from . import data, levels, store

price_fn = None   # set by main: returns the current price

MIN_GROUP_N = 8          # an observation may only cite a group with at least this many finished trades
MIN_TRADES_TO_ANALYSE = 20

TOOLS = [
    {
        "name": "get_setup",
        "description": (
            "Get the full stored record of one setup by id: why the engine identified it (evidence.why with the real numbers), "
            "the trigger candle and the two before it, levels crossed, plan (entry/stop/target/R:R), every target change, status "
            "and final result. Use this before explaining why a setup was identified or how it ended. Ids look like "
            "'daily-early-15m-1792409400-bull' and appear in the live list and in find_setups results."),
        "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    },
    {
        "name": "find_setups",
        "description": (
            "Search the stored setup history (newest first). Filters are optional. Use it to find a setup by kind/direction/outcome, "
            "to list what is live, or to look at examples behind a statistic."),
        "input_schema": {"type": "object", "properties": {
            "state": {"type": "string", "enum": ["all", "live", "done"]},
            "kind": {"type": "string", "enum": ["reversal", "early", "continuation", "hold", "manual"]},
            "tier": {"type": "string", "enum": ["daily", "weekly", "monthly", "manual"]},
            "direction": {"type": "string", "enum": ["bull", "bear"]},
            "outcome": {"type": "string", "enum": ["target", "stopped", "missed", "none"]},
            "limit": {"type": "integer", "description": "default 15, max 40"}}},
    },
    {
        "name": "history_stats",
        "description": (
            "Win rate, average R and a 95% confidence interval for finished trades, grouped by one dimension. "
            "Always quote n and the interval; small n means little can be concluded."),
        "input_schema": {"type": "object", "properties": {"group_by": {"type": "string", "enum": list(store.GROUPINGS)}}, "required": ["group_by"]},
    },
    {
        "name": "performance_summary",
        "description": (
            "Is it profitable? Wins, losses and missed setups with their R: R won, R lost, NET R (R won minus R lost), win rate with a 95% range, "
            "expectancy per trade, profit factor, the win rate needed to break even, max drawdown, and per-kind and per-day breakdowns. "
            "scope: engine (the engine's own setups, default), mine (the user's own setups) or all. days: 0 all time, -1 today, or the last N days. "
            "Always say how many were rebuilt from history rather than watched live, and that it is before spread and commission."),
        "input_schema": {"type": "object", "properties": {
            "scope": {"type": "string", "enum": ["engine", "mine", "all"]}, "days": {"type": "integer"},
            "watched_live_only": {"type": "boolean"},
            "feed": {"type": "string", "enum": ["active", "mt5", "twelvedata", "all"],
                     "description": "default active = the feed in use now. 'all' counts the same market move once per feed, i.e. some trades twice"}}},
    },
    {
        "name": "save_note",
        "description": (
            "Save a note to permanent memory. kind='lesson' ONLY when the user has just taught you a rule, correction or preference "
            "(it is stored as confirmed and you must follow it from now on). kind='observation' for something you noticed yourself; "
            "it is stored as a proposal until the user confirms it. Keep it one clear sentence, free of one-off numbers from today's market. "
            "Set changes_engine_rule=true when the lesson would change HOW THE ENGINE CALCULATES something (levels, validity, targets, entries, "
            "invalidation). Saving it does NOT change the engine: it is flagged so the user knows the engine still does the old thing."),
        "input_schema": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["lesson", "observation"]}, "text": {"type": "string"},
            "changes_engine_rule": {"type": "boolean"}}, "required": ["kind", "text"]},
    },
    {
        "name": "get_candles",
        "description": (
            "Read the most recent candles (open/high/low/close with readable times) of a timeframe from the live feed. Use it to check "
            "a close yourself, e.g. 'what did the Oct 14 Daily candle close at?', instead of asking the user. The last candle is "
            "still running."),
        "input_schema": {"type": "object", "properties": {
            "timeframe": {"type": "string", "enum": ["5m", "15m", "1h", "4h", "1d", "1w", "1mo"]},
            "count": {"type": "integer", "description": "default 15, max 60"},
            "up_to": {"type": "string", "description": "optional 'YYYY-MM-DD HH:MM' (feed clock): return the candles ENDING at this time, to read the past (e.g. the 1H candles of 18 Nov). If the feed does not go back that far you are told how far it does."}}, "required": ["timeframe"]},
    },
    {
        "name": "get_levels",
        "description": (
            "All currently VALID candle-close levels (Weekly, Daily, 4H) nearest to price, above and below, with distances. A level is valid "
            "if it is an older completed close, not the close just before the running candle, and no later close has broken it."),
        "input_schema": {"type": "object", "properties": {"per_timeframe": {"type": "integer", "description": "default 8, max 25"}}},
    },
    {
        "name": "get_upcoming_levels",
        "description": (
            "Think FORWARD: which levels become valid at the next rollover of each timeframe (when the running candle closes and a new one "
            "opens), and which stop being valid, assuming the running candle closes at the current price. Targets are dynamic, so a trade's "
            "target moves to such a level automatically when the new candle starts. Use it for any question about what the target will be "
            "soon, or when a candle is close to ending."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_level",
        "description": (
            "Check whether a price is a candle close on a timeframe and whether it is a valid level right now. Explains why not: "
            "excluded (the running candle or the one just before it), or broken (and by which later close). Use for questions like "
            "'is 4147.62 still a valid Daily level?'."),
        "input_schema": {"type": "object", "properties": {
            "price": {"type": "number"}, "timeframe": {"type": "string", "enum": ["4h", "1d", "1w", "1mo"]},
            "tolerance": {"type": "number", "description": "how close a close must be to count as the same price, default 0.35"}},
            "required": ["price", "timeframe"]},
    },
    {
        "name": "add_marked_level",
        "description": (
            "Mark a reference level on the user's chart, e.g. a close from another provider that the feed does not have. It is drawn on the "
            "chart in its own colour and labelled. It is only a reference: the engine does not use it for setups, and you must say so."),
        "input_schema": {"type": "object", "properties": {
            "price": {"type": "number"}, "timeframe": {"type": "string", "enum": ["4h", "1d", "1w", "1mo"]},
            "label": {"type": "string", "description": "short, e.g. 'D close Oct 14 (other provider)'"}, "note": {"type": "string"}},
            "required": ["price", "timeframe", "label"]},
    },
    {
        "name": "list_marked_levels",
        "description": "List the reference levels marked on the chart.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "remove_marked_level",
        "description": "Remove a marked reference level by id.",
        "input_schema": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
    },
    {
        "name": "add_my_setup",
        "description": (
            "Record the USER'S OWN trade setup so it appears on the setup card, the chart and the history and is tracked candle by candle like the "
            "engine's (dynamic target, stop, outcome). Use it when the user decides to take a trade the engine did not flag (for example off a level "
            "from another provider). Not an engine signal: say so. Needs direction, entry and stop; the target is optional (the engine then uses the "
            "next valid level and keeps it dynamic). trigger_time is the opening time, on the feed clock, of the candle that triggered it "
            "('YYYY-MM-DD HH:MM'); leave it out for the last closed candle."),
        "input_schema": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["buy", "sell"]},
            "entry": {"type": "number"}, "stop": {"type": "number"}, "target": {"type": "number"},
            "trigger_timeframe": {"type": "string", "enum": ["15m", "1h", "4h"], "description": "default 1h"},
            "trigger_time": {"type": "string"},
            "level_timeframe": {"type": "string", "enum": ["1d", "1w"], "description": "which level timeframe the target follows, default 1d"},
            "origin": {"type": "number", "description": "the level it is based on, e.g. a marked level"},
            "label": {"type": "string"}, "note": {"type": "string", "description": "why the user is taking it"}},
            "required": ["direction", "entry", "stop"]},
    },
    {
        "name": "ignore_feed_glitch",
        "description": (
            "CORRECT a setup the engine judged wrongly because of a bad candle in the feed (for example a spurious wick that tripped the stop, "
            "when the user's own chart shows price never got there). The engine then ignores that candle (and earlier ones) for THIS setup's "
            "stop, target and fill and carries on from the next candle, so the card and history show the trade as it really is. Use it instead "
            "of deleting and re-adding a setup, which would just be stopped again by the same candle. Only when the user says the hit was not real. "
            "By default it ignores up to the candle that decided the outcome."),
        "input_schema": {"type": "object", "properties": {
            "setup_id": {"type": "string"}, "note": {"type": "string"},
            "ignore_until": {"type": "string", "description": "optional 'YYYY-MM-DD HH:MM' feed time; default = the candle that resolved it"}},
            "required": ["setup_id"]},
    },
    {
        "name": "remove_my_setup",
        "description": "Remove one of the user's own setups by its id (the number after 'manual-').",
        "input_schema": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
    },
    {
        "name": "annotate_chart",
        "description": (
            "DRAW on the user's chart while you explain, so they can see what you are talking about. Call it at the moment you mention the thing, "
            "then keep talking. Items: hline {price,label,color,style}, zone {price_from,price_to,label,color,time_from?,time_to?}, "
            "marker {time,price,label,position(above|below),color}, box {time,tf_sec,price_from,price_to,label,color} (a rectangle around ONE candle: price_from = its low, price_to = its high), trendline {time_from,price_from,time_to,price_to,label,color}. "
            "Times are 'YYYY-MM-DD HH:MM' on the feed clock (the same clock get_candles shows). Colors: green red blue yellow pink white purple orange. "
            "Keep labels short. Use real prices from the data, never invented ones."),
        "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object"}}}, "required": ["items"]},
    },
    {
        "name": "add_custom_level",
        "description": (
            "The user says a level exists somewhere (e.g. a close on their TradingView chart). Record it so the ENGINE uses it like one of its own "
            "levels: setups can start from it (buy from a buying level, sell from a selling level), CPLOCs crossing it count, and it can be a target. "
            "direction: 'buy' = support (buyers act there), 'sell' = resistance. tf = the timeframe it acts as ('1d' for a Daily level, '4h', '1w'). "
            "since = when it started counting, 'YYYY-MM-DD HH:MM' on the feed clock (the end of the candle that made it); omit for 7 days back. "
            "Ask the user for direction and timeframe if they did not say. Afterwards, call find_setups / read the recommendation to tell them "
            "whether any setup came from it. Setups from custom levels are the user's own and are kept out of the engine's win rate."),
        "input_schema": {"type": "object", "properties": {
            "price": {"type": "number"}, "direction": {"type": "string", "enum": ["buy", "sell"]},
            "tf": {"type": "string", "enum": ["1d", "4h", "1w"]}, "since": {"type": "string"}, "label": {"type": "string"}},
            "required": ["price", "direction"]},
    },
    {
        "name": "list_custom_levels",
        "description": "The levels the user told the engine about, with whether each is still intact.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "remove_custom_level",
        "description": "Remove a custom level (and the setups that came from it). Use the id from list_custom_levels.",
        "input_schema": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
    },
    {
        "name": "focus_setup",
        "description": (
            "EXPLAIN A SETUP ON THE CHART, one step at a time. This clears every drawing that is not about this setup, hides the chart's other "
            "levels / zones / markers, switches the chart to the right timeframe, scrolls to the right time and draws the things that belong to "
            "that step: step 'level' (the level or FVG the setup formed on, and what it crossed), 'trigger' (the trigger candle and the two candles "
            "before it that set the entry), 'plan' (entry, stop and target), 'fvg' (any FVG warning on its path), 'all' (everything), 'done' "
            "(restore the normal chart). Use it instead of annotate_chart/show_chart when walking through a setup, call it at the start of each "
            "step, then talk about what is now on the chart. NEVER write JSON or tool syntax in your reply: calling this tool is how you draw."),
        "input_schema": {"type": "object", "properties": {
            "setup_id": {"type": "string"}, "step": {"type": "string", "enum": ["level", "trigger", "plan", "fvg", "all", "done"]}},
            "required": ["setup_id", "step"]},
    },
    {
        "name": "set_market_phase",
        "description": (
            "Record what the user says the current month, week or day IS in the AMDX cycle: A accumulation, M manipulation, D distribution, "
            "X reversal/continuation ('?' clears it). Call it whenever the user tells you (e.g. 'today is an M day, this is an M week'). "
            "It is stored for the current period only (it does not carry to the next day/week/month) and the engine adds a warning to "
            "reversal setups that go against an M week or M day."),
        "input_schema": {"type": "object", "properties": {
            "scope": {"type": "string", "enum": ["month", "week", "day"]}, "phase": {"type": "string", "enum": ["A", "M", "D", "X", "?"]},
            "note": {"type": "string"}}, "required": ["scope", "phase"]},
    },
    {
        "name": "show_chart",
        "description": (
            "Move the user's chart: switch it to a timeframe (1w, 1d, 4h, 1h, 15m, 5m) and scroll it to a time, so both of you look at the same candles. "
            "Use it when you say 'let's go down to the 4 hour' or 'look at the Oct 19 candle', then draw the thing you mention with annotate_chart. "
            "focus_time is 'YYYY-MM-DD HH:MM' on the feed clock; bars is how many candles to show around it (default 60)."),
        "input_schema": {"type": "object", "properties": {"timeframe": {"type": "string"}, "focus_time": {"type": "string"}, "bars": {"type": "integer"}}, "required": ["timeframe"]},
    },
    {
        "name": "clear_annotations",
        "description": "Remove everything you drew on the chart.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_chat_days",
        "description": "List the trading days that have a saved chat (newest first) with a one-line preview. Each trading day has its own chat.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_chat_day",
        "description": (
            "Read the saved chat of an earlier trading day ('YYYY-MM-DD', the feed's date), to recall what was discussed or decided. Use it when the "
            "user refers to a previous day or to something said earlier that is not in today's conversation."),
        "input_schema": {"type": "object", "properties": {"day": {"type": "string"}, "max_chars": {"type": "integer", "description": "default 6000"}},
                         "required": ["day"]},
    },
    {
        "name": "list_notes",
        "description": "List the saved lessons and observations (confirmed and proposed).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_observations",
        "description": (
            "Analyse the whole stored history and write new observations (stored as proposals for the user to confirm or reject). "
            "Use when the user asks what you have noticed or learned. Needs at least 20 finished trades."),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _t(t: int | None) -> str | None:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d %H:%M") if t else None


def _with_times(d):
    """Add readable times next to epoch times so the model does not have to convert them."""
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out[k] = _with_times(v)
            if k in ("time", "trigger_time", "resolved_time", "fill_time", "touched_at", "bos_candle_time", "bos_known_at", "parent_time") and isinstance(v, int):
                out[k + "_text"] = _t(v)
        return out
    if isinstance(d, list):
        return [_with_times(x) for x in d]
    return d


def memory_block() -> str:
    lessons = store.list_notes("confirmed", "lesson")
    obs_ok = store.list_notes("confirmed", "observation")
    obs_prop = store.list_notes("proposed")
    parts = []
    if lessons:
        parts.append("LESSONS taught by the user (follow these):\n" + "\n".join(f"- {n['text']}" for n in lessons))
    if obs_ok:
        parts.append("OBSERVATIONS the user confirmed:\n" + "\n".join(f"- {n['text']}" for n in obs_ok))
    if obs_prop:
        parts.append("UNCONFIRMED observations (your own hypotheses; say they are unconfirmed if you use them):\n"
                     + "\n".join(f"- {n['text']}" for n in obs_prop[:8]))
    marked = store.list_marked_levels()
    if marked:
        parts.append("MARKED REFERENCE LEVELS on the chart (not used by the engine): "
                     + "; ".join(f"{m['price']} {m['tf']} '{m['label']}'" for m in marked))
    c = store.count_setups()
    st = store.stats("kind", data.active_source())
    parts.append(f"STORED HISTORY: {c['total']} setups ({c['finished']} finished, across both feeds). Finished filled trades by kind on the ACTIVE feed ({data.active_source()}) only, because the same market move is stored once per feed: "
                 + json.dumps(st["groups"]) + f" overall: {json.dumps(st['overall'])}")
    return "\n\n".join(parts)


def _check_level(price: float, tf: str, tol: float) -> dict:
    """Is `price` a close on `tf`, and is it a valid level now? Says exactly why not."""
    cs = data.get_candles(tf)
    now_price = price_fn()
    n = len(cs)
    matches = []
    for k, c in enumerate(cs):
        if abs(c["close"] - price) > tol:
            continue
        lv = c["close"]
        if k == n - 1:
            status = "excluded: this is the running candle (its close is not final)"
        elif k == n - 2:
            status = "excluded: this is the close just before the running candle (it becomes a level once the next candle starts)"
        else:
            status = "valid"
            for j in range(k + 1, n):
                later = cs[j]["close"]
                # a level above price is broken by a later close above it; a level below price by a later close below it
                if (lv > now_price and later > lv) or (lv < now_price and later < lv):
                    status = f"broken: the {_t(cs[j]['time'])} candle closed at {later}, beyond it"
                    break
        nxt = cs[k + 1] if k + 1 < n else None
        matches.append({"candle": _t(c["time"]), "close": lv, "status": status, "formation": levels.formation(c, nxt),
                        "made_by": {"buy": "a BUYING level (the candle after it went up): a buy can start here; never use it as resistance",
                                    "sell": "a SELLING level (the candle after it went down): a sell can start here; never use it as support",
                                    "both": "either (the next candle was a doji or has not finished yet)"}[levels.polarity(nxt)]})
    return {"timeframe": tf, "asked_price": price, "current_price": now_price, "closes_within_tolerance": matches[-6:],
            "found": bool(matches), "note": "" if matches else "no candle on this feed closed within the tolerance of that price (a different provider's close, or a different candle boundary)"}


COLORS = {"green": "#26a69a", "red": "#ef5350", "blue": "#4aa3e0", "yellow": "#e0b84a", "pink": "#ff7ab2",
          "white": "#e6e9ef", "purple": "#c792ea", "orange": "#ff9f43"}
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400}


def parse_feed_time(v) -> int | None:
    """'YYYY-MM-DD HH:MM' (the feed clock, which the chart labels in UTC style) -> epoch seconds; ints pass through."""
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return int(dt.datetime.strptime(str(v).strip(), fmt).replace(tzinfo=dt.UTC).timestamp())
        except ValueError:
            pass
    raise ValueError(f"cannot read time {v!r}; use 'YYYY-MM-DD HH:MM'")


def create_manual_setup(a: dict) -> dict:
    """Validate and store one of the user's own setups. Raises ValueError with a plain message."""
    direction = {"buy": "bull", "sell": "bear", "bull": "bull", "bear": "bear"}.get(str(a.get("direction", "")).lower())
    if not direction:
        raise ValueError("direction must be buy or sell")
    entry, stop = float(a["entry"]), float(a["stop"])
    target = float(a["target"]) if a.get("target") not in (None, "") else None
    bull = direction == "bull"
    if (bull and stop >= entry) or (not bull and stop <= entry):
        raise ValueError(f"a {'buy' if bull else 'sell'} needs its stop {'below' if bull else 'above'} the entry")
    if target is not None and ((bull and target <= entry) or (not bull and target >= entry)):
        raise ValueError(f"a {'buy' if bull else 'sell'} needs its target {'above' if bull else 'below'} the entry")
    tf = a.get("trigger_timeframe") or "1h"
    if tf not in TF_SECONDS:
        raise ValueError("trigger_timeframe must be 15m, 1h or 4h")
    lt = a.get("level_timeframe") or "1d"
    if lt not in ("1d", "1w"):
        raise ValueError("level_timeframe must be 1d or 1w")
    origin = float(a["origin"]) if a.get("origin") not in (None, "") else None
    return store.add_manual_setup({
        "source": data.active_source(), "direction": direction, "entry": entry, "stop": stop, "target": target,
        "level_tf": lt, "trigger_tf": tf, "trigger_time": parse_feed_time(a.get("trigger_time")), "origin": origin,
        "label": (a.get("label") or "").strip(), "note": (a.get("note") or "").strip()})


def apply_glitch_correction(setup_id: str, note: str = "", ignore_until=None) -> dict:
    """Mark the candle that decided this setup's outcome as a feed glitch; the engine then re-judges the setup without it."""
    from . import reversals    # local import: reversals imports store, assistant is imported by main
    until = parse_feed_time(ignore_until)
    if until is None:
        s = next((x for x in reversals.find() if x["id"] == setup_id), None)
        if not s:
            raise ValueError(f"no live setup {setup_id!r} to correct (finished setups from before this window cannot be rebuilt)")
        until = max((s.get("plan") or {}).get("resolved_time") or 0, s.get("resolved_time") or 0)
        if not until:
            raise ValueError("this setup has not been stopped or hit anything, so there is nothing to correct")
    out = store.set_override(setup_id, until, note)
    return {**out, "ignore_until_text": _t(out["ignore_until"]),
            "result": "The engine now ignores candles up to that time for this setup and carries on from the next one."}


def normalise_annotations(items: list[dict]) -> list[dict]:
    """Check what the model asked to draw and turn it into what the chart understands (epoch times, hex colours)."""
    out = []
    for it in items:
        t = it.get("type")
        col = COLORS.get(str(it.get("color", "white")).lower(), COLORS["white"])
        base = {"type": t, "color": col, "label": str(it.get("label", ""))[:60]}
        if t == "hline":
            out.append({**base, "price": float(it["price"]), "style": it.get("style", "dashed"), "time_from": parse_feed_time(it.get("time_from"))})
        elif t == "zone":
            a, b = float(it["price_from"]), float(it["price_to"])
            out.append({**base, "price_from": min(a, b), "price_to": max(a, b),
                        "time_from": parse_feed_time(it.get("time_from")), "time_to": parse_feed_time(it.get("time_to"))})
        elif t == "box":
            out.append({**base, "time": parse_feed_time(it["time"]), "tf_sec": int(it.get("tf_sec", 3600)),
                        "price_from": float(it["price_from"]), "price_to": float(it["price_to"])})
        elif t == "marker":
            out.append({**base, "time": parse_feed_time(it["time"]), "price": float(it["price"]),
                        "position": "below" if it.get("position") == "below" else "above"})
        elif t == "trendline":
            out.append({**base, "time_from": parse_feed_time(it["time_from"]), "price_from": float(it["price_from"]),
                        "time_to": parse_feed_time(it["time_to"]), "price_to": float(it["price_to"])})
        else:
            raise ValueError(f"unknown annotation type {t!r}")
    return out


def focus_events(setup_id: str, step: str, session: str) -> dict:
    """What to show on the chart for one step of explaining a setup: events (clear, focus, view, drawings) plus a short summary for the model.
    Built from the stored setup, not from the model, so it works the same whichever model is talking."""
    if step == "done":
        store.clear_annotations(session)
        return {"events": [{"clear": True}, {"focus_off": True}], "shown": "the normal chart is back"}
    rec = store.get_setup(setup_id)
    if not rec:
        return {"error": f"no setup {setup_id!r}"}
    r = rec["record"]
    ev = r.get("evidence") or {}
    plan = r.get("plan") or {}
    bull = r["direction"] == "bull"
    blue, red, green, yellow, white, pink = "#4aa3e0", "#ef5350", "#26a69a", "#e0b84a", "#ffffff", "#f06292"
    levels: list[tuple[float, str, str, str | None]] = []       # price, label, colour, drawn-from time
    zones: list[dict] = []
    items: list[dict] = []
    prices: list[float] = []
    view_tf, focus_t, bars = r["trigger_tf"], r["time"], 60

    z = ev.get("fvg_zone")
    if r["tier"] == "fvg" and z:
        zones.append({"type": "zone", "price_from": z["bottom"], "price_to": z["top"], "time_from": z["formed"], "time_to": None, "color": yellow,
                      "label": f"{TF_NAME_[z['tf']]} {'bullish' if z['bull'] else 'bearish'} FVG" + (" (caused a BOS)" if z["flagged"] else "")})
        prices += [z["top"], z["bottom"]]
    else:
        oc = ev.get("origin_candle") or {}
        pol = {"buy": "buying", "sell": "selling"}.get(oc.get("polarity"), "")
        lvtf = TF_NAME_.get(r["level_tf"], r["level_tf"])
        hit = ev.get("hit_level", r.get("origin"))
        levels.append((hit, f"{lvtf} {pol} level".replace("  ", " "), blue, ev.get("hit_level_tf") or r["level_tf"]))
        for l in ev.get("levels_crossed") or []:
            if abs(l["price"] - hit) > 0.001:
                levels.append((l["price"], f"{TF_NAME_.get(l['tf'], l['tf'])} level crossed", yellow, l["tf"]))
    for pr, _, _, _ in levels:
        prices.append(pr)

    if step in ("level", "all"):
        if r["tier"] == "fvg":
            view_tf = z["tf"] if z else view_tf
            focus_t, bars = (z["formed"] if z else focus_t), 70
        else:
            view_tf = r["level_tf"] if r["level_tf"] in ("1w", "1d", "4h") else "1d"
            oc = ev.get("origin_candle") or {}
            focus_t = oc.get("time") or r["time"]
            bars = 40
        rank = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1mo": 2678400}
        for pr, lab, col, ltf in levels:
            if step == "level" and rank.get(ltf, 0) < rank.get(view_tf, 0):
                continue          # a lower-timeframe level is not talked about, or drawn, on a higher-timeframe chart: it comes in the step that moves to that chart
            items.append({"type": "hline", "price": pr, "label": lab, "color": col, "style": "dashed"})
        items += zones
    if step in ("trigger", "all"):
        if step == "trigger":
            view_tf, focus_t, bars = r["trigger_tf"], r["time"], 60
        trig, p1, p2 = ev.get("trigger") or {}, ev.get("prev1") or {}, ev.get("prev2") or {}
        for pr, lab, col, _ in levels:
            items.append({"type": "hline", "price": pr, "label": lab, "color": col, "style": "dashed"})
        items += zones
        if p1 and p2:
            items.append({"type": "zone", "price_from": min(p1["low"], p2["low"]), "price_to": max(p1["high"], p2["high"]), "time_from": p2["time"],
                          "time_to": trig.get("time", r["time"]), "color": white, "label": ""})
        prices += [r["low"], r["high"]]
    if step in ("plan", "all") and plan:
        if step == "plan":
            view_tf, focus_t, bars = r["trigger_tf"], r["time"], 80
        for pr, lab, col in ((plan["entry"], f"Entry {plan['entry']}", blue), (plan["stop"], f"Stop {plan['stop']}", red),
                             (plan["target"], f"Target {plan['target']}", green)):
            items.append({"type": "hline", "price": pr, "label": lab, "color": col, "style": "solid", "time_from": r["time"]})
            prices.append(pr)
        if plan.get("initial_target") and abs(plan["initial_target"] - plan["target"]) > 0.001:
            items.append({"type": "hline", "price": plan["initial_target"], "label": f"First target {plan['initial_target']}", "color": green, "style": "dotted", "time_from": r["time"]})
            prices.append(plan["initial_target"])
        if plan.get("fill_time"):
            items.append({"type": "marker", "time": plan["fill_time"], "price": plan["entry"], "position": "below" if bull else "above", "color": blue, "label": "entry filled"})
        if plan.get("resolved_time"):
            res_price = plan["stop"] if plan.get("outcome") == "stopped" else plan["target"]
            items.append({"type": "marker", "time": plan["resolved_time"], "price": res_price, "position": "above" if bull else "below",
                          "color": red if plan.get("outcome") == "stopped" else green, "label": str(plan.get("outcome") or "")})
    f = r.get("fvg")
    if step in ("fvg", "all") and f:
        if step == "fvg":
            view_tf, focus_t, bars = "1h", r["time"], 70
        for zz in f.get("zones", [])[:3]:
            items.append({"type": "zone", "price_from": zz["bottom"], "price_to": zz["top"], "time_from": zz["formed"], "time_to": None, "color": yellow,
                          "label": f"{TF_NAME_[zz['tf']]} {'bullish' if zz['bull'] else 'bearish'} FVG ahead"})
            prices += [zz["top"], zz["bottom"]]
        if f.get("cap"):
            items.append({"type": "hline", "price": f["cap"], "label": f"TP cap {f['cap']}", "color": yellow, "style": "dotted"})
            prices.append(f["cap"])
    # ---- point at the actual candles: a labelled box around each one the explanation refers to ----
    SEC_ = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1mo": 2678400}
    bycache: dict[str, dict[int, int]] = {}

    def cnd(tf: str, t: int | None = None, close: float | None = None, before: int | None = None):
        cs = data.try_candles(tf)
        if not cs:
            return None, None
        if tf not in bycache:
            bycache[tf] = {c["time"]: i for i, c in enumerate(cs)}
        if t is not None and t in bycache[tf]:
            i = bycache[tf][t]
        elif close is not None:
            idx = [k for k, c in enumerate(cs) if abs(c["close"] - close) < 0.006 and (before is None or c["time"] < before)]
            if not idx:
                return None, None
            i = idx[-1]
        else:
            return None, None
        return cs, i

    def box(tf: str, c: dict, label: str, color: str) -> dict:
        return {"type": "box", "time": c["time"], "tf_sec": SEC_[tf], "price_from": c["low"], "price_to": c["high"], "label": label, "color": color}

    boxes: list[dict] = []
    facts_candles: dict = {}
    if step in ("level", "all"):
        if r["tier"] == "fvg" and z:
            cs, k = cnd(z["tf"], t=z["formed"])
            if cs and 0 < k < len(cs) - 1:
                for j, nm in ((k - 1, "1st candle of the gap"), (k, "2nd (the big move)"), (k + 1, "3rd candle: its " + ("low stays above" if z["bull"] else "high stays below") + " the 1st candle's " + ("high" if z["bull"] else "low"))):
                    boxes.append(box(z["tf"], cs[j], nm, white))
        else:
            tf_h = ev.get("hit_level_tf") or r["level_tf"]
            oc = ev.get("origin_candle") or {}
            cs, k = cnd(tf_h, t=oc.get("time")) if oc.get("time") else (None, None)
            if cs is None:
                cs, k = cnd(tf_h, close=ev.get("hit_level", r.get("origin")), before=r["time"])
            if cs is not None:
                lvl, nxt = cs[k], (cs[k + 1] if k + 1 < len(cs) else None)
                kind_word = {"buy": "buying", "sell": "selling"}.get(oc.get("polarity"), "")
                boxes.append(box(tf_h, lvl, f"the {TF_NAME_[tf_h]} candle whose close {lvl['close']} is the level", blue))
                facts_candles["level_candle"] = {"timeframe": TF_NAME_[tf_h], "time": _t(lvl["time"]), "open": lvl["open"], "high": lvl["high"], "low": lvl["low"], "close": lvl["close"]}
                if nxt:
                    went = "up" if nxt["close"] > lvl["close"] else "down" if nxt["close"] < lvl["close"] else "flat"
                    boxes.append(box(tf_h, nxt, f"next candle went {went} → {kind_word or 'either'} level", yellow))
                    facts_candles["candle_after_it"] = {"timeframe": TF_NAME_[tf_h], "time": _t(nxt["time"]), "open": nxt["open"], "close": nxt["close"], "went": went,
                                                        "so_the_level_is_for": kind_word or "either"}
            for l in ev.get("levels_crossed") or []:
                if l.get("custom") or (step == "level" and SEC_[l["tf"]] < SEC_[tf_h]):
                    continue
                cs2, k2 = cnd(l["tf"], t=l.get("time"))
                if cs2 and abs(l["price"] - (ev.get("hit_level", r.get("origin")) or 0)) > 0.001:
                    boxes.append(box(l["tf"], cs2[k2], f"the {TF_NAME_[l['tf']]} candle that made level {l['price']}", yellow))
                    facts_candles["crossed_level_candle"] = {"timeframe": TF_NAME_[l["tf"]], "time": _t(cs2[k2]["time"]), "close": cs2[k2]["close"]}
    if step in ("trigger", "all"):
        tg = r["trigger_tf"]
        for key, label, col in (("prev2", "2nd candle before", white), ("prev1", "candle before", white), ("trigger", "TRIGGER", pink)):
            c0 = ev.get(key)
            if c0:
                boxes.append(box(tg, c0, label, col))
        if ev.get("touched_at"):
            cs3, k3 = cnd(tg, t=ev["touched_at"])
            if cs3:
                boxes.append(box(tg, cs3[k3], f"price first touched the level here ({_t(ev['touched_at'])})", yellow))
        ll = ev.get("lower_level")
        if ll and ll.get("time") and ll["tf"] != tg:
            cs4, k4 = cnd(ll["tf"], t=ll["time"])
            if cs4 and not ll.get("custom"):
                boxes.append(box(ll["tf"], cs4[k4], f"the {TF_NAME_[ll['tf']]} candle that made the level it crossed ({ll['price']})", yellow))
    if step in ("plan", "all") and plan:
        tg = r["trigger_tf"]
        for tkey, label, col in (("fill_time", "entry filled in this candle", blue), ("resolved_time", {"stopped": "this candle broke the stop", "target": "this candle hit the target"}.get(plan.get("outcome"), "ended here"), red if plan.get("outcome") == "stopped" else green)):
            if plan.get(tkey):
                cs5, k5 = cnd(tg, t=plan[tkey])
                if cs5:
                    boxes.append(box(tg, cs5[k5], label, col))
    if step in ("fvg", "all") and f:
        for zz in f.get("zones", [])[:2]:
            cs6, k6 = cnd(zz["tf"], t=zz["formed"])
            if cs6 and 0 < k6 < len(cs6) - 1:
                for j in (k6 - 1, k6, k6 + 1):
                    boxes.append(box(zz["tf"], cs6[j], f"{TF_NAME_[zz['tf']]} FVG candle", white))
    items += boxes
    items = normalise_annotations(items)
    store.clear_annotations(session)
    staged = [{**it, "cues": cues_for(it)} for it in items]          # drawn one by one, when the narration mentions each (see main.chat)
    focus = {"id": setup_id, "prices": sorted({round(p_, 2) for p_ in prices}),
             "zones": [{"top": it["price_to"], "bottom": it["price_from"]} for it in items if it["type"] == "zone"], "time": r["time"]}
    def c_(c: dict) -> dict:
        return {"time": _t(c["time"]), "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"]}
    facts: dict = {"setup": setup_id, "candles_boxed_on_the_chart": facts_candles, "direction": "BUY" if bull else "SELL", "kind_rule": (ev.get("why") or [""])[0]}
    facts["chart_timeframe"] = TF_NAME_.get(view_tf, view_tf)
    if step == "level":
        # Only what is on THIS chart: a Daily-chart step says nothing about a 4H level, a 1H candle or any lower-timeframe move. Those come in the step that moves there.
        facts.pop("kind_rule", None)
        facts["level"] = {"level_price": ev.get("hit_level", r.get("origin")), "fvg_zone": ({k: ev["fvg_zone"][k] for k in ("tf", "bull", "top", "bottom", "flagged")} if ev.get("fvg_zone") else None)}
        facts["only_talk_about"] = f"the {TF_NAME_.get(view_tf, view_tf)} chart that is on the screen; do not mention any lower timeframe, its levels or its candles yet"
    elif step == "all":
        facts["level"] = {"price": r.get("origin"), "formed_from": ev.get("origin_candle"), "levels_crossed": ev.get("levels_crossed"),
                          "fvg_zone": ev.get("fvg_zone"), "why": (ev.get("why") or [])[:5]}
    if step in ("trigger", "all"):
        facts["trigger_candle"] = c_(ev["trigger"]) if ev.get("trigger") else None
        facts["two_candles_before"] = [c_(ev["prev2"]), c_(ev["prev1"])] if ev.get("prev1") and ev.get("prev2") else None
        facts["timeframe_of_candles"] = r["trigger_tf"]
        ll = ev.get("lower_level")
        if ll:        # the lower-timeframe level this candle crossed: a LINE drawn on this chart, made by an older candle of that timeframe
            facts["level_the_trigger_crossed"] = {"timeframe": TF_NAME_.get(ll["tf"], ll["tf"]), "price": ll["price"], "made_by_the_close_of_the_candle_at": _t(ll.get("time")),
                                                  "how": "it is drawn as a line on this chart; the chart itself is the " + TF_NAME_.get(view_tf, view_tf) + " chart"}
    if step in ("plan", "all"):
        facts["plan"] = {k: (_t(v) if k in ("fill_time", "resolved_time") and v else v) for k, v in plan.items()}
        facts["stop"], facts["current_or_final_target"], facts["first_target"] = r.get("stop"), r.get("target"), r.get("initial_target")
    if step in ("fvg", "all") and f:
        facts["fvg_warning"] = f.get("warning")
    return {"events": [{"clear": True}, {"focus": focus}, {"view": {"timeframe": view_tf, "focus_time": focus_t, "bars": bars}}, {"stage": staged}],
            "facts_for_this_step__use_only_these_numbers": facts,
            "shown": f"step '{step}': chart on {view_tf}, everything unrelated hidden; drawn: " + ", ".join(sorted({it.get('label') or it['type'] for it in items}))[:400]}


def cues_for(it: dict) -> list[str]:
    """Regexes: when the narration (the words the assistant says or the voice reads) matches one of them, this drawing appears."""
    lab, t = (it.get("label") or ""), it["type"]
    num = lambda v: re.escape(f"{v:g}" if abs(v - round(v, 2)) > 1e-9 else f"{v:.2f}".rstrip("0").rstrip("."))
    if t == "hline":
        low = lab.lower()
        if low.startswith("entry"):
            return [r"entry|enter"]
        if low.startswith("stop"):
            return [r"stop"]
        if low.startswith("first target"):
            return [r"first target|initial target|original target"]
        if low.startswith("target"):
            return [r"target|take.?profit"]
        if low.startswith("tp cap"):
            return [r"cap"]
        return [num(it["price"]), r"\blevel\b"]
    if t == "zone":
        return [r"fvg|fair value|gap|zone"] if "FVG" in lab else [r"range|combined"]
    if t == "box":
        when = _t(it["time"])                                           # "MM-DD HH:MM"
        if it.get("tf_sec", 3600) >= 86400:
            mm, dd = int(when[5:7]), int(when[8:10])
            mon = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"][mm - 1]
            base = [re.escape(when[5:10]), rf"\b{mon}[a-z]*\.? 0?{dd}\b", rf"\b0?{dd} {mon}"]       # "10-15", "Oct 15", "15 Oct"
        else:
            base = [re.escape(when[11:16])]
        low = lab.lower()
        extra = (r"next .{0,15}candle|candle after|went (up|down)" if low.startswith("next candle") else
                 r"\b1st\b|first candle" if low.startswith("1st") else
                 r"\b2nd\b|second|big move|middle" if low.startswith("2nd (") else
                 r"\b3rd\b|third" if low.startswith("3rd") else
                 r"trigger" if low.startswith("trigger") else
                 r"two candles before|2nd candle|second candle|second before" if low.startswith("2nd candle before") else
                 r"candle before|previous candle|one before|prior candle" if low.startswith("candle before") else
                 r"touch|tapped|reached|came to|hit the level" if "first touched" in low else
                 r"fill|retrace|entered|entry" if "filled" in low else
                 r"stop|stopped|stopped out|broke" if "stop" in low else
                 r"target|won|worked|hit" if "target" in low else
                 r"the level candle|level candle|whose close|closed at" if "is the level" in low else
                 r"candle that made")
        m = re.search(r"(\d{3,5}(?:\.\d+)?)", lab)
        return [extra] + base + ([re.escape(m.group(1))] if m and "is the level" in low else [])
    return [r"."]


TF_NAME_ = {"15m": "15M", "1h": "1H", "4h": "4H", "1d": "Daily", "1w": "Weekly", "1mo": "Monthly"}


async def run_tool(name: str, args: dict, ctx: dict | None = None) -> str:

    try:
        if name == "get_setup":
            rec = store.get_setup(args.get("id", ""))
            if not rec:
                return json.dumps({"error": f"no setup with id {args.get('id')!r}. Use find_setups to look one up."})
            r = rec["record"]
            slim = {k: v for k, v in rec.items() if k != "record"}
            slim.update(
                why=r["evidence"]["why"], evidence={k: v for k, v in r["evidence"].items() if k != "why"},
                plan=r.get("plan"), target_history=r.get("target_history"), stop=r.get("stop"),
                invalid=r.get("invalid"), final_status=r["status"])
            return json.dumps(_with_times(slim))[:14000]
        if name == "find_setups":
            rows = store.list_setups(args.get("state", "all"), args.get("kind"), args.get("tier"), args.get("direction"),
                                     args.get("outcome"), min(int(args.get("limit", 15)), 40))
            return json.dumps(_with_times(rows))[:14000]
        if name == "performance_summary":
            days = int(args.get("days", 0))
            since = None
            if days != 0:
                now = data.get_candles("5m")[-1]["time"]
                since = now - now % 86400 if days < 0 else now - days * 86400
            feed = args.get("feed", "active")
            source = data.active_source() if feed == "active" else (None if feed == "all" else feed)
            out = store.performance(args.get("scope", "engine"), since, bool(args.get("watched_live_only", False)), source)
            out.pop("curve", None)
            out["by_day"] = out["by_day"][:14]
            return json.dumps(out)
        if name == "history_stats":
            return json.dumps(store.stats(args["group_by"], data.active_source()))
        if name == "add_my_setup":
            ms = create_manual_setup(args)
            return json.dumps({"saved": ms, "note": "It now appears on the setup card, the chart and the history as 'Your setup'. It is not an engine signal."})
        if name == "ignore_feed_glitch":
            return json.dumps(apply_glitch_correction(args["setup_id"], args.get("note", ""), args.get("ignore_until")))
        if name == "remove_my_setup":
            store.delete_manual_setup(int(args["id"]))
            return json.dumps({"removed": args["id"]})
        if name == "annotate_chart":
            items = normalise_annotations(args.get("items", []))
            session = (ctx or {}).get("session", "default")
            store.add_annotations(session, items)
            return json.dumps({"drawn": len(items), "event": {"annotate": items}})
        if name == "add_custom_level":
            row = store.add_custom_level({"price": args["price"], "direction": args["direction"], "tf": args.get("tf", "1d"),
                                          "time": parse_feed_time(args.get("since")), "label": args.get("label", "")})
            from . import reversals
            found = [x for x in reversals.find() if (x.get("custom_level") or {}).get("id") == row["id"]]
            st = levels.custom_status({"tf": row["tf"], "price": row["price"], "since": row["time"] or 0}, price_fn())
            return json.dumps({"saved": row, "level_status": st, "note": "Custom levels stay valid until the user deletes them; closes on this feed never break them (other liquidity providers differ). If level_status has closed_beyond, you may mention it, but still use the level.", "setups_from_this_level": [
                {"id": x["id"], "kind": x["kind"], "trigger_tf": x["trigger_tf"], "direction": x["direction"], "trigger": _t(x["time"]),
                 "status": x["status"], "target": x["target"]} for x in found]})
        if name == "list_custom_levels":
            price = price_fn()
            return json.dumps([{**{k: r[k] for k in ("id", "price", "direction", "tf", "label")}, "since": _t(r["time"]),
                                "status": levels.custom_status({"tf": r["tf"], "price": r["price"], "since": r["time"] or 0}, price)}
                               for r in store.list_custom_levels()])
        if name == "remove_custom_level":
            store.delete_custom_level(int(args["id"]))
            return json.dumps({"removed": args["id"]})
        if name == "focus_setup":
            return json.dumps(focus_events(args["setup_id"], args.get("step", "level"), (ctx or {}).get("session", "default")))
        if name == "set_market_phase":
            keys = store.period_keys(data.get_candles("1h")[-1]["time"])
            return json.dumps(store.set_phase(args["scope"], keys[args["scope"]], args["phase"], args.get("note", "")))
        if name == "show_chart":
            tf = args.get("timeframe", "")
            if tf not in data.TIMEFRAMES:
                return json.dumps({"error": f"timeframe must be one of {list(data.TIMEFRAMES)}"})
            view = {"timeframe": tf, "bars": max(20, min(int(args.get("bars") or 60), 300))}
            if args.get("focus_time"):
                view["focus_time"] = parse_feed_time(args["focus_time"])
            return json.dumps({"shown": view, "event": {"view": view}})
        if name == "clear_annotations":
            store.clear_annotations((ctx or {}).get("session", "default"))
            return json.dumps({"cleared": True, "event": {"clear": True}})
        if name == "get_candles":
            tf, n = args["timeframe"], max(1, min(int(args.get("count", 15)), 60))
            allc = data.get_candles(tf)
            up_to = parse_feed_time(args.get("up_to"))
            if up_to is not None:
                if up_to < allc[0]["time"]:
                    return json.dumps({"error": f"the feed's {tf} history starts at {_t(allc[0]['time'])}, which is after the requested time"})
                allc = [c for c in allc if c["time"] <= up_to]
            cs = allc[-n:]
            return json.dumps({"timeframe": tf, "note": "times are the feed's clock; the last candle is still running",
                               "candles": [{"time": _t(c["time"]), "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"]} for c in cs]})
        if name == "get_levels":
            price = price_fn()
            k = max(1, min(int(args.get("per_timeframe", 8)), 25))
            out = {"price": price}
            for tf in levels.LEVEL_TFS:
                cs = data.try_candles(tf)
                if not cs:
                    out[tf] = "not available on this feed yet"
                    continue
                lv = levels.intact_closes(cs, price)
                lv.sort(key=lambda l: abs(l["price"] - price))
                out[tf] = [{"price": l["price"], "distance": round(l["price"] - price, 2), "close_of_candle": _t(l["time"]),
                            "formation": l["formation"],
                            "made_by": {"buy": "a BUYING level (the candle after it went up): only a buy can start here, never call it resistance",
                                        "sell": "a SELLING level (the candle after it went down): only a sell can start here, never call it support",
                                        "both": "either (the next candle was a doji)"}[l["polarity"]]}
                           for l in lv[:k]]
            return json.dumps(out)
        if name == "get_upcoming_levels":
            price = price_fn()
            out = {"price": price, "assumption": "the running candle closes at the current price and the next candle opens on schedule"}
            for tf in levels.LEVEL_TFS:
                if not data.try_candles(tf):
                    out[tf] = "not available on this feed yet"
                    continue
                u = levels.upcoming(tf, price)
                out[tf] = {"rolls_at": _t(u["rolls_at"]),
                           "becomes_valid": [{"price": l["price"], "close_of_candle": _t(l["time"])} for l in u["becomes_valid"][:6]],
                           "stops_being_valid": [{"price": l["price"], "close_of_candle": _t(l["time"])} for l in u["stops_being_valid"][:6]]}
            return json.dumps(out)
        if name == "check_level":
            return json.dumps(_check_level(float(args["price"]), args["timeframe"], float(args.get("tolerance", 0.35))))
        if name == "add_marked_level":
            lv = store.add_marked_level(args["price"], args["timeframe"], args.get("label", ""), args.get("note", ""))
            return json.dumps({"marked": lv, "reminder": "reference only: drawn on the chart, NOT used by the engine"})
        if name == "list_marked_levels":
            return json.dumps(store.list_marked_levels())
        if name == "remove_marked_level":
            store.delete_marked_level(int(args["id"]))
            return json.dumps({"removed": args["id"]})
        if name == "save_note":
            kind = args["kind"]
            basis = {"engine_rule_request": True} if args.get("changes_engine_rule") else {}
            note = store.add_note(args["text"], kind, "user" if kind == "lesson" else "ai", basis=basis)
            return json.dumps({"saved": note})
        if name == "list_chat_days":
            return json.dumps(store.list_chat_days())
        if name == "get_chat_day":
            msgs = store.get_messages("day:" + args["day"].strip(), 400)
            if not msgs:
                return json.dumps({"error": f"no saved chat for {args['day']}", "days_with_chats": [d["day"] for d in store.list_chat_days()]})
            text = "\n".join(f"[{_t(int(m['created_at']))} real time] {m['role'].upper()}: {m['content']}" for m in msgs)
            cap = max(500, min(int(args.get("max_chars", 6000)), 14000))
            return json.dumps({"day": args["day"], "messages": len(msgs), "transcript": text[:cap] + (" …(cut)" if len(text) > cap else "")})
        if name == "list_notes":
            return json.dumps(store.list_notes())
        if name == "generate_observations":
            return json.dumps(await generate_observations())
        return json.dumps({"error": f"unknown tool {name}"})
    except Exception as e:  # tool errors go back to the model, which can explain them
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


async def generate_observations() -> dict:
    """Ask the model to write observations from the stored statistics; keep only well-supported ones as proposals."""
    n = store.stats("kind", data.active_source())["overall"]["n"]
    if n < MIN_TRADES_TO_ANALYSE:
        return {"created": [], "reason": f"only {n} finished trades so far; need at least {MIN_TRADES_TO_ANALYSE} before looking for patterns."}
    tables = {by: store.stats(by, data.active_source()) for by in store.GROUPINGS}   # one feed: never count a trade twice
    prompt = (
        "You are analysing the stored results of a rule-based gold (XAUUSD) trading engine. Below are statistics of finished, filled "
        "trades grouped several ways. R is the trade result in units of risk (target = +planned R:R, stop = -1).\n\n"
        + json.dumps(tables)
        + "\n\nWrite up to 5 observations a trader could test. Rules:\n"
          f"- Only cite a group with n >= {MIN_GROUP_N}. Give n and the 95% interval in the text.\n"
          "- Many groupings are being compared, so some differences will be luck. Prefer patterns that make structural sense "
          "(entry mode, timeframe, whether the target moved) over arbitrary ones (a weekday), and say when something could be noise.\n"
          "- Beware survivorship bias: never compare groups defined by what happened AFTER entry (how long a trade lasted, "
          "whether its target moved). Only groupings known at entry are valid.\n"
          "- If an interval overlaps the overall result substantially, say it is not distinguishable from average.\n"
          "- No trading advice and no certainty. Say what more data would settle it.\n"
          "- If nothing stands out, return [].\n"
          'Return ONLY a JSON array: [{"by": "<grouping name>", "group": "<group key>", "text": "<2-3 sentences>"}]')
    from . import llm
    raw = await llm.complete_text(prompt)
    m = re.search(r"\[.*\]", raw, re.S)
    items = json.loads(m.group(0)) if m else []
    existing = {(n_["basis"].get("by"), n_["basis"].get("group")) for n_ in store.list_notes() if n_["kind"] == "observation"}
    created, dropped = [], 0
    for it in items:
        by, group = it.get("by"), it.get("group")
        row = next((g for g in tables.get(by, {}).get("groups", []) if g["group"] == group), None)
        if not row or row["n"] < MIN_GROUP_N or (by, group) in existing or not it.get("text"):
            dropped += 1
            continue
        created.append(store.add_note(it["text"], "observation", "ai", basis={"by": by, "group": group, **row}))
    return {"created": created, "dropped_for_thin_evidence_or_duplicates": dropped, "finished_trades_analysed": n}
