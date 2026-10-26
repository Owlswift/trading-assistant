from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

import asyncio  # noqa: E402
import time  # noqa: E402
import datetime as dt  # noqa: E402
import json
import re  # noqa: E402
import logging  # noqa: E402

from . import assistant, data, frc_levels, levels, llm, mt5, recommendation, reversals, store  # noqa: E402
from .ticks import hub  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"

log = logging.getLogger("assistant")


def _sync_history() -> dict:
    """Store every setup the engine currently identifies; finished ones are frozen for good."""
    store.ensure_rules_version(f"{reversals.RULES_VERSION}-polarity-{levels._MODE}-fvgexit-{reversals.FVG_EXIT}")
    setups = reversals.find()
    return store.sync_setups(setups, data.active_source(), mt5.status().get("symbol") or "XAU/USD")


async def _history_loop() -> None:
    await asyncio.sleep(5)   # let the data feeds come up first
    while True:
        try:
            r = await run_in_threadpool(_sync_history)
            if r["new"]:
                log.info("history: stored %s new setup(s)", r["new"])
        except Exception as e:  # never let the loop die: data feeds come and go
            log.warning("history sync failed: %s", e)
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(_: FastAPI):
    hub.start()
    task = asyncio.create_task(_history_loop())
    yield
    task.cancel()
    await hub.stop()


app = FastAPI(title="Gold Trading Assistant", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class CustomLevelIn(BaseModel):
    direction: str                  # "buy" (support) or "sell" (resistance)
    price: float
    tf: str = "1d"                  # the timeframe it acts as: 1w / 1d / 4h
    time: int | str | None = None   # when it started counting, feed clock 'YYYY-MM-DD HH:MM' (default: 7 days back)
    label: str = ""
    note: str = ""


def _strip_markup(text: str, st: dict) -> str:
    """Some models type their own tool-call markup into the reply (like <|annotate_chart ...|>). It is never shown or spoken: drop everything from "<|" to the
    closing ">" (or the end of the line). `st` carries the state between streamed chunks."""
    text = st["hold"] + text
    st["hold"] = ""
    out, i = [], 0
    while i < len(text):
        if st["in"]:
            j = min([p for p in (text.find(">", i), text.find("\n", i)) if p != -1], default=-1)
            if j == -1:
                return "".join(out)
            st["in"] = False
            i = j + 1 if text[j] == ">" else j
            continue
        j = text.find("<|", i)
        if j == -1:
            tail = text[i:]
            if tail.endswith("<"):                                   # a "<|" may be split across two chunks
                st["hold"], tail = "<", tail[:-1]
            out.append(tail)
            break
        out.append(text[i:j])
        st["in"] = True
        i = j + 2
    return "".join(out)


_TOOL_NAMES = ("annotate_chart", "focus_setup", "show_chart", "clear_annotations", "set_market_phase", "add_marked_level", "add_custom_level", "get_candles",
               "get_levels", "get_setup", "find_setups", "check_level", "save_note")


_META = re.compile(r"^\s*(we (should|need|must|can|have|will)|we're|the user|user (says|said|wants|asks)|according to|let'?s (produce|answer|respond|craft|write)|"
                   r"i (should|need to|must|will) (respond|answer|produce|say|mention)|so (respond|answer)|they said|use tools|ok,? let'?s|the snapshot|need to (comply|respond))", re.I)


def _strip_leak(text: str, st: dict, final: bool = False) -> str:
    """Some local models write their own planning into the reply before the answer ("We should produce a response... Sure.I'll jump to..."). When a reply
    STARTS like that, hold it back until the point where the planning runs into the answer (a sentence end glued to the next capital letter, or a blank
    line after a planning paragraph) and drop everything before it. Replies that start normally pass straight through."""
    if st.get("done"):
        return text
    buf = st.get("buf", "") + text
    if len(buf) < 40 and not final:
        st["buf"] = buf
        return ""
    if not _META.match(buf):
        st["done"] = True
        return buf
    m = re.search(r"[.!?:](?=[A-Z])", buf)                     # "...accordingly.Trigger step" / "Sure.I'll jump"
    if m:
        st["done"] = True
        return buf[m.end():]
    if final or len(buf) > 6000:
        st["done"] = True
        paras = [p for p in re.split(r"\n\s*\n", buf) if p.strip() and not _META.match(p)]
        return "\n\n".join(paras)
    st["buf"] = buf
    return ""


def _strip_calls(text: str, st: dict, final: bool = False) -> str:
    """Remove tool calls the model typed into its reply as text, e.g. *annotate_chart({"type": "zone", ...})*: from the tool name through the matching closing
    bracket, and the asterisks or backticks around it. `st` carries the state between streamed chunks (the call may arrive in pieces)."""
    text = st.get("buf", "") + text
    st["buf"] = ""
    if st.get("eat"):                                               # a call closed at the end of the last chunk: its closing "*" / "`" may start this one
        stripped = text.lstrip("*`_")
        st["eat"] = not stripped
        text = stripped
    out, i, n = [], 0, len(text)
    while i < n:
        if st.get("skipline"):                                      # "*annotate_chart: 1768.97*": the rest of that line is the call
            j = text.find("\n", i)
            if j == -1:
                i = n
                break
            st["skipline"] = False
            i = j
            continue
        if st.get("depth", 0) > 0:                                  # inside a call: skip to the matching ")"
            ch = text[i]
            st["depth"] += (ch == "(") - (ch == ")")
            i += 1
            if st["depth"] == 0:
                while i < n and text[i] in "*`_":                    # asterisks / backticks that wrapped it
                    i += 1
                st["eat"] = i >= n
            continue
        hit = None
        for name in _TOOL_NAMES:
            j = text.find(name, i)
            if j != -1 and (hit is None or j < hit[0]):
                hit = (j, name)
        if hit is None:
            tail = text[i:]
            if not final:                                           # a tool name may be split across chunks: hold back a possible partial name
                k = max((m for m in range(1, min(len(tail), 18) + 1) if any(nm.startswith(tail[-m:]) for nm in _TOOL_NAMES)), default=0)
                if not k:                                           # "*", "`" or "[" that may open a call arriving in the next chunk
                    k = len(tail) - len(tail.rstrip("*`_["))
                if k:
                    while k < len(tail) and tail[-(k + 1)] in "*`_[":      # the asterisks / backticks just before it belong to the call
                        k += 1
                    st["buf"], tail = tail[-k:], tail[:-k]
            out.append(tail)
            break
        j, name = hit
        k = j + len(name)
        while k < n and text[k] == " ":
            k += 1
        linked = k < n and text[k] == "]" and text[i:j].rstrip(" *`_").endswith("[")    # written as a markdown link: [annotate_chart]([...])
        if linked:
            k += 1
            while k < n and text[k] == " ":
                k += 1
        if k >= n:
            if final:
                out.append(text[i:])
            else:
                pre = text[i:j]
                kept = pre.rstrip("*`_[")
                out.append(kept)
                st["buf"] = pre[len(kept):] + text[j:]                # keep the wrapping "*", "`" or "[" with the name: they belong to the call
            break
        if text[k] == ":" and not linked:                           # "*annotate_chart: ...*" typed as a line
            before = text[i:j]
            out.append(before.rstrip("*`_ ") if before.strip("*`_ ") == "" else before.rstrip("*`"))
            st["skipline"] = True
            i = k + 1
            continue
        if text[k] != "(":                                          # the name used as plain words: leave it
            out.append(text[i:j + len(name)])
            i = j + len(name)
            continue
        before = text[i:j].rstrip(" *`_")[:-1] if linked else text[i:j]      # drop the "[" of the link form
        out.append(before.rstrip("*`_ ") if before.strip("*`_ ") == "" else before.rstrip("*`"))
        st["depth"] = 1
        i = k + 1
    return "".join(out)


class _NoFocus(Exception):
    pass


_FOCUS: dict[str, dict] = {}      # chat session -> the setup being explained and the step the chart is on


class ChatRequest(BaseModel):
    messages: list[Message]
    timeframe: str = "1h"
    session: str = "default"
    focus_setup: str | None = None      # the user pressed "why?" on this setup: the chart is put on its first step before the model says a word
    focus_step: str | None = None       # ...or pressed a step button (level / trigger / plan / fvg): the chart is moved there by the app, not by the model


@app.get("/api/candles")
async def candles(timeframe: str = "1h"):
    try:
        return await run_in_threadpool(data.get_candles, timeframe)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Market data error: {e}")


class MT5Candles(BaseModel):
    symbol: str = ""
    frames: dict[str, list[list[float]]]   # {"1h": [[time, open, high, low, close], ...]}


class MT5Tick(BaseModel):
    t: int
    bid: float
    ask: float = 0.0


@app.post("/api/mt5/candles")
async def mt5_candles(body: MT5Candles, x_secret: str | None = Header(default=None)):
    if not mt5.secret_ok(x_secret):
        raise HTTPException(403, "bad secret")
    await run_in_threadpool(mt5.upsert, body.frames, body.symbol)
    return {"ok": True, "need_history": mt5.needs_history()}


@app.post("/api/mt5/tick")
async def mt5_tick(body: MT5Tick, x_secret: str | None = Header(default=None)):
    if not mt5.secret_ok(x_secret):
        raise HTTPException(403, "bad secret")
    hub.mt5_tick(body.t, body.bid)
    return {"ok": True}


@app.get("/api/source")
async def source() -> dict:
    return {"source": data.active_source(), "mt5": mt5.status(), "ticks": hub.status}


@app.delete("/api/mt5")
async def mt5_clear():
    """Forget everything the EA sent (the app then falls back to Twelve Data until the EA pushes again)."""
    mt5.clear()
    data._cache.clear()
    return {"ok": True}


@app.get("/api/frc")
async def frc(timeframe: str = "1h", hide_repeats: bool = False):
    """FRC/CPLOC candles that cross a valid HTF level (1H -> Daily, 4H -> Daily, 15M -> 4H only)."""
    if not frc_levels.supported(timeframe):
        return {"supported": False, "frc": []}
    try:
        found = await run_in_threadpool(frc_levels.find, timeframe, hide_repeats)
    except Exception as e:
        raise HTTPException(502, f"Market data error: {e}")
    return {"supported": True, "frc": found}


@app.get("/api/reversals")
async def get_reversals():
    """Reversal setups (reversal / early / continuation), all targeting the next valid Daily level."""
    try:
        return {"setups": await run_in_threadpool(reversals.find)}
    except Exception as e:
        raise HTTPException(502, f"Market data error: {e}")


assistant.price_fn = lambda: _current_price()


def phase_context() -> dict:
    """What the user has said the current month / week / day is (A, M, D, X), and which way the day and week are running."""
    now = data.get_candles("1h")[-1]["time"]
    keys = store.period_keys(now)
    ctx = {"keys": keys, "phase": {sc: store.get_phase(sc, keys[sc]) for sc in ("month", "week", "day")}, "trend": {}}
    for sc, tf in (("day", "1d"), ("week", "1w")):
        cs = data.try_candles(tf)
        if cs and len(cs) > 1:        # direction = the running candle's close against the previous close
            ctx["trend"][sc] = "bull" if cs[-1]["close"] > cs[-2]["close"] else "bear" if cs[-1]["close"] < cs[-2]["close"] else None
    ctx["unknown"] = [sc for sc, v in ctx["phase"].items() if v is None]
    # Weigh the three periods together, the DAY most: M is a trend period (+1); A and D lean to range / reversal (-1); X is neutral.
    weights, lean = {"day": 3, "week": 2, "month": 1}, {"M": 1, "A": -1, "D": -1, "X": 0}
    known = {sc: v for sc, v in ctx["phase"].items() if v}
    score = sum(weights[sc] * lean[v] for sc, v in known.items())
    ctx["balance"] = {"score": score, "weights": weights,
                      "verdict": None if not known else "trend" if score > 0 else "reversal-friendly" if score < 0 else "mixed",
                      "missing": ctx["unknown"]}
    return ctx


def _with_phase(rec: dict) -> dict:
    """The month, week and day are weighed together (the day most). When the balance says TREND (M outweighs), a reversal-type trade against the
    running trend gets a warning. When it leans to reversal / range (A or D outweighs), there is no warning: reversals are fair."""
    try:
        ctx = phase_context()
    except Exception:
        return rec
    rec["phase"] = ctx
    bal = ctx["balance"]
    if bal["verdict"] != "trend":
        return rec
    trend = ctx["trend"].get("day") or ctx["trend"].get("week")
    for t in rec.get("trades", []):
        if t.get("manual") or t.get("tier") == "fvg" or t.get("kind") == "continuation" or not trend:
            continue                         # your own trades and trend-continuation kinds are not reversals
        if (trend == "bull") != (t["direction"] == "BUY"):
            ph = ctx["phase"]
            t["phase_warning"] = (f"Trending period: month {ph['month'] or '?'}, week {ph['week'] or '?'}, day {ph['day'] or '?'} weigh to trend (score +{bal['score']}, the day counts most). "
                                  f"It is running {'down' if trend == 'bear' else 'up'} and this is a {t['direction']} against it. Do not anticipate a reversal.")
    return rec


def _with_expected_shifts(rec: dict, setups: list[dict], price: float) -> dict:
    """Add 'next_target' to every live trade: where its target moves at the next rollover (if it does)."""
    by_id = {s["id"]: s for s in setups}
    for t in rec.get("trades", []):
        if t["state"] in ("waiting", "open") and t["id"] in by_id:
            try:
                t["next_target"] = reversals.expected_shift(by_id[t["id"]], price)
            except Exception:
                t["next_target"] = None
    return _with_phase(rec)


@app.get("/api/decisions")
async def get_decisions(day: str | None = None, days: int = 1, decision: str = "all", tier: str = "all", direction: str = "all", limit: int = 1000):
    """Every candidate the engine looked at, TAKEN or DROPPED, with the reason (see reversals.decisions). `day` is YYYY-MM-DD on the feed clock (default: the
    latest day with a decision); `days` days from it. decision: all | taken | dropped | let go. tier: all | daily | weekly | monthly | fvg."""
    import datetime as _dt
    rows = await run_in_threadpool(reversals.decisions)
    if not rows:
        return {"day": None, "rows": [], "counts": {}}
    if day:
        try:
            start = int(_dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=_dt.UTC).timestamp())
        except ValueError:
            raise HTTPException(400, "day must be YYYY-MM-DD")
    else:
        start = rows[-1]["time"] - rows[-1]["time"] % 86400
    end = start + max(1, min(days, 31)) * 86400
    sel = [r for r in rows if start <= r["time"] < end and (tier == "all" or r["tier"] == tier) and (direction == "all" or r["direction"] == direction)]
    counts = {k: sum(1 for r in sel if r["decision"] == k) for k in ("taken", "let go", "dropped")}
    if decision != "all":
        sel = [r for r in sel if r["decision"] == decision]
    for r in sel:
        r["label"] = recommendation.SHORT_LABEL.get((r["tier"], r["kind"]), f"{r['tier']} {r['kind']}")
    return {"day": _dt.datetime.fromtimestamp(start, _dt.UTC).strftime("%Y-%m-%d"), "days": days, "counts": counts, "rows": sel[:max(1, min(limit, 5000))]}


@app.get("/api/study")
async def get_study(rule: str = "monthly"):
    """A deep-dive of one rule (see study.py): how the engine applies it, its results by year, and every trade with the engine's own explanation."""
    from . import study
    if rule not in study.STUDIES:
        raise HTTPException(400, f"rule must be one of {list(study.STUDIES)}")
    return await run_in_threadpool(study.STUDIES[rule])


@app.get("/api/recommendation")
async def get_recommendation():
    """The engine's BUY / SELL / NO TRADE call, with entry, stop, target and the reasons behind it."""
    try:
        def work():
            m5 = data.get_candles("5m")
            setups = reversals.find()
            price = _current_price()
            rec = recommendation.recommend(setups, price, m5[-1]["time"], m5)
            return _with_expected_shifts(rec, setups, price)
        return await run_in_threadpool(work)
    except Exception as e:
        raise HTTPException(502, f"Market data error: {e}")


def _current_price() -> float:
    if hub.last and hub.status == "live":
        return hub.last["p"]
    return data.get_candles("1h")[-1]["close"]


@app.get("/api/levels")
async def get_levels(atr_mult: float = 0.5, tfs: str = "1mo,1w,1d,4h"):
    wanted = tuple(t for t in tfs.split(",") if t in levels.LEVEL_TFS)

    def work():
        price = _current_price()
        return {"price": price, "levels": levels.nearby_levels(price, atr_mult, wanted)}

    try:
        return await run_in_threadpool(work)
    except Exception as e:
        raise HTTPException(502, f"Market data error: {e}")


class PhaseIn(BaseModel):
    scope: str
    phase: str
    note: str = ""


@app.get("/api/phase")
async def get_phase():
    return await run_in_threadpool(phase_context)


@app.post("/api/phase")
async def set_phase(body: PhaseIn):
    try:
        def work():
            keys = store.period_keys(data.get_candles("1h")[-1]["time"])
            return store.set_phase(body.scope, keys[body.scope], body.phase, body.note)
        return await run_in_threadpool(work)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/fvg")
async def get_fvg():
    """Unfilled fair value gaps (1H / 4H / Daily) for the chart."""
    from . import fvg
    return await run_in_threadpool(fvg.as_chart_items)


@app.post("/api/custom-levels")
async def add_custom_level(body: CustomLevelIn):
    try:
        d = body.model_dump()
        d["time"] = assistant.parse_feed_time(d["time"])
        return await run_in_threadpool(store.add_custom_level, d)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/custom-levels")
async def list_custom_levels():
    try:
        def work():
            price = _current_price()
            return [{**r, "status": levels.custom_status({"tf": r["tf"], "price": r["price"], "since": r["time"] or 0}, price)}
                    for r in store.list_custom_levels()]
        return await run_in_threadpool(work)
    except Exception as e:
        raise HTTPException(502, str(e))


@app.delete("/api/custom-levels/{level_id}")
async def delete_custom_level(level_id: int):
    try:
        store.delete_custom_level(level_id)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, str(e))


def _fmt_t(t: int) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%m-%d %H:%M")


@app.get("/api/feed-check")
async def feed_check(n: int = 60):
    """Where this feed's candle colour (open->close) disagrees with the engine's reading (close vs previous close)."""
    def work():
        out = {}
        for tf in ("1d", "4h", "1h", "15m"):
            try:
                out[tf] = data.direction_conflicts(tf, n)
            except Exception as e:
                out[tf] = {"error": str(e)}
        return {"source": data.active_source(), "conflicts": out}
    return await run_in_threadpool(work)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(400, "Last message must be from the user")
    try:
        cs = await run_in_threadpool(data.get_candles, req.timeframe)
    except Exception as e:
        raise HTTPException(502, f"Market data error: {e}")
    snapshot = data.summarize(cs)
    snapshot["data_source"] = data.active_source()
    try:
        snapshot["market_phase"] = phase_context()
    except Exception:
        pass
    try:
        snapshot["feed_direction_conflicts_daily_4h"] = {
            tf: [{"time": _fmt_t(x["time"]), "gap": x["gap"], "open_close": x["by_open_close"], "vs_prev_close": x["by_prev_close"]}
                 for x in data.direction_conflicts(tf, 15)] for tf in ("1d", "4h")}
    except Exception:
        pass
    if hub.last and hub.status == "live":
        snapshot["live_price"] = hub.last["p"]
    try:
        near = await run_in_threadpool(lambda: levels.nearby_levels(_current_price(), 0.5))
        snapshot["nearby_close_levels"] = [
            {"tf": l["tf"], "price": l["price"], "distance": l["dist"]} for l in near[:12]
        ]
    except Exception:
        pass
    try:
        setups = await run_in_threadpool(reversals.find)
        snapshot["reversal_setups_latest"] = [{k: v for k, v in x.items() if k != "evidence"} for x in setups[-3:]]
        m5 = await run_in_threadpool(data.get_candles, "5m")
        snapshot["recommendation"] = _with_expected_shifts(
            recommendation.recommend(setups, _current_price(), m5[-1]["time"], m5), setups, _current_price())
    except Exception:
        pass
    history = [m.model_dump() for m in req.messages if m.role in ("user", "assistant")]
    if not req.session.startswith(("day:", "chat:", "test-")):
        req.session = await run_in_threadpool(_trading_day_session)     # old clients: file it under today
    await run_in_threadpool(store.add_message, req.session, "user", req.messages[-1].content)

    pre_events: list[dict] = []
    direct: str | None = None            # a step of a Deep-dive setup: the app says the engine's own explanation, word for word (no model in between)
    focus = _FOCUS.get(req.session)
    try:
        import re as _re0
        text = req.messages[-1].content.strip().lower()
        if req.focus_setup:                                       # "why?" or a step button: the app picks the step
            step = req.focus_step if req.focus_step in ("level", "trigger", "plan", "fvg", "all") else "level"
            focus = _FOCUS[req.session] = {"setup": req.focus_setup, "step": step}
            move = True
        elif focus and _re0.search(r"\b(done|exit|stop explaining|new setup|another setup)\b", text):
            _FOCUS.pop(req.session, None)                          # the user is finished with this setup
            focus, move = None, False
            pre_events = (await run_in_threadpool(assistant.focus_events, "", "done", req.session)).get("events", [])
        elif focus:                                                # a plain "yes / continue / next": the app moves to the next step itself
            order = ["level", "trigger", "plan"]
            nxt = order[order.index(focus["step"]) + 1] if focus["step"] in order[:-1] and "?" not in text and _re0.search(
                r"\b(continue|next|go on|proceed|lets? go|switch|yes|yeah|yep|ok|okay|sure|go ahead|show)\b", text) else focus["step"]
            move = nxt != focus["step"]
            focus["step"] = nxt
        # "go to the monthly chart": the APP moves the chart (the model has no tools while a setup is being explained)
        tf_ask = None
        if focus and not req.focus_setup:                         # a step button / "why?" press is not a timeframe request
            said = _re0.sub(r"\b(daily|weekly|monthly|fvg|manual)-[\w-]+", " ", text)      # setup ids ("monthly-level-hold-1d-...") are not the user asking for a chart
            m_tf = _re0.search(r"\b(monthly|month|weekly|week|daily|day|4\s?h(?:our)?|1\s?h(?:our)?|15\s?m(?:in)?)\b", said)
            if m_tf and _re0.search(r"\b(go|switch|show|move|open|take|put|chart|tf|timeframe|view)\b", said):
                w = m_tf.group(1).replace(" ", "")
                tf_ask = {"monthly": "1mo", "month": "1mo", "weekly": "1w", "week": "1w", "daily": "1d", "day": "1d"}.get(w) or (
                    "4h" if w.startswith("4h") else "1h" if w.startswith("1h") else "15m")
                move = False
        if focus:
            fe = await run_in_threadpool(assistant.focus_events, focus["setup"], focus["step"], req.session)
            if fe.get("error"):                  # the setup is not in the history any more (the rules changed and the history was rebuilt)
                for k in ("reversal_setups_latest", "recommendation", "nearby_close_levels", "recent_candles_ohlc"):
                    snapshot.pop(k, None)
                snapshot["chart_focus"] = {"setup": focus["setup"], "error": "THIS SETUP DOES NOT EXIST under the current rules (the history is rebuilt whenever a rule changes, "
                                           "so older setups can disappear).", "rule": "Say exactly that in one or two sentences and suggest picking a setup that is in the History tab now. Say nothing else about it and do not describe any trade."}
                _FOCUS.pop(req.session, None)
                raise _NoFocus()
            if move:
                pre_events = fe.get("events", [])
            # While one setup is being explained it is the ONLY setup in play: nothing else about the market's current trades is handed to the model, so
            # it cannot mix another trade's numbers into the explanation.
            for k in ("reversal_setups_latest", "recommendation", "nearby_close_levels", "recent_candles_ohlc"):
                snapshot.pop(k, None)
            moved = None
            if tf_ask:
                v = await run_in_threadpool(assistant.focus_view, focus["setup"], tf_ask)
                if v:
                    pre_events = [{"view": v["view"]}] + ([{"annotate": v["draw"]}] if v.get("draw") else [])
                    if v.get("draw"):
                        await run_in_threadpool(store.add_annotations, req.session, v["draw"])
                    moved = {"done": f"The app has ALREADY moved the chart to the {assistant.TF_NAME_.get(tf_ask, tf_ask)} chart, as the user asked.",
                             "facts_about_it": v["facts"],
                             "say": "Say facts_about_it.in_plain_words (you may shorten it, never change a number, a timeframe or what crossed what), then ask what they see. Do not tell the user to switch "
                                    "it themselves, do not say you will move it, never describe a tool, and use only these numbers."}
            snapshot["chart_focus"] = {"setup": focus["setup"], "step_on_the_chart": focus["step"], "shown": fe.get("shown"),
                                       "facts_use_only_these_numbers": fe.get("facts_for_this_step__use_only_these_numbers"),
                                       "rule": "This is the ONLY setup under discussion. Every number you say must come from these facts. Any other trade is not part of this conversation."}
            say = (fe.get("facts_for_this_step__use_only_these_numbers") or {}).get("engine_explanation__say_this")
            if move and say and not tf_ask:
                direct = "\n\n".join(say) + "\n\nDo you see the same on your chart?"
            if moved:
                snapshot["chart_focus"] = {"setup": focus["setup"], "the_app_moved_the_chart": moved,
                                           "engine_decision": (fe.get("facts_for_this_step__use_only_these_numbers") or {}).get("engine_decision"),
                                           "rule": "This is the ONLY setup under discussion. Every number you say must come from these facts."}
    except _NoFocus:
        pre_events = []
    except Exception:
        pre_events = []

    async def gen():
        import re as _re
        parts: list[str] = []
        pending: list[dict] = []        # drawings waiting for the narration to reach them
        markup = {"hold": "", "in": False}
        calls = {"buf": "", "depth": 0}
        leak: dict = {}
        scan = ""                       # what has been said since they were staged

        def stage(items: list[dict]) -> None:
            nonlocal pending, scan
            pending, scan = list(items), ""

        def reveal(final: bool = False) -> list[str]:
            """Drawings whose cue has now been said (or, at the end of the message, whatever is left), in order."""
            nonlocal pending
            out, keep = [], []
            norm = _re.sub("[\u2010-\u2015\u2212]", "-", scan).lower()
            for it in pending:
                hit = final or any(_re.search(c, norm, _re.I) for c in it.get("cues", []))
                if hit:
                    clean = {k: v for k, v in it.items() if k != "cues"}
                    store.add_annotations(req.session, [clean])
                    out.append("\x1e" + json.dumps({"annotate": [clean]}) + "\x1f")
                else:
                    keep.append(it)
            pending = keep
            return out

        def handle(e_: dict) -> list[str]:
            if "stage" in e_:
                stage(e_["stage"])
                return []
            return ["\x1e" + json.dumps(e_) + "\x1f"]

        for e_ in pre_events:
            for piece in handle(e_):
                yield piece
        async def spoken():
            for part in _re.split(r"(?<=\n\n)", direct):
                yield part
        try:
            async for chunk in (spoken() if direct else llm.stream_reply(history, req.timeframe, snapshot, req.session)):
                if isinstance(chunk, dict):          # a chart move / drawing: sent between control characters, never saved as chat text
                    for piece in handle(chunk):
                        yield piece
                    continue
                chunk = _strip_leak(_strip_calls(_strip_markup(chunk, markup), calls), leak)
                if not chunk:
                    continue
                parts.append(chunk)
                yield chunk
                if pending:
                    scan += chunk
                    for piece in reveal():
                        yield piece
            tail = _strip_leak(_strip_calls("", calls, final=True), leak, final=True)       # characters held back at the very end
            if tail:
                parts.append(tail)
                yield tail
        except Exception as e:  # surface API errors (missing key etc.) in the chat
            msg = f"\n\n[Error from LLM: {e}]"
            parts.append(msg)
            yield msg
        finally:
            reply = "".join(parts)
            if reply.strip():
                await run_in_threadpool(store.add_message, req.session, "assistant", reply)
        if pending:
            for piece in reveal(final=True):         # the picture is complete when the message ends
                yield piece

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


class MarkedLevelIn(BaseModel):
    price: float
    tf: str = "1d"
    label: str = ""
    note: str = ""


@app.get("/api/marked-levels")
async def marked_levels():
    return await run_in_threadpool(store.list_marked_levels)


@app.post("/api/marked-levels")
async def add_marked_level(body: MarkedLevelIn):
    if body.tf not in ("4h", "1d", "1w", "1mo"):
        raise HTTPException(400, "tf must be 4h, 1d, 1w or 1mo")
    return await run_in_threadpool(store.add_marked_level, body.price, body.tf, body.label, body.note)


@app.delete("/api/marked-levels/{level_id}")
async def remove_marked_level(level_id: int):
    await run_in_threadpool(store.delete_marked_level, level_id)
    return {"ok": True}


class ManualSetupIn(BaseModel):
    direction: str
    entry: float
    stop: float
    target: float | None = None
    trigger_timeframe: str = "1h"
    trigger_time: str | None = None
    level_timeframe: str = "1d"
    origin: float | None = None
    label: str = ""
    note: str = ""


@app.post("/api/my-setups")
async def add_my_setup(body: ManualSetupIn):
    try:
        return await run_in_threadpool(assistant.create_manual_setup, body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/my-setups/{setup_id}")
async def remove_my_setup(setup_id: int):
    await run_in_threadpool(store.delete_manual_setup, setup_id)
    return {"ok": True}


class OverrideIn(BaseModel):
    setup_id: str
    note: str = ""
    ignore_until: str | None = None


@app.post("/api/overrides")
async def add_override(body: OverrideIn):
    """'That stop / target was not really hit: the candle is a feed glitch.' Re-judges the setup without it."""
    try:
        out = await run_in_threadpool(assistant.apply_glitch_correction, body.setup_id, body.note, body.ignore_until)
        await run_in_threadpool(_sync_history)
        return out
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/overrides/{setup_id}")
async def remove_override(setup_id: str):
    await run_in_threadpool(store.clear_override, setup_id)
    return {"ok": True}


@app.get("/api/annotations")
async def get_annotations(session: str = "default"):
    return await run_in_threadpool(store.list_annotations, session)


@app.delete("/api/annotations")
async def clear_annotations(session: str = "default"):
    await run_in_threadpool(store.clear_annotations, session)
    return {"ok": True}


def _feed_now() -> int:
    """The feed's clock right now (MT5 broker time, or UTC for Twelve Data): its latest 5m candle."""
    return data.get_candles("5m")[-1]["time"]


def _trading_day_session() -> str:
    """Each trading day has its own chat. The day is the date on the feed's clock, so it rolls over when the chart's day does."""
    return "day:" + dt.datetime.fromtimestamp(_feed_now(), dt.UTC).strftime("%Y-%m-%d")


def _migrate_old_chats_once() -> None:
    if store.meta_get("chat_days_migrated") == "1":
        return
    offset = round((_feed_now() - time.time()) / 1800) * 1800     # the feed's offset from real time, to the half hour
    store.migrate_sessions_to_days(offset)
    store.meta_set("chat_days_migrated", "1")


@app.get("/api/chat/today")
async def chat_today():
    def work():
        _migrate_old_chats_once()
        key = _trading_day_session()
        return {"session": key, "day": key[4:]}
    try:
        return await run_in_threadpool(work)
    except Exception as e:
        raise HTTPException(502, f"Market data error: {e}")


class RefileIn(BaseModel):
    offset_minutes: int      # the browser's offset from UTC, e.g. +60 for Lagos


@app.post("/api/chat/refile")
async def chat_refile(body: RefileIn):
    """Each chat day follows the USER's calendar (their browser's clock), not the data feed's. Done once, for chats saved before that:
    they are re-filed by the user's own date. After that, new messages are simply saved under the day the browser says it is."""
    def work():
        if store.meta_get("chat_days_user_tz") is not None:
            return {"refiled": 0, "already": True}
        out = store.refile_days(body.offset_minutes * 60)
        store.meta_set("chat_days_user_tz", str(body.offset_minutes))
        return {"refiled": out["moved"], "already": False}
    return await run_in_threadpool(work)


@app.get("/api/chat/list")
async def chat_list():
    return await run_in_threadpool(store.list_chats)


@app.get("/api/chat/days")
async def chat_days():
    return await run_in_threadpool(store.list_chat_days)


@app.get("/api/chat/history")
async def chat_history(session: str = "default"):
    return await run_in_threadpool(store.get_messages, session)


# ---- permanent history of setups ----
@app.get("/api/history")
async def history(state: str = "all", kind: str | None = None, tier: str | None = None, direction: str | None = None,
                  outcome: str | None = None, limit: int = 100, offset: int = 0):
    rows = await run_in_threadpool(store.list_setups, state, kind, tier, direction, outcome, min(limit, 500), offset)
    try:            # the engine's own reason (decision log), shown on every row
        why = {d["id"]: d for d in await run_in_threadpool(reversals.decisions) if d.get("id")}
        for r in rows:
            d = why.get(r["id"])
            if d:
                r["decision"], r["decision_reason"] = d["decision"], (f"{d['what'].rstrip('.')}. " if d.get("what") else "") + d["reason"]
    except Exception:
        pass
    return {"counts": store.count_setups(), "setups": rows}


@app.get("/api/history/stats")
async def history_stats(by: str = "kind", feed: str = "active"):
    if by not in store.GROUPINGS:
        raise HTTPException(400, f"by must be one of {list(store.GROUPINGS)}")
    source = data.active_source() if feed == "active" else (None if feed == "all" else feed)
    return await run_in_threadpool(store.stats, by, source)


@app.get("/api/history/performance")
async def history_performance(scope: str = "engine", days: int = 0, live_only: bool = False, feed: str = "active"):
    """Won / lost / missed, R won, R lost, net R, expectancy, drawdown and an equity curve.
    days: 0 = all time, -1 = since the feed's day began, N = the last N days (on the feed's clock)."""
    if scope not in ("engine", "mine", "all"):
        raise HTTPException(400, "scope must be engine, mine or all")

    def work():
        since = None
        if days != 0:
            now = _feed_now()
            since = now - now % 86400 if days < 0 else now - days * 86400
        source = data.active_source() if feed == "active" else (None if feed == "all" else feed)
        return store.performance(scope, since, live_only, source)
    return await run_in_threadpool(work)


@app.get("/api/history/{setup_id}")
async def history_one(setup_id: str):
    rec = await run_in_threadpool(assistant.get_setup_any, setup_id)
    if not rec:
        raise HTTPException(404, "unknown setup")
    return rec


# ---- the assistant's notes: lessons the user teaches, observations it proposes ----
class NoteIn(BaseModel):
    text: str
    kind: str = "lesson"


class NoteStatus(BaseModel):
    status: str


@app.get("/api/notes")
async def notes(status: str | None = None, kind: str | None = None):
    return await run_in_threadpool(store.list_notes, status, kind)


@app.post("/api/notes")
async def add_note(body: NoteIn):
    if not body.text.strip():
        raise HTTPException(400, "empty note")
    return await run_in_threadpool(store.add_note, body.text, body.kind, "user")


@app.post("/api/notes/generate")
async def generate_notes():
    try:
        return await assistant.generate_observations()
    except Exception as e:
        raise HTTPException(502, f"Could not generate observations: {e}")


@app.patch("/api/notes/{note_id}")
async def note_status(note_id: int, body: NoteStatus):
    if body.status not in ("proposed", "confirmed", "rejected"):
        raise HTTPException(400, "bad status")
    note = await run_in_threadpool(store.set_note_status, note_id, body.status)
    if not note:
        raise HTTPException(404, "unknown note")
    return note


@app.delete("/api/notes/{note_id}")
async def remove_note(note_id: int):
    await run_in_threadpool(store.delete_note, note_id)
    return {"ok": True}


@app.websocket("/ws/ticks")
async def ws_ticks(ws: WebSocket):
    await ws.accept()
    q = hub.subscribe()
    try:
        await ws.send_json({"status": hub.status})
        if hub.last:
            await ws.send_json(hub.last)
        while True:
            await ws.send_json(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unsubscribe(q)


@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
