"""Permanent memory: every setup the engine ever identified, the notes the assistant keeps, and the chat.

SQLite file at data/assistant.db (stdlib only). Three tables:

setups        one row per setup id, holding the whole record (evidence, plan, target history, status, result).
              A row is FROZEN once its outcome is final: later data refreshes can no longer rewrite history,
              and it survives the moment the candle window moves on and the engine can no longer rebuild it.
observations  the assistant's notes. kind = "lesson" (taught by the user) or "observation" (written by the
              assistant from statistics); status = proposed / confirmed / rejected.
messages      the conversation, per session, so a page refresh does not lose it.
"""
import json
import math
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path

from . import data  # noqa: E402

DB = Path(__file__).resolve().parents[2] / "data" / "assistant.db"
_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS setups (
    id TEXT PRIMARY KEY,
    tier TEXT, kind TEXT, trigger_tf TEXT, direction TEXT, trigger_time INTEGER,
    status TEXT,                -- setup: pending / completed / failed
    plan_status TEXT,           -- waiting / open / target / stopped / missed / none
    outcome TEXT,               -- target / stopped / missed / none  (the trade plan's result)
    result_r REAL,              -- realised R: +rr on target, -1 on stopped, 0 on missed; NULL while unresolved
    invalid_reason TEXT,
    source TEXT, symbol TEXT,
    first_seen REAL, last_seen REAL,
    resolved_at_first_seen INTEGER,   -- 1 = already finished when first stored (rebuilt from history, not watched live)
    frozen INTEGER DEFAULT 0,
    data TEXT                   -- the full setup as JSON
);
CREATE INDEX IF NOT EXISTS idx_setups_time ON setups(trigger_time);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL, updated_at REAL,
    kind TEXT,                  -- lesson / observation
    status TEXT,                -- proposed / confirmed / rejected
    source TEXT,                -- user / ai
    text TEXT,
    basis TEXT                  -- JSON: the numbers behind it (group, n, win rate...)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT, role TEXT, content TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session, id);
CREATE TABLE IF NOT EXISTS marked_levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    price REAL, tf TEXT,        -- tf: 1w / 1d / 4h
    label TEXT, note TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS manual_setups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL, source TEXT,              -- source: the data feed whose clock trigger_time is on (mt5 / twelvedata)
    direction TEXT, entry REAL, stop REAL, target REAL,   -- target may be NULL: the engine then picks the next valid level
    level_tf TEXT, trigger_tf TEXT, trigger_time INTEGER, origin REAL,
    label TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS custom_levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL,
    source TEXT,                 -- the feed whose clock `time` is on (mt5 / twelvedata); the level itself is custom
    direction TEXT,               -- "buy" (support) or "sell" (resistance)
    price REAL,
    time INTEGER,                 -- the candle close that produced the level, on the custom level's feed clock
    tf TEXT,                      -- time frame the level came from, e.g. "5m"
    label TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, created_at REAL, item TEXT
);
CREATE TABLE IF NOT EXISTS overrides (
    setup_id TEXT PRIMARY KEY, ignore_until INTEGER, note TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS market_phases (
    scope TEXT, key TEXT, phase TEXT, note TEXT, updated_at REAL, PRIMARY KEY (scope, key)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def db() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            DB.parent.mkdir(exist_ok=True)
            _conn = sqlite3.connect(DB, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.executescript(SCHEMA)
            _conn.commit()
        return _conn


def reset_for_tests(path: Path) -> None:
    """Point the store at another file (tests only)."""
    global _conn, DB
    with _lock:
        if _conn:
            _conn.close()
        _conn, DB = None, path


# ---------------------------------------------------------------- setups

def _final(s: dict) -> bool:
    p = s.get("plan")
    plan_done = p is None or p.get("outcome") is not None
    return s["status"] != "pending" and plan_done


def _result_r(s: dict) -> float | None:
    p = s.get("plan")
    if not p or p.get("outcome") is None:
        return None
    if p.get("result_r") is not None:        # a trade with a partial exit knows its own result
        return float(p["result_r"])
    return {"target": float(p["rr"]), "stopped": -1.0, "missed": 0.0}.get(p["outcome"])


def sync_setups(setups: list[dict], source: str, symbol: str) -> dict:
    """Store new setups and update the ones that are still unresolved. Frozen rows are never touched."""
    now = time.time()
    new = updated = 0
    with _lock:
        c = db()
        for s in setups:
            row = c.execute("SELECT frozen FROM setups WHERE id=?", (s["id"],)).fetchone()
            if row and row["frozen"]:
                continue
            p = s.get("plan")
            final = _final(s)
            vals = (
                s["tier"], s["kind"], s["trigger_tf"], s["direction"], s["time"],
                s["status"], (p or {}).get("status", "none"), (p or {}).get("outcome") or "none",
                _result_r(s), (s.get("invalid") or {}).get("reason"),
                source, symbol, now, 1 if final else 0, json.dumps(s),
            )
            if row is None:
                c.execute(
                    "INSERT INTO setups (id, tier, kind, trigger_tf, direction, trigger_time, status, plan_status, outcome,"
                    " result_r, invalid_reason, source, symbol, first_seen, last_seen, resolved_at_first_seen, frozen, data)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (s["id"], *vals[:12], now, now, 1 if final else 0, 1 if final else 0, vals[14]),
                )
                new += 1
            else:
                c.execute(
                    "UPDATE setups SET status=?, plan_status=?, outcome=?, result_r=?, invalid_reason=?, source=?, symbol=?,"
                    " last_seen=?, frozen=?, data=? WHERE id=?",
                    (vals[5], vals[6], vals[7], vals[8], vals[9], vals[10], vals[11], now, 1 if final else 0, vals[14], s["id"]),
                )
                updated += 1
        c.commit()
    return {"new": new, "updated": updated}


def _summary(r: sqlite3.Row) -> dict:
    d = json.loads(r["data"])
    p = d.get("plan") or {}
    return {
        "id": r["id"], "tier": r["tier"], "kind": r["kind"], "trigger_tf": r["trigger_tf"], "direction": r["direction"],
        "trigger_time": r["trigger_time"], "status": r["status"], "plan_status": r["plan_status"], "outcome": r["outcome"],
        "result_r": r["result_r"], "invalid_reason": r["invalid_reason"], "source": r["source"], "symbol": r["symbol"],
        "finished": bool(r["frozen"]), "backfilled": bool(r["resolved_at_first_seen"]), "custom_level": d.get("custom_level"),
        "origin": d.get("origin"), "initial_target": d.get("initial_target"), "target": d.get("target"),
        "entry": p.get("entry"), "stop": p.get("stop"), "rr": p.get("rr"), "entry_mode": p.get("mode"),
        "target_changes": max(0, len(d.get("target_history", [])) - 1), "resolved_time": d.get("resolved_time"),
        "plan_resolved_time": p.get("resolved_time"),
    }


def list_setups(state: str = "all", kind: str | None = None, tier: str | None = None, direction: str | None = None,
                outcome: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    q, args = "SELECT * FROM setups WHERE 1=1", []
    if state == "live":
        q += " AND frozen=0"
    elif state == "done":
        q += " AND frozen=1"
    for col, val in (("kind", kind), ("tier", tier), ("direction", direction), ("outcome", outcome)):
        if val:
            q += f" AND {col}=?"
            args.append(val)
    q += " ORDER BY trigger_time DESC LIMIT ? OFFSET ?"
    with _lock:
        return [_summary(r) for r in db().execute(q, (*args, limit, offset)).fetchall()]


def get_setup(setup_id: str) -> dict | None:
    with _lock:
        r = db().execute("SELECT * FROM setups WHERE id=?", (setup_id,)).fetchone()
    if not r:
        return None
    return {**_summary(r), "record": json.loads(r["data"])}


def count_setups() -> dict:
    with _lock:
        r = db().execute("SELECT COUNT(*) n, SUM(frozen) done FROM setups").fetchone()
    return {"total": r["n"], "finished": r["done"] or 0}


# ---------------------------------------------------------------- statistics

def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def _group_key(row: dict, by: str) -> str:
    import datetime as dt
    if by == "kind":
        return f"{row['tier']}/{row['kind']}"
    if by == "direction":
        return row["direction"]
    if by == "entry_mode":
        return row["entry_mode"] or "?"
    if by == "trigger_tf":
        return row["trigger_tf"]
    if by == "weekday":
        return dt.datetime.fromtimestamp(row["trigger_time"], dt.UTC).strftime("%a")
    if by == "session":   # hour of the feed's clock (broker time for MT5, UTC otherwise)
        h = dt.datetime.fromtimestamp(row["trigger_time"], dt.UTC).hour
        return "00-07" if h < 8 else "08-15" if h < 16 else "16-23"
    if by == "rr":
        r = row["rr"] or 0
        return "<2R" if r < 2 else "2-4R" if r < 4 else ">4R"
    if by == "source":
        return row["source"] or "?"
    return "all"


# Only groupings knowable AT ENTRY. Anything defined by what happened afterwards (how long the trade lasted, whether
# the target moved) is survivorship bias: trades that stopped out early can never be in those groups.
GROUPINGS = ("kind", "direction", "entry_mode", "trigger_tf", "weekday", "session", "rr", "source")


def stats(by: str = "kind", source: str | None = None) -> dict:
    """Win rate and average R for trades that FILLED and finished (target or stopped). Invalid setups are excluded."""
    with _lock:
        rows = [_summary(r) for r in db().execute(
            "SELECT * FROM setups WHERE frozen=1 AND outcome IN ('target','stopped') AND invalid_reason IS NULL").fetchall()]
    if source:
        rows = [r for r in rows if r["source"] == source]
    groups = defaultdict(list)
    for r in rows:
        groups[_group_key(r, by)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        wins = sum(1 for r in rs if r["outcome"] == "target")
        lo, hi = _wilson(wins, len(rs))
        rr = [r["result_r"] for r in rs]
        out.append({"group": key, "n": len(rs), "wins": wins, "win_rate": round(wins / len(rs), 3),
                    "win_rate_95ci": [lo, hi], "avg_r": round(sum(rr) / len(rr), 2), "total_r": round(sum(rr), 1)})
    allr = [r["result_r"] for r in rows]
    overall = {"n": len(rows), "wins": sum(1 for r in rows if r["outcome"] == "target"),
               "avg_r": round(sum(allr) / len(allr), 2) if allr else None, "total_r": round(sum(allr), 1) if allr else 0.0}
    return {"by": by, "feed": source or "all feeds", "overall": overall, "groups": out,
            "note": "Only filled trades that reached target or stop, valid setups only. Small n means wide uncertainty (see win_rate_95ci)."}


# ---------------------------------------------------------------- observations / lessons

def add_note(text: str, kind: str = "observation", source: str = "ai", status: str | None = None,
             basis: dict | None = None) -> dict:
    status = status or ("confirmed" if source == "user" else "proposed")
    now = time.time()
    with _lock:
        cur = db().execute(
            "INSERT INTO observations (created_at, updated_at, kind, status, source, text, basis) VALUES (?,?,?,?,?,?,?)",
            (now, now, kind, status, source, text.strip(), json.dumps(basis or {})))
        db().commit()
        return get_note(cur.lastrowid)


def get_note(note_id: int) -> dict | None:
    with _lock:
        r = db().execute("SELECT * FROM observations WHERE id=?", (note_id,)).fetchone()
    return _note(r) if r else None


def _note(r: sqlite3.Row) -> dict:
    return {"id": r["id"], "created_at": r["created_at"], "kind": r["kind"], "status": r["status"],
            "source": r["source"], "text": r["text"], "basis": json.loads(r["basis"] or "{}")}


def list_notes(status: str | None = None, kind: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM observations WHERE status != 'rejected'" if not status else "SELECT * FROM observations WHERE status=?", []
    if status:
        args.append(status)
    if kind:
        q += " AND kind=?"
        args.append(kind)
    q += " ORDER BY id DESC"
    with _lock:
        return [_note(r) for r in db().execute(q, args).fetchall()]


def set_note_status(note_id: int, status: str) -> dict | None:
    with _lock:
        db().execute("UPDATE observations SET status=?, updated_at=? WHERE id=?", (status, time.time(), note_id))
        db().commit()
    return get_note(note_id)


def delete_note(note_id: int) -> None:
    with _lock:
        db().execute("DELETE FROM observations WHERE id=?", (note_id,))
        db().commit()


# ---------------------------------------------------------------- chat messages

def add_message(session: str, role: str, content: str) -> None:
    with _lock:
        db().execute("INSERT INTO messages (session, role, content, created_at) VALUES (?,?,?,?)",
                     (session, role, content, time.time()))
        db().commit()


def get_messages(session: str, limit: int = 200) -> list[dict]:
    with _lock:
        rows = db().execute("SELECT role, content, created_at FROM messages WHERE session=? ORDER BY id DESC LIMIT ?",
                            (session, limit)).fetchall()
    return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in reversed(rows)]


# ---------------------------------------------------------------- meta

def meta_get(key: str) -> str | None:
    with _lock:
        r = db().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return r["value"] if r else None


def meta_set(key: str, value: str) -> None:
    with _lock:
        db().execute("INSERT INTO meta (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        db().commit()


# ---------------------------------------------------------------- marked levels (reference only)

def add_marked_level(price: float, tf: str, label: str = "", note: str = "") -> dict:
    """A level the user (or the assistant on their behalf) wants shown on the chart, e.g. a close from another
    provider. Purely a reference: the engine never uses it for setups."""
    with _lock:
        cur = db().execute("INSERT INTO marked_levels (price, tf, label, note, created_at) VALUES (?,?,?,?,?)",
                           (float(price), tf, label.strip(), note.strip(), time.time()))
        db().commit()
        return {"id": cur.lastrowid, "price": float(price), "tf": tf, "label": label.strip(), "note": note.strip()}


def list_marked_levels() -> list[dict]:
    with _lock:
        return [dict(r) for r in db().execute(
            "SELECT id, price, tf, label, note, created_at FROM marked_levels ORDER BY price DESC").fetchall()]


def delete_marked_level(level_id: int) -> None:
    with _lock:
        db().execute("DELETE FROM marked_levels WHERE id=?", (level_id,))
        db().commit()


# ---------------------------------------------------------------- AMDX: what the month / week / day is

def period_keys(t: int) -> dict[str, str]:
    """The keys of the month, week (its Monday) and day that contain feed time t."""
    import datetime as _dt
    d = _dt.datetime.fromtimestamp(t, _dt.UTC).date()
    return {"month": d.strftime("%Y-%m"), "week": (d - _dt.timedelta(days=d.weekday())).isoformat(), "day": d.isoformat()}


def set_phase(scope: str, key: str, phase: str, note: str = "") -> dict:
    """phase: A (accumulation), M (manipulation), D (distribution) or X (reversal / continuation); '?' clears it."""
    if scope not in ("month", "week", "day"):
        raise ValueError("scope must be month, week or day")
    phase = phase.strip().upper()
    with _lock:
        if phase in ("?", "", "UNKNOWN"):
            db().execute("DELETE FROM market_phases WHERE scope=? AND key=?", (scope, key))
        elif phase in ("A", "M", "D", "X"):
            db().execute("INSERT INTO market_phases (scope, key, phase, note, updated_at) VALUES (?,?,?,?,?) "
                         "ON CONFLICT(scope, key) DO UPDATE SET phase=excluded.phase, note=excluded.note, updated_at=excluded.updated_at",
                         (scope, key, phase, note.strip(), time.time()))
        else:
            raise ValueError("phase must be A, M, D, X or ?")
        db().commit()
    return {"scope": scope, "key": key, "phase": None if phase in ("?", "", "UNKNOWN") else phase}


def get_phase(scope: str, key: str) -> str | None:
    with _lock:
        r = db().execute("SELECT phase FROM market_phases WHERE scope=? AND key=?", (scope, key)).fetchone()
    return r["phase"] if r else None


# ---------------------------------------------------------------- the user's own setups

def add_custom_level(d: dict) -> dict:
    """A level the USER says exists (e.g. a close on another provider's chart). The engine uses it like one of its own levels of that
    timeframe (tf: 1w / 1d / 4h): setups can start from it and it can be a target. direction: "buy" (support) or "sell" (resistance).
    time: when it started counting (feed clock); if omitted, 7 days back so recent price action is judged against it too."""
    if d.get("direction") not in ("buy", "sell"):
        raise ValueError('direction must be "buy" (support) or "sell" (resistance)')
    if d.get("tf", "1d") not in ("1w", "1d", "4h"):
        raise ValueError("tf must be 1w, 1d or 4h (the timeframe the level acts as)")
    since = d.get("time")
    if not since:
        last = data.get_candles("1h")[-1]["time"]
        since = last - 7 * 86400
    with _lock:
        cur = db().execute(
            "INSERT INTO custom_levels (created_at, source, direction, price, time, tf, label, note)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), d.get("source") or data.active_source(), d["direction"], float(d["price"]),
             int(since), d.get("tf", "1d"), d.get("label", ""), d.get("note", "")))
        db().commit()
        row = get_custom_level(cur.lastrowid)
    from . import levels
    levels.forget_custom_cache()
    return row


def get_custom_level(level_id: int) -> dict | None:
    with _lock:
        r = db().execute("SELECT * FROM custom_levels WHERE id=?", (level_id,)).fetchone()
    return dict(r) if r else None


def list_custom_levels() -> list[dict]:
    with _lock:
        return [dict(r) for r in db().execute("SELECT * FROM custom_levels ORDER BY created_at DESC").fetchall()]


def delete_custom_level(level_id: int) -> None:
    """Remove the level and the setups that were started from it (they no longer exist without it)."""
    with _lock:
        db().execute("DELETE FROM custom_levels WHERE id=?", (level_id,))
        db().execute("DELETE FROM setups WHERE json_extract(data, '$.custom_level.id') = ?", (level_id,))
        db().commit()
    from . import levels
    levels.forget_custom_cache()


def count_custom_levels() -> int:
    with _lock:
        return db().execute("SELECT COUNT(*) AS n FROM custom_levels").fetchone()["n"]


def get_manual_setup(setup_id: int) -> dict | None:
    with _lock:
        r = db().execute("SELECT * FROM manual_setups WHERE id=?", (setup_id,)).fetchone()
    return dict(r) if r else None


def list_manual_setups() -> list[dict]:
    with _lock:
        return [dict(r) for r in db().execute("SELECT * FROM manual_setups ORDER BY id").fetchall()]


def delete_manual_setup(setup_id: int) -> None:
    with _lock:
        db().execute("DELETE FROM manual_setups WHERE id=?", (setup_id,))
        db().execute("DELETE FROM setups WHERE id=?", (f"manual-{setup_id}",))   # its stored history goes with it
        db().commit()


# ---------------------------------------------------------------- chart annotations drawn by the assistant

def add_annotations(session: str, items: list[dict]) -> None:
    with _lock:
        for it in items:
            db().execute("INSERT INTO annotations (session, created_at, item) VALUES (?,?,?)", (session, time.time(), json.dumps(it)))
        db().commit()


def list_annotations(session: str) -> list[dict]:
    with _lock:
        return [json.loads(r["item"]) for r in db().execute("SELECT item FROM annotations WHERE session=? ORDER BY id", (session,)).fetchall()]


def clear_annotations(session: str) -> None:
    with _lock:
        db().execute("DELETE FROM annotations WHERE session=?", (session,))
        db().commit()


# ---------------------------------------------------------------- corrections: "that candle was a feed glitch"

def set_override(setup_id: str, ignore_until: int, note: str = "") -> dict:
    """Tell the engine to ignore every candle up to and including `ignore_until` when judging this setup's stop, target and
    fill (a data glitch, e.g. a spurious wick). The stored row is dropped so it is rebuilt without the glitch."""
    with _lock:
        row = db().execute("SELECT ignore_until FROM overrides WHERE setup_id=?", (setup_id,)).fetchone()
        until = max(int(ignore_until), row["ignore_until"] if row else 0)
        db().execute(
            "INSERT INTO overrides (setup_id, ignore_until, note, created_at) VALUES (?,?,?,?) "
            "ON CONFLICT(setup_id) DO UPDATE SET ignore_until=excluded.ignore_until, note=excluded.note",
            (setup_id, until, note.strip(), time.time()))
        db().execute("DELETE FROM setups WHERE id=?", (setup_id,))
        db().commit()
    return {"setup_id": setup_id, "ignore_until": until, "note": note.strip()}


def get_overrides() -> dict[str, dict]:
    with _lock:
        return {r["setup_id"]: {"ignore_until": r["ignore_until"], "note": r["note"]}
                for r in db().execute("SELECT setup_id, ignore_until, note FROM overrides").fetchall()}


def clear_override(setup_id: str) -> None:
    with _lock:
        db().execute("DELETE FROM overrides WHERE setup_id=?", (setup_id,))
        db().execute("DELETE FROM setups WHERE id=?", (setup_id,))
        db().commit()


# ---------------------------------------------------------------- one chat per trading day

def list_chat_days() -> list[dict]:
    """Every trading day that has a chat, newest first, with a one-line preview of what was asked."""
    with _lock:
        rows = db().execute(
            "SELECT session, COUNT(*) n, MIN(created_at) first, MAX(created_at) last FROM messages "
            "WHERE session LIKE 'day:%' GROUP BY session ORDER BY session DESC").fetchall()
        out = []
        for r in rows:
            first = db().execute("SELECT content FROM messages WHERE session=? AND role='user' ORDER BY id LIMIT 1", (r["session"],)).fetchone()
            out.append({"session": r["session"], "day": r["session"][4:], "messages": r["n"], "last": r["last"],
                        "preview": (first["content"][:90] if first else "")})
        return out


def list_chats(limit: int = 80) -> list[dict]:
    """Every saved chat (a new chat per topic, plus the old one-per-day chats), newest first, titled by the first thing asked."""
    with _lock:
        rows = db().execute(
            "SELECT session, COUNT(*) n, MAX(created_at) last FROM messages "
            "WHERE (session LIKE 'chat:%' OR session LIKE 'day:%') GROUP BY session ORDER BY last DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            first = db().execute("SELECT content FROM messages WHERE session=? AND role='user' ORDER BY id LIMIT 1", (r["session"],)).fetchone()
            title = (first["content"] if first else "").strip().replace("\n", " ")
            for pre in ("Let's go through setup ", "Lets go through setup "):
                if title.startswith(pre):
                    title = "Setup " + title[len(pre):].split(" ")[0]
                    break
            out.append({"session": r["session"], "messages": r["n"], "last": r["last"], "title": title[:60] or "(empty)"})
        return out


def migrate_sessions_to_days(offset_seconds: int) -> dict:
    """One-time: chats and drawings saved under a browser id move to the trading day they were made on.
    The day is the date on the feed's clock (real time + the feed's offset from real time)."""
    import datetime as dt
    moved = 0
    with _lock:
        for table in ("messages", "annotations"):
            rows = db().execute(
                f"SELECT id, created_at FROM {table} WHERE session NOT LIKE 'day:%' AND session NOT LIKE 'test-%'").fetchall()
            for r in rows:
                day = dt.datetime.fromtimestamp(r["created_at"] + offset_seconds, dt.UTC).strftime("%Y-%m-%d")
                db().execute(f"UPDATE {table} SET session=? WHERE id=?", ("day:" + day, r["id"]))
                moved += 1
        db().commit()
    return {"moved": moved}


# ---------------------------------------------------------------- performance: is it profitable?

def _block(rows: list[dict]) -> dict:
    """Won / lost / missed counts and R for one group of setups.

    R is in units of risk: a win pays its planned reward-to-risk at the target that was hit, a loss costs exactly 1R, a
    missed setup (price ran to the target without ever retracing to the entry) is not a trade, so it costs and pays nothing.
    Net R = R won - R lost.
    """
    won = [r for r in rows if r["outcome"] == "target"]
    lost = [r for r in rows if r["outcome"] == "stopped"]
    missed = [r for r in rows if r["outcome"] == "missed"]
    skipped = [r for r in rows if r["outcome"] == "skipped"]
    live = [r for r in rows if not r["finished"] and r["plan_status"] in ("open", "waiting")]
    win_r = round(sum(r["result_r"] for r in won), 2)
    loss_r = round(-sum(r["result_r"] for r in lost), 2)          # shown as a positive amount lost
    n = len(won) + len(lost)
    lo, hi = _wilson(len(won), n)
    avg_win = round(win_r / len(won), 2) if won else None
    return {
        "won": len(won), "lost": len(lost), "missed": len(missed), "skipped": len(skipped), "live": len(live), "trades": n,
        "win_r": win_r, "loss_r": loss_r, "net_r": round(win_r - loss_r, 2),
        "avg_win_r": avg_win, "avg_loss_r": 1.0 if lost else None,
        "win_rate": round(len(won) / n, 3) if n else None, "win_rate_95ci": [lo, hi] if n else None,
        "expectancy_r": round((win_r - loss_r) / n, 2) if n else None,
        "profit_factor": round(win_r / loss_r, 2) if loss_r else None,
        # the share of trades that must win just to break even at this average win size: 1 / (1 + avg win R)
        "breakeven_win_rate": round(1 / (1 + avg_win), 3) if avg_win else None,
        "missed_planned_r": round(sum(r["rr"] or 0 for r in missed), 2),
    }


def performance(scope: str = "engine", since: int | None = None, watched_live_only: bool = False, source: str | None = None) -> dict:
    """Results of the stored setups. scope: engine (the engine's own setups), mine (your own setups), all.
    since: only setups that ENDED at or after this feed-clock time. watched_live_only: leave out setups rebuilt from history.
    source: count one data feed only. The same market move is stored once per feed (their clocks and prices differ), so counting
    both feeds together would count some trades twice."""
    import datetime as dt
    with _lock:
        rows = [_summary(r) for r in db().execute("SELECT * FROM setups").fetchall()]
    if source:
        rows = [r for r in rows if r["source"] == source]
    if scope == "engine":      # setups from levels the user marked are the user's, not the engine's own: kept out of its win rate
        rows = [r for r in rows if r["tier"] != "manual" and not r["custom_level"]]
    elif scope == "mine":
        rows = [r for r in rows if r["tier"] == "manual" or r["custom_level"]]
    if watched_live_only:
        rows = [r for r in rows if not r["backfilled"]]
    if since is not None:
        rows = [r for r in rows if (r["plan_resolved_time"] or r["resolved_time"] or 0) >= since or not r["finished"]]
    finished = sorted([r for r in rows if r["outcome"] in ("target", "stopped")], key=lambda r: r["plan_resolved_time"] or r["resolved_time"] or 0)

    total = _block(rows)
    by_kind = []
    kinds = defaultdict(list)
    for r in rows:
        kinds[f"{r['tier']}/{r['kind']}"].append(r)
    for k, rs in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        by_kind.append({"group": k, **_block(rs)})

    curve, cum, peak, max_dd = [], 0.0, 0.0, 0.0
    for r in finished:
        cum += r["result_r"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        curve.append({"time": r["plan_resolved_time"] or r["resolved_time"], "r": round(cum, 2), "id": r["id"], "result_r": r["result_r"]})
    total["max_drawdown_r"] = round(max_dd, 2)

    days = defaultdict(list)
    for r in rows:
        t = r["plan_resolved_time"] or r["resolved_time"]
        if r["finished"] and t:
            days[dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d")].append(r)
    by_day = [{"day": d, **_block(rs)} for d, rs in sorted(days.items(), reverse=True)]

    return {
        "scope": scope, "feed": source or "all feeds (some trades are counted twice)", "watched_live_only": watched_live_only, "total": total, "by_kind": by_kind, "by_day": by_day, "curve": curve,
        "rules_version": meta_get("engine_rules_version"),
        "setups_counted": len(rows), "rebuilt_from_history": sum(1 for r in rows if r["backfilled"]),
        "watched_live": sum(1 for r in rows if not r["backfilled"]),
        "note": ("Planned R at the target that was hit (targets move), -1R per loss, missed setups cost nothing. Before spread, slippage and "
                 "commission. Most stored setups were rebuilt from history with today's rules, not watched live, so this is an in-sample "
                 "result, not a forecast."),
    }


def refile_days(offset_seconds: int) -> dict:
    """Re-file every saved chat message and drawing under the user's OWN calendar day: day = date(created_at + their UTC offset)."""
    import datetime as dt
    moved = 0
    with _lock:
        for table in ("messages", "annotations"):
            for r in db().execute(f"SELECT id, session, created_at FROM {table} WHERE session LIKE 'day:%'").fetchall():
                new = "day:" + dt.datetime.fromtimestamp(r["created_at"] + offset_seconds, dt.UTC).strftime("%Y-%m-%d")
                if new != r["session"]:
                    db().execute(f"UPDATE {table} SET session=? WHERE id=?", (new, r["id"]))
                    moved += 1
        db().commit()
    return {"moved": moved}


# ---------------------------------------------------------------- rule versions: never mix results from different rules

def ensure_rules_version(version: str) -> dict:
    """If the engine's rules changed since the stored setups were made, move the old ones to `setups_archive` (kept, labelled with the
    rules they were made under) and clear the live table so it is rebuilt under the new rules. Win rates then never mix two rule sets."""
    with _lock:
        have = meta_get("engine_rules_version")
        if have == version:
            return {"changed": False, "version": version}
        c = db()
        c.execute("""CREATE TABLE IF NOT EXISTS setups_archive (id TEXT, tier TEXT, kind TEXT, trigger_tf TEXT, direction TEXT, trigger_time INTEGER,
                     status TEXT, plan_status TEXT, outcome TEXT, result_r REAL, invalid_reason TEXT, source TEXT, symbol TEXT,
                     first_seen REAL, last_seen REAL, resolved_at_first_seen INTEGER, frozen INTEGER, data TEXT,
                     archived_at REAL, rules_version TEXT)""")
        n = c.execute("SELECT COUNT(*) n FROM setups").fetchone()["n"]
        c.execute("INSERT INTO setups_archive SELECT *, ?, ? FROM setups", (time.time(), have or "1"))
        c.execute("DELETE FROM setups")
        c.commit()
        meta_set("engine_rules_version", version)
        return {"changed": True, "version": version, "archived": n, "old_version": have or "1"}
