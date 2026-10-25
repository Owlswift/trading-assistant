"""Candle store fed by the MT5 Expert Advisor (mt5/TradingAssistantFeed.mq5).

MT5 is the source of truth when it is connected: candles match the user's MT5 chart exactly.
Times are kept EXACTLY as MT5 reports them: broker server time stored as if it were UTC, so the chart's
time labels equal MT5's, and day / 4H / week boundaries fall where the broker puts them. (Converting to
real UTC would need the broker's historical DST offsets and would put candles off by an hour at DST
changes.) Every comparison in the app is relative, so this is consistent end to end.
"""
import json
import os
import threading
import time
from pathlib import Path

CACHE = Path(__file__).resolve().parents[2] / "data" / "mt5_cache.json"
MAX_CANDLES = 80000
TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d", "1w", "1mo")
MIN_HISTORY = 50   # fewer candles than this on a timeframe -> ask the EA to resend history

_lock = threading.Lock()
_store: dict[str, list[dict]] = {}
_last_push = 0.0
_last_save = 0.0
_symbol = ""


def _load() -> None:
    global _symbol
    try:
        raw = json.loads(CACHE.read_text())
        _store.update({tf: raw["frames"][tf] for tf in raw.get("frames", {}) if tf in TIMEFRAMES})
        _symbol = raw.get("symbol", "")
    except Exception:
        pass


def _save(force: bool = False) -> None:
    global _last_save
    if not force and time.time() - _last_save < 10:
        return
    try:
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps({"symbol": _symbol, "frames": _store}))
        _last_save = time.time()
    except Exception:
        pass


def upsert(frames: dict[str, list[list[float]]], symbol: str = "") -> None:
    """frames: {"1h": [[time, open, high, low, close], ...]} ; merges by candle time."""
    global _last_push, _symbol
    with _lock:
        for tf, rows in frames.items():
            if tf not in TIMEFRAMES:
                continue
            merged = {c["time"]: c for c in _store.get(tf, [])}
            for t, o, h, l, c in rows:
                o, h, l, c = float(o), float(h), float(l), float(c)
                if h < max(o, c) or l > min(o, c):
                    continue  # malformed candle
                merged[int(t)] = {"time": int(t), "open": o, "high": h, "low": l, "close": c}
            _store[tf] = [merged[t] for t in sorted(merged)][-MAX_CANDLES:]
        _last_push = time.time()
        if symbol:
            _symbol = symbol
        _save()


def get(timeframe: str) -> list[dict] | None:
    with _lock:
        c = _store.get(timeframe)
        return list(c) if c else None


def needs_history() -> bool:
    with _lock:
        # monthly history is short by nature, and an EA from before monthly support never sends it: do not ask for history forever
        return any(len(_store.get(tf, [])) < (3 if tf == "1mo" else MIN_HISTORY) for tf in TIMEFRAMES if tf != "1mo" or _store.get("1mo"))


def status() -> dict:
    with _lock:
        return {
            "symbol": _symbol,
            "candles": {tf: len(_store.get(tf, [])) for tf in TIMEFRAMES},
            "seconds_since_push": round(time.time() - _last_push, 1) if _last_push else None,
        }


def clear() -> None:
    global _last_push, _symbol
    with _lock:
        _store.clear()
        _last_push, _symbol = 0.0, ""
        _save(force=True)


def secret_ok(header_value: str | None) -> bool:
    want = os.getenv("MT5_SECRET", "")
    return not want or header_value == want


_load()
