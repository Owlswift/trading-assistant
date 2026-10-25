"""Market data service: fetches gold OHLC candles and validates them.

Primary source is Twelve Data (spot XAU/USD) when TWELVE_DATA_API_KEY is set.
Without a key it falls back to Yahoo Finance GC=F (COMEX futures, not spot).
"""
import os
import threading
import time
from collections import defaultdict, deque

import httpx
import pandas as pd
import yfinance as yf

from . import mt5

SYMBOL = "GC=F"

# timeframe -> (yfinance interval, yfinance period, resample rule or None)
TIMEFRAMES = {
    "5m": ("5m", "30d", None),
    "15m": ("15m", "60d", None),
    "1h": ("1h", "180d", None),
    "4h": ("1h", "360d", "4h"),
    "1d": ("1d", "5y", None),
    "1w": ("1wk", "10y", None),
    "1mo": ("1mo", "max", None),
}

# Twelve Data free tier: 8 requests/min, 800/day. Each refresh costs 1 request, so keep this
# generous; lower it only on a paid plan.
REFRESH_SECONDS = int(os.getenv("DATA_REFRESH_SECONDS", "60"))

TD_INTERVALS = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day", "1w": "1week", "1mo": "1month"}
TD_HISTORY = {"5m": 1500, "15m": 1500, "1h": 1500, "4h": 1500, "1d": 1000, "1w": 600, "1mo": 240}

_cache: dict[str, dict] = {}  # timeframe -> {"candles": [...], "fetched": ts}

# Twelve Data allows 8 requests/minute on the free plan. Stay under it (7) by sleeping when needed,
# and make concurrent callers for the same timeframe share one fetch.
MAX_CALLS_PER_MIN = 7
_calls: deque[float] = deque()
_rate_lock = threading.Lock()
_tf_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


def _throttle() -> None:
    with _rate_lock:
        now = time.time()
        while _calls and now - _calls[0] >= 60:
            _calls.popleft()
        if len(_calls) >= MAX_CALLS_PER_MIN:
            time.sleep(max(0.0, 60 - (now - _calls[0]) + 0.5))
            _calls.popleft()
        _calls.append(time.time())


def source() -> str:
    return "twelvedata" if os.getenv("TWELVE_DATA_API_KEY") else "yahoo"


def _twelve(timeframe: str, size: int, _retry: bool = False) -> list[dict]:
    _throttle()
    r = httpx.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": "XAU/USD",
            "interval": TD_INTERVALS[timeframe],
            "outputsize": size,
            "order": "ASC",
            "timezone": "UTC",
            "apikey": os.environ["TWELVE_DATA_API_KEY"],
        },
        timeout=20,
    )
    body = r.json()
    if "run out of API credits for the current minute" in str(body.get("message", "")) and not _retry:
        time.sleep(62)   # Twelve Data's own per-minute window (e.g. after a server restart reset our counter)
        return _twelve(timeframe, size, _retry=True)
    if body.get("status") == "error" or "values" not in body:
        raise RuntimeError(f"Twelve Data: {body.get('message', body)}")
    out = []
    for v in body["values"]:
        ts = pd.Timestamp(v["datetime"], tz="UTC")
        o, h, l, c = (round(float(v[k]), 2) for k in ("open", "high", "low", "close"))
        if h < max(o, c) or l > min(o, c):
            continue  # malformed candle
        out.append({"time": int(ts.timestamp()), "open": o, "high": h, "low": l, "close": c})
    return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Drop malformed candles (high/low inconsistent with open/close).
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    return df[~bad]


def _yahoo(timeframe: str) -> list[dict]:
    interval, period, resample = TIMEFRAMES[timeframe]
    df = yf.download(SYMBOL, interval=interval, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance")
    df = _clean(df)

    if resample:
        df = df.resample(resample, origin="start_day").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open"])

    # lightweight-charts wants UTC unix seconds.
    idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return [
        {
            "time": int(ts.timestamp()),
            "open": round(float(r.open), 2),
            "high": round(float(r.high), 2),
            "low": round(float(r.low), 2),
            "close": round(float(r.close), 2),
        }
        for ts, r in zip(idx, df.itertuples())
    ]


def active_source() -> str:
    """"mt5" once the MT5 Expert Advisor has delivered candles, else Twelve Data / Yahoo.

    DATA_SOURCE=mt5 forces MT5 (errors until it has data); DATA_SOURCE=twelvedata ignores MT5.
    """
    mode = os.getenv("DATA_SOURCE", "auto")
    if mode == "twelvedata":
        return source()
    if mode == "mt5" or mt5.get("1h"):
        return "mt5"
    return source()


def get_candles(timeframe: str) -> list[dict]:
    return tag_direction(_load(timeframe))


def tag_direction(candles: list[dict]) -> list[dict]:
    """A candle is bullish when it CLOSED ABOVE the previous candle's close, bearish when below (not open vs close: broker feeds
    open with gaps, so open->close can disagree with the real move). Stored as c["dir"]; the first candle falls back to open vs close.
    The whole list is tagged once; after that only the newest candles (the only ones that can still change) are re-tagged, because
    every part of the engine asks for the candles many thousands of times."""
    n = len(candles)
    if not n:
        return candles
    start = max(0, n - 3) if candles[-1].get("_n") == n and "dir" in candles[-1] else 0
    prev = candles[start - 1]["close"] if start > 0 else None
    for c in candles[start:]:
        ref = prev if prev is not None else c["open"]
        c["dir"] = "bull" if c["close"] > ref else "bear" if c["close"] < ref else "doji"
        prev = c["close"]
    candles[-1]["_n"] = n
    return candles


def direction_conflicts(timeframe: str, n: int = 60) -> list[dict]:
    """Recent candles whose open-to-close colour disagrees with close-vs-previous-close (the rule the engine uses). A broker feed that
    opens with a gap (e.g. after its daily break) makes them differ; the engine follows the close-vs-previous-close reading."""
    c = get_candles(timeframe)
    out = []
    for i in range(max(1, len(c) - n), len(c) - 1):          # completed candles only
        k, p = c[i], c[i - 1]
        by_open = "bull" if k["close"] > k["open"] else "bear" if k["close"] < k["open"] else "doji"
        if by_open != k["dir"]:
            out.append({"time": k["time"], "open": k["open"], "prev_close": p["close"], "close": k["close"],
                        "gap": round(k["open"] - p["close"], 2), "by_open_close": by_open, "by_prev_close": k["dir"]})
    return out


def _load(timeframe: str) -> list[dict]:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if active_source() == "mt5":
        candles = mt5.get(timeframe)
        if not candles:
            raise RuntimeError(f"Waiting for MT5 to send {timeframe} candles (is the EA attached?)")
        return candles

    def fresh() -> list[dict] | None:
        e = _cache.get(timeframe)
        return e["candles"] if e and time.time() - e["fetched"] < REFRESH_SECONDS else None

    if (hit := fresh()) is not None:
        return hit

    with _tf_locks[timeframe]:  # one fetch per timeframe at a time
        if (hit := fresh()) is not None:  # another caller refreshed while we waited
            return hit
        entry = _cache.get(timeframe)
        try:
            if source() == "twelvedata":
                if entry is None:
                    candles = _twelve(timeframe, TD_HISTORY[timeframe])
                else:  # incremental refresh: re-fetch only the newest few candles and merge
                    merged = {c["time"]: c for c in entry["candles"]}
                    merged.update({c["time"]: c for c in _twelve(timeframe, 3)})
                    candles = [merged[t] for t in sorted(merged)]
            else:
                candles = _yahoo(timeframe)
        except Exception:
            if entry is None:
                raise
            entry["fetched"] = time.time() - REFRESH_SECONDS + 10  # keep serving old candles, retry in ~10 s
            return entry["candles"]
        _cache[timeframe] = {"candles": candles, "fetched": time.time()}
        return candles


def try_candles(timeframe: str) -> list[dict] | None:
    """Like get_candles, but None when that timeframe is not available yet (e.g. the MT5 EA has not been updated to send monthly candles)."""
    try:
        return get_candles(timeframe)
    except Exception:
        return None


def summarize(candles: list[dict], n: int = 60) -> dict:
    """Compact, factual snapshot handed to the LLM. All numbers are computed here."""
    if not candles:
        return {}
    recent = candles[-n:]
    df = pd.DataFrame(recent)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(df) >= 15 else None
    return {
        "last_close": recent[-1]["close"],
        "last_candle_time": pd.to_datetime(recent[-1]["time"], unit="s", utc=True).isoformat(),
        "window_high": float(df["high"].max()),
        "window_low": float(df["low"].min()),
        "atr14": round(atr14, 2) if atr14 else None,
        "sma20": round(float(df["close"].tail(20).mean()), 2) if len(df) >= 20 else None,
        "recent_candles_ohlc": [
            [pd.to_datetime(c["time"], unit="s", utc=True).strftime("%m-%d %H:%M"), c["open"], c["high"], c["low"], c["close"]]
            for c in recent[-30:]
        ],
    }
