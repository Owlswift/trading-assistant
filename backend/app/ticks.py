"""Live tick hub: one upstream Twelve Data WebSocket, fanned out to many browser sockets."""
import asyncio
import json
import os
import time

import websockets

from . import mt5

SYMBOL = "XAU/USD"
UPSTREAM = "wss://ws.twelvedata.com/v1/quotes/price?apikey={key}"


class TickHub:
    def __init__(self) -> None:
        self.subscribers: set[asyncio.Queue] = set()
        self.last: dict | None = None  # {"t": unix seconds, "p": price}
        self.status = "idle"  # idle | connecting | live | error: <msg>
        self._task: asyncio.Task | None = None
        self._watch: asyncio.Task | None = None
        self.mt5_seen = 0.0   # when the last MT5 tick arrived (0 = never)

    def start(self) -> None:
        if os.getenv("TWELVE_DATA_API_KEY") and self._task is None:
            self._task = asyncio.create_task(self._run())
        if self._watch is None:
            self._watch = asyncio.create_task(self._watchdog())

    async def stop(self) -> None:
        for t in (self._task, self._watch):
            if t:
                t.cancel()

    def mt5_fresh(self) -> bool:
        return self.mt5_seen > 0 and time.time() - self.mt5_seen < 15

    @staticmethod
    def mt5_is_source() -> bool:
        """MT5 candles are in use: ticks must then also come from MT5 (broker time), never Twelve Data (UTC)."""
        return bool(mt5.get("1h"))

    def mt5_tick(self, t: int, price: float) -> None:
        """A tick from the MT5 EA (t = broker server time, as MT5 shows it)."""
        self.mt5_seen = time.time()
        tick = {"t": int(t), "p": float(price)}
        if self.status != "live":
            self.status = "live"
            self._publish({"status": "live"})
        if tick != self.last:
            self.last = tick
            self._publish(tick)

    async def _watchdog(self) -> None:
        """If MT5 was feeding ticks and goes quiet, say so (Twelve Data ticks take over only if configured)."""
        while True:
            await asyncio.sleep(5)
            if self.mt5_is_source() and not self.mt5_fresh() and self.status == "live":
                self.status = "MT5 feed stale"
                self._publish({"status": self.status})

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def _publish(self, msg: dict) -> None:
        for q in list(self.subscribers):
            if q.full():  # slow client: drop its oldest tick rather than block everyone
                q.get_nowait()
            q.put_nowait(msg)

    async def _heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(10)
            await ws.send(json.dumps({"action": "heartbeat"}))

    async def _run(self) -> None:
        delay = 2
        while True:
            self.status = "connecting"
            self._publish({"status": self.status})
            try:
                async with websockets.connect(UPSTREAM.format(key=os.environ["TWELVE_DATA_API_KEY"])) as ws:
                    await ws.send(json.dumps({"action": "subscribe", "params": {"symbols": SYMBOL}}))
                    hb = asyncio.create_task(self._heartbeat(ws))
                    try:
                        async for raw in ws:
                            m = json.loads(raw)
                            ev = m.get("event")
                            if ev == "subscribe-status":
                                if m.get("fails"):
                                    raise RuntimeError(f"subscribe failed: {m['fails']}")
                                delay = 2
                                if not self.mt5_is_source():
                                    self.status = "live"
                                    self._publish({"status": self.status})
                            elif ev == "price":
                                if self.mt5_is_source():
                                    continue  # Twelve Data times are real UTC, MT5's are broker time: never mix
                                tick = {"t": int(m["timestamp"]), "p": float(m["price"])}
                                if tick != self.last:  # upstream repeats identical ticks
                                    self.last = tick
                                    self._publish(tick)
                            elif ev == "error" or m.get("status") == "error":
                                raise RuntimeError(m.get("message", m))
                    finally:
                        hb.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.status = f"error: {e}"
                self._publish({"status": self.status})
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)


hub = TickHub()
