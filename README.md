# Gold Trading Assistant

Browser app: XAUUSD candlestick chart on the left, LLM chat on the right. Discussion only, no trade execution.

## Run

```
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
cp .env.example .env        # then put your ANTHROPIC_API_KEY in .env
.venv/bin/uvicorn backend.app.main:app --port 8000
```

Open http://localhost:8000

## Layout

- `backend/app/data.py`: fetches and validates candles (Yahoo `GC=F` for now), computes the snapshot (last close, ATR14, SMA20, recent candles) that is given to the LLM.
- `backend/app/llm.py`: system prompt and streaming chat. The LLM only sees numbers computed in `data.py`.
- `backend/app/main.py`: FastAPI routes (`/api/candles`, `/api/chat`) and static frontend.
- `frontend/index.html`: chart (lightweight-charts) + chat panel.

## Known limits

- `GC=F` is COMEX gold futures, not spot XAUUSD, and Yahoo data is delayed. Fine for prototyping; swap `get_candles` for a broker feed later.
- Chat history lives in the browser tab and is lost on refresh.

## MT5 data feed (matches your MT5 chart exactly)

The app can use your MT5 broker's candles and ticks instead of Twelve Data. MT5 has no public data API,
so an Expert Advisor inside the MT5 desktop terminal pushes the data to this app.

1. Install the MetaTrader 5 desktop app (macOS or Windows) and log in to your broker. Web terminals cannot run EAs.
2. In MT5: File > Open Data Folder > MQL5 > Experts, copy `mt5/TradingAssistantFeed.mq5` there, open it in MetaEditor (F4) and compile (F7).
3. In MT5: Tools > Options > Expert Advisors > tick "Allow WebRequest for listed URL" and add `http://127.0.0.1:8000`.
4. Open an XAUUSD chart (any timeframe), drag the EA onto it, make sure Algo Trading is on.
5. The app switches to MT5 automatically; the status line shows "MT5 <symbol>". Times are broker server time, like your MT5 chart.

Optional: set `MT5_SECRET` in `.env` and the same value in the EA's `Secret` input. `DATA_SOURCE=twelvedata` ignores MT5; `DELETE /api/mt5` forgets what the EA sent.

## History, notes and the assistant's memory

- Every setup the engine identifies is stored permanently in `data/assistant.db` (SQLite): the evidence behind it, the plan,
  every target change and the result. A row is frozen once its outcome is final. Back this file up; it is the history.
  The server saves new setups every 30 seconds. Setups found when the server first sees them already finished are tagged
  "rebuilt" (reconstructed from candles, not watched live).
- Chat tab: ask why a setup was identified (or press **why?** on a live setup). The assistant looks the setup up and explains
  from the stored evidence. The conversation is saved per browser.
- History tab: every stored setup with its result, plus win rate / average R with a 95% range per kind.
- Notes tab: lessons you teach it (always followed) and observations it proposes from the statistics (unconfirmed until you
  confirm them). "Generate observations" analyses the history; it needs at least 20 finished trades.
- The model is not retrained. It "learns" through these saved notes, which are put in front of it in every conversation.

## Your own setups, voice and drawings

- **Your own setups**: the card shows engine setups only. If you take a trade the engine did not flag (for example off a level from
  another provider), press "＋ Add my own setup" on the card, or tell the assistant. It is then tracked like any other setup
  (dynamic target, stop, outcome), shown on the chart and stored in the history as "Your setup". It never changes the headline,
  which follows the engine. A setup is only shown while the same data feed is active that its trigger time was recorded on.
- **Talk to it**: 🎤 Talk uses your browser's speech recognition (Chrome, Edge and Safari; Chrome sends the audio to Google).
  "auto-send" sends when you stop talking; "hands-free" starts listening again after the assistant finishes.
- **Hear it**: 🔈 Voice reads the answer aloud as it streams, sentence by sentence, using your browser's built-in voices (pick one
  in the dropdown). ■ Stop cuts it off.
- **Drawings**: while it explains, the assistant draws lines, zones, markers and trendlines on the chart, each right after the
  sentence that mentions it (and in step with the speech when voice is on). They are saved per chat session; "Clear drawings"
  removes them.

## One chat per trading day, corrections, dictation

- **One chat per trading day.** The day is the date on the data feed's clock, so it rolls over when the chart's day does. A new day
  starts an empty chat; the day picker above the card opens earlier days read-only ("Back to today" returns). Lessons and notes carry
  over. The assistant can also read earlier days itself (`get_chat_day`). Drawings are saved with the day they were made on.
- **"not really hit"** (on a finished trade in the card or the History tab): use it when the feed showed a wick that did not happen on
  your own chart and the engine stopped or filled a trade because of it. The engine then ignores that candle (and earlier ones) for
  that one setup and carries on; a real later hit still counts. Undo with `DELETE /api/overrides/<setup id>`. The assistant can do it
  too (`ignore_feed_glitch`). It is per setup, not a global fix of the feed.
- **Dictation** listens until YOU stop it: press 🎤 to start and press it again to stop. There is no silence timer. If the browser
  ends a session by itself it restarts at once and keeps your words; the text appears live in the box. "send when I stop" sends it
  when you press the button (turn it off to edit the text first). Pressing Send/Enter yourself also stops it.

## The three-timeframe level hold ("hold" kind)

For any level the engine looks at three timeframes: **Monthly** level: Monthly / Weekly / Daily. **Weekly** level: Weekly / Daily / 4H.
**Daily** level: Daily / 4H / 1H. Price hits the level (it came down to it for a buy, up to it for a sell, within the last 3 candles of
the confirmation timeframe); a confirmation level on the next timeframe sits at the same price or beyond it (within one ATR of that
timeframe); then a candle on the third timeframe closes over that confirmation level in the reversal direction: the first close beyond
it, a bullish candle for a buy, bearish for a sell. It does **not** have to be a CPLOC. Target = next valid level of the first
timeframe; entry and stop work as for every other setup. It is an extra kind next to the CPLOC kinds, so the History panel shows each
kind's results separately.

Monthly levels need monthly candles. Twelve Data supplies them. For MT5, **recompile and re-attach the EA** (it now also sends MN1);
until then the Monthly tier simply finds nothing and everything else keeps working.

## Level polarity: what each close is a level FOR

What a close is a level for depends on what price did **right after it**, i.e. on the candle that **followed** the level candle:

| Level candle → next candle | The level is for |
|---|---|
| Bullish → Bearish | **Selling** |
| Bearish → Bullish | **Buying** |
| Bullish → Bullish | **Buying** |
| Bearish → Bearish | **Selling** |

So: the next candle went up = a buying level (support), it went down = a selling level (resistance); the level candle's own colour does not
matter, and a doji after it serves both. That is also why the close just before the running candle is not a level yet: the candle after it
has not finished. A buy can only START at a buying level (the level price hits, and its confirmation level), a sell only at a selling one.
Targets are the next valid level of any kind. The chart labels levels ▲ (buying) and ▼ (selling), and the assistant is told never to call a
buying level resistance or a selling level support.

`LEVEL_POLARITY` in `.env`: `origin` (default: the rule above), `on` (also restricts targets: a buy aims at a selling level, a sell at a
buying level), `target` (targets only), `off` (every close alike, the old behaviour). Every rule change is stored as a **rules version**:
results made under older rules are moved to `setups_archive` (kept, labelled) and the live history is rebuilt, so win rates never mix two
rule sets.
