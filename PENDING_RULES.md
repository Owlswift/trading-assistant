# Pending rules (not in the engine)

Rules you have given that are **written down only**. The engine does not use them until you say so; then each one moves to RULES.md with its tag.

Each rule has a tag (R101, R102, …), your words, how it is read, your examples, and anything still open.

---

## R101 · A level candle surrounded by wicks

**Your rule:** When a level is encountered and its candle is surrounded by wicks on both sides, price won't use its close but the close of the candle before it.

**How it is read (from your Monthly chart of Oct 25, 2026):**
- "Surrounded by wicks on both sides" = the level candle's **body sits inside the wicks of its neighbours**: the wick of the candle before it on one side, and the wick of the candle after it on the other.
- Then that candle's close is not the level. The close of the **candle before it** is the level instead.
- It is not about the candle's own wicks: almost every candle has a wick above and below its body, so that reading would replace nearly every level.

**Your examples (Monthly), and what the data gives with "at least 70% of the body inside each neighbour's wick":**

| Candle | O / H / L / C | Body inside the wick before / after | Level becomes |
|---|---|---|---|
| May 2020 | 1686.02 / 1765.08 / 1670.40 / 1730.41 | 100% (April's upper wick) / 100% (June's lower wick) | **1685.98**, the April 2020 close |
| Oct 2020 | 1887.32 / 1933.18 / 1859.92 / 1878.37 | 77% (September's lower wick) / 90% (November's upper wick) | **1885.23**, the September 2020 close |

From July 2019 to July 2021 these are the only two Monthly candles it catches, the same two you marked. April 2020 (your "level to consider") is not caught.

**Settled (Oct 25, 2026):** you confirmed the reading and the close to use. The 70% cover stays as the measure.
The close of the candle before takes its kind (buying / selling) from the candle after it, as usual. Here: April 2020's close 1685.98, May went up, so a buying level.

**Status:** confirmed, not in the engine yet.

---

## R102 · The first FVG after a reversal decides

**Your rule:** When a reversal occurs and the first FVG doesn't push price back, then the rest of the FVGs become useless.

**How I read it:**
- A reversal happens (a level holds and price turns, for example the Monthly buy on a Monthly buying level).
- On its way, price meets FVGs of the OLD direction (bearish gaps after a buy reversal, bullish gaps after a sell reversal).
- The **first** of those that price gets into is the test. If it does not push price back (no rejection: no candle of the old trend's colour closing beyond the gap, on the tap candle or the next 3), it has failed.
- Then **every other** old-direction FVG further along is useless for that move: no exit on it (B), no trade the other way from it (C, D), and it does not block trades.
- If the first one does push price back, it works as today (exit, trade the other way).

**Open:**
1. Is "doesn't push price back" the same as no rejection (as above), or must price close right through the gap?
2. Which FVGs: only those of the timeframe of the trade's signal candle (Daily for a Monthly trade), and useless until when: until that trade ends, or until the next reversal?

**Status:** written down, not in the engine.
