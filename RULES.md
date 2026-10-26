# Trading rules (XAUUSD)

These are the rules the engine applies now (rules version 50). The code that applies them is in `backend/app/`, and each section names the function.
When a rule changes, this file changes with it.

Items marked **[engine choice]** were not given by you; they were picked while building the engine. Tell me which to keep, change or drop.

---

## 1. Levels

- A **level** is the close of a completed candle on the 4H, Daily, Weekly or Monthly chart.
- The close of the candle just **before the running one** is not a level yet (the candle after it has not finished, so its job is not known).
- A level is **broken** once a later candle of the same timeframe **closes** beyond it. Wicks through a level do not break it. A broken level is gone for good.
- **Polarity** (what a level is for) comes from the candle **after** it:
  - next candle went up → **buying level** (a buy can start there)
  - next candle went down → **selling level** (a sell can start there)
  - a doji next candle serves both.
- Candle direction (up / down, green / red for levels) = its close compared with the **previous candle's close**, not its own open.
- A buy can only **start** at a buying level, a sell at a selling level.
- **Targets** must be the opposite kind: a buy aims at a selling level, a sell at a buying level (`LEVEL_POLARITY=on` in `.env`).
- **Your own levels** (custom levels you add) are sticky: they are used as you marked them, whatever the engine thinks of them.

Code: `levels.py`, `frc_levels.py`, `data.tag_direction`.

## 2. CPLOC

A **CPLOC** candle crosses a valid level and closes beyond it:
- bullish: low reaches the level, close above it
- bearish: high reaches the level, close below it

Timeframe pairs: 1H CPLOC → Daily level, 4H CPLOC → Daily (and Weekly) level, 15M CPLOC → 4H level, Daily CPLOC → Monthly level.

Code: `frc_levels.find`.

## 3. Daily-level trades (Daily tier)

| Signal | What happens |
|---|---|
| **1. Reversal** | A 1H CPLOC crosses a valid Daily level. |
| **2. Hold** | Price hits a valid Daily level. A 4H level lies **inside the Daily candle's range** (from its close out to its low for a buy / high for a sell). A 1H candle crosses that 4H level and closes beyond it. The 1H candle must have the trade's colour (green for a buy, red for a sell). It does not have to be a CPLOC. |
| **3. Early** | A 15M CPLOC on a 4H level, after price hit a Daily level, with the 4H level inside that Daily candle's range. |
| **Continuation** | A 15M CPLOC on a 4H level while a 1H reversal is still running, after a 1H break of structure in its direction. |

- **Target**: the next valid Daily level beyond the trigger's close (and beyond the level that was hit), of the opposite kind.
- The target is **dynamic**: it is re-checked on every candle with the levels that exist then (no hindsight). A new level between price and the old target becomes the target.

Code: `_find_reversals`, `_find_hold`, `_hold_candidates`, `_gated_core`, `_find_continuations`.

## 4. Weekly and Monthly levels

**As targets (upgrade):** when a Daily trade happens where a Weekly or Monthly level is involved, it stays the same trade (same entry, same stop) but its target becomes the next valid Weekly (or Monthly) level. "Involved" means a valid Weekly/Monthly level of the right kind that:
- is the Daily level itself, or
- price tapped recently, within **72 hours** before the trigger **[engine choice]**, with the Daily level at or beyond it,

and the Daily level lies between that level and the extreme of the candle that made it, and price has not broken that extreme since. Monthly wins over Weekly.

**As their own trades** (no early signals at these tiers; when both a Daily and a higher trade fire, they are 2 separate trades):

| Tier | Signal 1 | Signal 2 |
|---|---|---|
| Weekly | 4H CPLOC on a Weekly level | Daily level inside the Weekly candle's range + a 4H candle closing beyond it |
| Monthly | Daily CPLOC on a Monthly level | Weekly level inside the Monthly candle's range + a Daily candle closing beyond it |

- **Monthly: the furthest level.** When price runs through several valid Monthly levels of the trade's kind, the setup is taken from the **furthest** one it reached: the lowest buying level for a buy, the highest selling level for a sell. The move is judged on Daily candles from the start of the previous Monthly candle up to the trigger. Both signals are checked only at that level: signal 1 = a Daily CPLOC across it; signal 2 = a Weekly level at or beyond it (inside its candle's range) that a Daily candle crosses and closes beyond. (Weekly and Daily levels: not reviewed yet.)
- Target: the next valid level of that tier (Weekly / Monthly).
- On the **same entry candle**, a Daily-tier and a Weekly/Monthly-tier setup are **one** trade: it keeps the higher tier's target.

Code: `_upgrade_targets`, `_find_level_reversals`, `_find_level_holds`, `_one_trade_per_candle`.

## 5. Entry, stop and plan (level trades)

- **Entry**:
  - CPLOC signals and the Daily hold: look at the two candles before the trigger. If the trigger closes **inside** their full high-low range, enter at its close. If it closes beyond it, wait for price to come back to the range of the second candle before the trigger and enter there.
  - Weekly / Monthly signal 2 (a 4H / Daily candle closing beyond a level): enter at that candle's close.
- **Stop**:
  - CPLOC signals and the Daily hold: the lowest low (buy) / highest high (sell) of the trigger candle and the two before it.
  - Weekly / Monthly signal 2: the rejection candle's own extreme.
- **Close-based stop for CPLOC signals** (1H reversal, 15M early, 15M continuation, Weekly/Monthly signal 1): only a candle **closing** beyond the stop ends the trade. A wick through the stop that closes back keeps it running. The other trades stop on a wick.
- **R:R below 1**: shown, but let go (not taken, not counted in the results).
- A retrace entry that never fills before price reaches the target is **missed** **[engine choice]**.
- A candle that touches both the stop and the target counts as **stopped** (the order inside a candle is unknown) **[engine choice]**.

Code: `_build`, `_plan`, `_resolve`.

## 6. Breaker block (BB)

- A **BB level** is the close of candle 1. Sell: candle 1 bullish, candle 2 a bearish CPLOC closing below candle 1's body, candle 3 retests the level (its high reaches it) and closes back below, leaving a wick. Buy = mirror image.
- A BB is **only valid once it has rejected price on its own timeframe**: candle 3 must be a completed candle of the BB's timeframe (a Daily BB needs a completed Daily candle that retested the level and closed back). A retest still happening inside the running candle does not count.
- A trade on a BB level **skips every close inside the formation** as a target and aims at the next level beyond.
- **Half off at candle 2's close**, the rest runs to the target.
- **FVG exit** (BB trades only, `FVG_EXIT=bb`): a 1H FVG in the trade's direction printed after the partial exit; when price comes back into it and a 1H candle closes beyond it against the trade, the trade is closed at that close.

Code: `_bb_closes`, `_plan`.

## 7. Busted levels, repeats

- **Busted level**: when a trade on a level is stopped out by a CPLOC the **other way**, that level takes no more trades of any kind or direction. Not for FVG trades or your own setups.
- **No repeats**: a setup in the same direction with the same final target while its earlier twin **on the same timeframe** is still running is not taken again. A different timeframe is a different trade (a 1H and a 4H FVG buy with the same target are 2 trades).

Code: `_apply_busted`, `_dup`, `_is_repeat`.

## 8. FVG trades (trend continuation)

**The gap** (as `indicator.pine`): three candles, the gap between the first and the third. Found on the 1H, 4H and Daily charts.
- The **unfilled part** shrinks to what price has not traded through (wicks count for this).
- A gap is **used up** only when a candle **closes** through it. A wick through the whole gap that closes back leaves it valid.

**An FVG only works on its own timeframe.** A 1H gap is traded with 1H candles and a 1H target, a 4H gap with 4H candles and a 4H target, a Daily gap with Daily candles and a Daily target.

**The gap's unfilled part, and what counts as a rejection (rules version 50):**
- A wick that runs through the **whole** gap and closes back outside gives the **whole original gap** back. Only a close through the far side uses the gap up.
- A candle of the trend's colour that **wicked outside the gap on its far side** (above a bearish gap, below a bullish one) and closed back inside it is itself a rejection, even though it did not close beyond the gap. Its close is the entry; its extreme is the stop.
- (Tested and left out: letting a candle close back inside the gap without cancelling the setup made the results worse.)

**The setup:**
1. **Trend**: an FVG is a continuation, not a reversal. At the tap, the latest break of structure on the gap's timeframe (swing-5, as the indicator) must be in the gap's direction. If it was the other way, there is no setup.
2. **Tap**: a candle touches the unfilled part of the gap.
3. **Trigger**: the first candle of the trend's colour (green for a buy, red for a sell) that closes beyond the gap: the tap candle itself or one of the next 3 **[engine choice: 3]**. It fails if a candle closes through the gap's far side, or back into the unfilled part, first.
4. **Every** tap that rejects is a setup (there is no one-per-gap limit).

**The plan:**
- Entry: the trigger's close.
- Stop: the extreme of the tap (and trigger) candle(s).
- Target: the highest high (buy) / lowest low (sell) on the gap's timeframe **from that break of structure up to the tap**, i.e. the peak the pullback into the gap came from. It is fixed, it never moves.
- If that peak is not beyond the entry, there is **no trade** (no fallback target).

Code: `fvg.py` (`zones`, `bounds_at`, `find_rejection`, `trend_start`), `reversals._find_fvg_continuations`, `reversals._fvg_target`.

## 9. Higher-timeframe FVG blocking

- A trade is **not taken** while a higher-timeframe FVG setup **the other way** is in play:
  - **alive**: a 4H / Daily FVG leg the other way that has not reached its target or been stopped, or
  - **forming**: the running 4H / Daily candle is printing that rejection (the gap was tapped by it or by one of the 3 candles before it, the running candle has the trend's colour so far and is beyond the gap).
- Either way it only counts when it would be a real FVG setup: the trend and target rules of section 8 must hold.
- This applies to FVG trades and to the Daily / Weekly / Monthly level trades.
- While a **Daily** FVG leg is alive in the trade's direction, Daily-tier trades aim at that leg's target.

Code: `fvg.legs`, `fvg.alive_leg`, `fvg.opposing_signal`, `reversals._build`.

## 10. AMDX phases

The market phase you set (Accumulation, Manipulation, Distribution, Expansion) for the day, the week and the month is weighed together: day 3, week 2, month 1. It is a warning on the trade, not a filter.

Code: `main.phase_context`, `store` (`market_phases`).

## 11. Engine choices that are tolerances, not rules

- Prices from different timeframes within **0.05** are treated as the same level **[engine choice]**.
- A custom level has no candle, so **one Daily ATR(14)** stands in for its candle range in the hold rule **[engine choice]**.
- A higher-timeframe opposite setup whose target crosses a pending trade's entry/stop is **flagged** on the trade, never removed **[engine choice]**.

## 12. The decision log

Every candidate the engine looks at is recorded with what it was, and **taken**, **let go** (R:R below 1) or **dropped**, with the reason. It is shown in the app's **Decisions** tab (pick a day). Clicking a row puts the chart on that candle.

A candidate is any signal that reached a rule: a CPLOC on a valid level, a candle crossing and closing beyond a level (holds), or an FVG tap that rejected. Candles that never formed a signal (for example a CPLOC on no valid level, or a hold candle of the wrong colour) are not listed.

Code: `reversals.decisions`, `/api/decisions`.

