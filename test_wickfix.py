"""Compare the engine with and without the FVG wick fix (FVG_WICK_FIX). Runs the engine twice in separate processes; the live app is untouched.
    python test_wickfix.py
"""
import json
import os
import subprocess
import sys
import collections
import datetime as dt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = "/private/tmp/claude-501/wickfix"

CHILD = r'''
import sys, json
sys.path.insert(0, ROOT)
from backend.app import reversals, fvg
out = reversals._find_uncached()
rows = []
for s in out:
    p = s.get("plan")
    if not p:
        continue
    r = None
    if p.get("status") != "skipped":
        if p.get("outcome") == "target": r = p.get("result_r", p.get("rr_final", p["rr"]))
        elif p.get("outcome") == "stopped": r = p.get("result_r", -1.0)
    rows.append({"id": s["id"], "tier": s["tier"], "kind": s["kind"], "tf": s["trigger_tf"], "time": s["time"], "dir": s["direction"], "entry": p["entry"], "stop": p["stop"],
                 "target": s["initial_target"], "rr": p["rr"], "status": p.get("status"), "outcome": p.get("outcome"), "r": r})
json.dump({"fix": fvg.WICK_FIX, "rows": rows}, open(sys.argv[1], "w"))
'''


_OFF = {"FVG_FIX_WICK": "0", "FVG_FIX_WICKOUT": "0", "FVG_FIX_CLOSEIN": "0"}
VARIANTS = {"base": {**_OFF}, "wick": {**_OFF, "FVG_FIX_WICK": "1"}, "wickout": {**_OFF, "FVG_FIX_WICKOUT": "1"}, "wick+wickout": {**_OFF, "FVG_FIX_WICK": "1", "FVG_FIX_WICKOUT": "1"}}


def run(name: str) -> list[dict]:
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.json"
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("FVG_")}, **VARIANTS[name]}
    subprocess.run([sys.executable, "-W", "ignore", "-c", f"ROOT={ROOT!r}\n" + CHILD, path], env=env, check=True, cwd=ROOT)
    return json.load(open(path))["rows"]


def stats(rows: list[dict]) -> dict:
    won = sum(1 for r in rows if r["r"] is not None and r["r"] > 0)
    lost = sum(1 for r in rows if r["r"] is not None and r["r"] <= 0)
    return {"setups": len(rows), "taken": won + lost, "won": won, "lost": lost, "let_go": sum(1 for r in rows if r["status"] == "skipped"),
            "win_rate": round(won / (won + lost), 3) if won + lost else None, "net_r": round(sum(r["r"] or 0 for r in rows), 2)}


if __name__ == "__main__":
    R = {n: run(n) for n in VARIANTS}
    LABEL = {"base": "now", "wick": "(1) whole-gap wick", "wickout": "(3) wicked-out candle", "wick+wickout": "(1)+(3)"}
    picks = (("ALL setups", lambda r: True), ("FVG setups", lambda r: r["tier"] == "fvg"), ("non-FVG setups", lambda r: r["tier"] not in ("fvg", "manual")))
    for name, pick in picks:
        print(f"\n{name}")
        for n in VARIANTS:
            st = stats([r for r in R[n] if pick(r)])
            print(f"  {LABEL[n]:<24} setups {st['setups']:>5}  taken {st['taken']:>4}  won {st['won']:>4}  lost {st['lost']:>4}  win {st['win_rate']}  net {st['net_r']:+}R")
    print("\nFVG by timeframe (taken, won/lost, net R)")
    for tf in ("1h", "4h", "1d"):
        print(f"  {tf}: " + "   ".join(f"{LABEL[n]}: {(a:=stats([r for r in R[n] if r['tier'] == 'fvg' and r['kind'] == tf]))['taken']} ({a['won']}/{a['lost']}) {a['net_r']:+}R" for n in VARIANTS))
    base_ids = {r["id"] for r in R["base"]}
    for n in ("wick", "wickout", "wick+wickout"):
        ids = {r["id"] for r in R[n]}
        add, rem = [r for r in R[n] if r["id"] not in base_ids], [r for r in R["base"] if r["id"] not in ids]
        print(f"\n{LABEL[n]}: added {len(add)} setups {stats(add)['won']}W/{stats(add)['lost']}L {stats(add)['net_r']:+}R  |  removed {len(rem)} setups {stats(rem)['won']}W/{stats(rem)['lost']}L {stats(rem)['net_r']:+}R")
    print("\nThe Oct 26 setup (1H bearish gap 4005.59-4016.07, the red candle that wicked above it):")
    for n in VARIANTS:
        hit = [r for r in R[n] if r["tier"] == "fvg" and r["kind"] == "1h" and r["dir"] == "bear" and r["time"] >= 1792900000]
        print(f"  {LABEL[n]:<24}", [(dt.datetime.fromtimestamp(r['time'], dt.UTC).strftime('%d %H:%M'), r['entry'], r['stop'], r['target'], r['rr'], r['status'], r['outcome']) for r in hit] or "no setup")
    json.dump(R, open(f"{OUT}/compare.json", "w"))
