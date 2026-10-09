"""Live loop: analyse after every H1 close, journal signals, paper-trade them,
notify via Telegram, and optionally route orders to MetaTrader 5.

Nothing is sent to a broker unless ``--execute`` is passed *and* the
MetaTrader5 package is installed and logged in (Windows terminal).
"""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from ..config import JOURNAL_DIR

JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS = JOURNAL_DIR / "signals.csv"
POSITIONS = JOURNAL_DIR / "paper_positions.json"


def notify_telegram(text: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text[:4000]}, timeout=15)
        return r.ok
    except requests.RequestException:
        return False


def journal_signal(live: dict, ai: dict | None) -> None:
    new = not SIGNALS.exists()
    p = live.get("plan", {})
    with SIGNALS.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time_utc", "symbol", "decision", "ai_decision", "price", "bias", "composite",
                        "entry", "stop", "tp1", "tp2", "risk_frac", "regime"])
        w.writerow([live["time_utc"], live["symbol"], live["decision"], (ai or {}).get("final_decision", ""),
                    live["price"], live["strategic"]["bias"], live["composite"], p.get("entry_ref"),
                    p.get("stop"), p.get("tp1_partial"), p.get("tp2_final"), p.get("risk_fraction"), live["regime"]])


def _load_positions() -> dict:
    try:
        return json.loads(POSITIONS.read_text())
    except Exception:
        return {}


def update_paper(live: dict, high: float, low: float, decision: str) -> list[str]:
    """Very small paper broker mirroring the backtest management rules."""
    pos = _load_positions()
    sym = live["symbol"]
    msgs = []
    p = pos.get(sym)
    if p:
        side = 1 if p["side"] == "LONG" else -1
        hit_stop = (low <= p["stop"]) if side > 0 else (high >= p["stop"])
        hit_tp1 = (high >= p["tp1"]) if side > 0 else (low <= p["tp1"])
        hit_tp2 = (high >= p["tp2"]) if side > 0 else (low <= p["tp2"])
        if hit_stop:
            msgs.append(f"{sym} paper {p['side']} closed at stop {p['stop']}")
            pos.pop(sym)
        elif hit_tp2:
            msgs.append(f"{sym} paper {p['side']} hit final target {p['tp2']}")
            pos.pop(sym)
        elif hit_tp1 and not p.get("partial"):
            p["partial"] = True
            p["stop"] = p["entry"]
            msgs.append(f"{sym} paper {p['side']}: TP1 reached, 50% closed, stop -> breakeven")
    if sym not in pos and decision in ("LONG", "SHORT"):
        pl = live["plan"]
        pos[sym] = {"side": decision, "entry": pl["entry_ref"], "stop": pl["stop"], "tp1": pl["tp1_partial"],
                    "tp2": pl["tp2_final"], "opened": live["time_utc"], "partial": False}
        msgs.append(f"{sym} paper {decision} opened @ {pl['entry_ref']} SL {pl['stop']} TP {pl['tp1_partial']}/{pl['tp2_final']}")
    POSITIONS.write_text(json.dumps(pos, indent=1))
    return msgs


def mt5_send(live: dict, execute: bool = False) -> str:
    if not execute:
        return "dry-run (pass --execute to route to MetaTrader 5)"
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError:
        return "MetaTrader5 package not installed (pip install MetaTrader5, Windows only)"
    if not mt5.initialize():
        return f"MT5 initialize failed: {mt5.last_error()}"
    pl = live["plan"]
    symbol = os.environ.get(f"MT5_SYMBOL_{live['symbol']}", live["symbol"])
    info = mt5.symbol_info_tick(symbol)
    if info is None:
        return f"symbol {symbol} not found in terminal"
    is_buy = pl["side"] == "LONG"
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": max(0.01, round(pl["lots_for_equity"], 2)),
           "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
           "price": info.ask if is_buy else info.bid, "sl": pl["stop"], "tp": pl["tp2_final"],
           "deviation": 20, "magic": 20261009, "comment": "aitrader", "type_filling": mt5.ORDER_FILLING_IOC}
    res = mt5.order_send(req)
    return f"MT5 order_send retcode={getattr(res, 'retcode', None)}"


def run_loop(symbols: list[str], use_ai: bool, execute: bool, once: bool = False, equity: float = 10_000.0):
    from ..pipeline import analyze_live
    from ..ai.claude_desk import run_desk
    from ..report.charts import render_png

    while True:
        for sym in symbols:
            try:
                res = analyze_live(sym, equity=equity, use_cache=False)
                live = res["live"]
                b = res["_bundle"]
                png = render_png(b.f, live, JOURNAL_DIR / f"{sym}_latest.png")
                ai = run_desk(live, png) if use_ai else None
                final = live["decision"]
                if ai and ai.get("mode") == "claude":
                    final = ai["decision"]["final_decision"]
                journal_signal(live, {"final_decision": final})
                last = b.f.iloc[-1]
                msgs = update_paper(live, float(last["high"]), float(last["low"]), final)
                if final in ("LONG", "SHORT"):
                    msgs.append(mt5_send(live, execute))
                head = f"[{sym}] {live['time_utc'][:16]} UTC price {live['price']} -> {final} " \
                       f"(bias {live['strategic']['bias']:+.2f}, desk {live['composite']:+.3f})"
                print(head)
                for m in msgs:
                    print("   ", m)
                if final in ("LONG", "SHORT") or msgs:
                    notify_telegram(head + "\n" + "\n".join(msgs))
            except Exception as e:  # keep the loop alive
                print(f"[{sym}] error: {e}")
        if once:
            return
        now = datetime.now(timezone.utc)
        nxt = (now + timedelta(hours=1)).replace(minute=2, second=0, microsecond=0)
        time.sleep(max(30, (nxt - now).total_seconds()))
