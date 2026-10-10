"""Persistent paper-trading accounts (demo) that mirror the backtest rules.

Two books, each starting with ``initial`` dollars:
  tactical - H1 trades from the production signal (structural stop, 50% at +1R
             then breakeven, final target at rr_target, time stop), position size
             from risk fraction, drawdown throttle and USD-correlation haircut
  swing    - daily book holding bias x vol-target leverage per instrument
             (split equally across instruments), marked to market each run

State lives in journal/ as plain JSON/CSV so it can be committed by CI and read
on GitHub. Every run replays all H1 bars since the last check, so missed runs
do not skip stops or targets.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import INSTRUMENTS, JOURNAL_DIR, StrategyConfig
from ..risk.manager import position_size, risk_fraction

STATE = JOURNAL_DIR / "paper_state.json"
TRADES = JOURNAL_DIR / "paper_trades.csv"
EQUITY = JOURNAL_DIR / "paper_equity.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state(initial: float = 10_000.0) -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"created": _now(), "initial": initial,
                "tactical": {"equity": initial, "peak": initial, "positions": {}, "last_signal_bar": {}},
                "swing": {"equity": initial, "peak": initial, "positions": {}}}


def save_state(st: dict) -> None:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, default=str))


def _log_trade(row: dict) -> None:
    new = not TRADES.exists()
    with TRADES.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _manage(sym: str, p: dict, bars: pd.DataFrame, cfg: StrategyConfig, book: dict) -> tuple[bool, list[str]]:
    """Replay bars after the last processed time. Returns (closed, messages)."""
    inst = INSTRUMENTS[sym]
    side = 1 if p["side"] == "LONG" else -1
    msgs = []
    last = pd.Timestamp(p["last_bar"])
    new = bars[bars.index > last]
    for t, b in new.iterrows():
        p["bars_held"] += 1
        exit_px, reason = None, None
        if (side > 0 and b["low"] <= p["stop"]) or (side < 0 and b["high"] >= p["stop"]):
            exit_px = p["stop"]
            if (side > 0 and b["open"] < p["stop"]) or (side < 0 and b["open"] > p["stop"]):
                exit_px = b["open"]
            exit_px -= side * inst.slippage
            reason = "breakeven" if p["partial"] else "stop"
        elif (side > 0 and b["high"] >= p["tp2"]) or (side < 0 and b["low"] <= p["tp2"]):
            exit_px, reason = p["tp2"], "target"
        else:
            if not p["partial"] and ((side > 0 and b["high"] >= p["tp1"]) or (side < 0 and b["low"] <= p["tp1"])):
                gain = cfg.partial_fraction * p["units"] * side * (p["tp1"] - p["entry"])
                p["realized"] += gain
                p["units_open"] = p["units"] * (1 - cfg.partial_fraction)
                p["partial"] = True
                p["stop"] = p["entry"] + side * (inst.spread / 2 + inst.slippage)
                msgs.append(f"{sym} {p['side']}: TP1 {p['tp1']} hit, 50% closed (+${gain:,.2f}), stop to breakeven")
            if p["bars_held"] >= cfg.time_stop_bars:
                exit_px, reason = b["close"] - side * inst.spread / 2, "time"
        p["last_bar"] = str(t)
        if exit_px is not None:
            pnl = p["realized"] + p["units_open"] * side * (exit_px - p["entry"])
            r_mult = pnl / p["risk_usd"] if p["risk_usd"] else 0.0
            book["equity"] += pnl
            book["peak"] = max(book["peak"], book["equity"])
            _log_trade({"symbol": sym, "side": p["side"], "opened": p["opened"], "closed": str(t),
                        "entry": p["entry"], "exit": round(exit_px, inst.digits), "stop0": p["stop0"],
                        "tp2": p["tp2"], "units": round(p["units"], 2), "risk_usd": round(p["risk_usd"], 2),
                        "pnl_usd": round(pnl, 2), "r_multiple": round(r_mult, 2), "reason": reason,
                        "equity_after": round(book["equity"], 2)})
            msgs.append(f"{sym} {p['side']} closed ({reason}) @ {exit_px:.{inst.digits}f}: "
                        f"{'+' if pnl >= 0 else ''}${pnl:,.2f} ({r_mult:+.2f}R), equity ${book['equity']:,.2f}")
            return True, msgs
    return False, msgs


def tactical_step(st: dict, live: dict, bars: pd.DataFrame, decision: str, cfg: StrategyConfig) -> list[str]:
    book = st["tactical"]
    sym = live["symbol"]
    inst = INSTRUMENTS[sym]
    msgs = []
    if sym in book["positions"]:
        closed, m = _manage(sym, book["positions"][sym], bars, cfg, book)
        msgs += m
        if closed:
            book["positions"].pop(sym)
            book["last_signal_bar"][sym] = live["time_utc"]   # cooldown: no re-entry on the exit bar
    if sym not in book["positions"] and decision in ("LONG", "SHORT") \
            and book["last_signal_bar"].get(sym) != live["time_utc"]:
        side = 1 if decision == "LONG" else -1
        dd = 1 - book["equity"] / book["peak"]
        rf = risk_fraction(cfg, None, dd) * (live["plan"]["risk_fraction"] / max(cfg.risk_per_trade, 1e-9)
                                             if live["plan"].get("news_note") else 1.0)
        usd_dir = side * inst.usd_side
        if any((1 if q["side"] == "LONG" else -1) * INSTRUMENTS[s].usd_side == usd_dir
               for s, q in book["positions"].items()):
            rf *= 0.5
        if rf > 0:
            pl = live["plan"]
            stop_d = pl["stop_distance"]
            entry = float(bars["close"].iloc[-1]) + side * (inst.spread / 2 + inst.slippage)
            risk_usd = book["equity"] * rf
            units = risk_usd / stop_d
            book["positions"][sym] = {
                "side": decision, "opened": live["time_utc"], "entry": round(entry, inst.digits),
                "stop": round(entry - side * stop_d, inst.digits), "stop0": round(entry - side * stop_d, inst.digits),
                "tp1": round(entry + side * cfg.partial_at_r * stop_d, inst.digits),
                "tp2": round(entry + side * cfg.rr_target * stop_d, inst.digits),
                "units": units, "units_open": units, "lots": round(position_size(book["equity"], rf, stop_d, inst), 2),
                "risk_usd": risk_usd, "risk_frac": rf, "partial": False, "realized": 0.0,
                "bars_held": 0, "last_bar": str(bars.index[-1])}
            book["last_signal_bar"][sym] = live["time_utc"]
            q = book["positions"][sym]
            msgs.append(f"{sym} paper {decision} opened @ {q['entry']} SL {q['stop']} TP1 {q['tp1']} TP2 {q['tp2']} "
                        f"({q['lots']} lots, risk ${risk_usd:,.2f} = {rf:.2%})")
    return msgs


def swing_step(st: dict, sym: str, price: float, bias: float, leverage: float, n_symbols: int) -> list[str]:
    """Rebalance the daily swing book to bias x leverage / n_symbols of equity."""
    book = st["swing"]
    inst = INSTRUMENTS[sym]
    pos = book["positions"].get(sym)
    msgs = []
    if pos:  # mark to market since last price
        pnl = pos["notional"] * (price / pos["price"] - 1)
        book["equity"] += pnl
        book["peak"] = max(book["peak"], book["equity"])
    target_frac = float(np.clip(bias * leverage / max(1, n_symbols), -3, 3)) if np.isfinite(leverage) else 0.0
    target_notional = target_frac * book["equity"]
    old = pos["notional"] if pos else 0.0
    if abs(target_notional - old) > 0.1 * book["equity"] / max(1, n_symbols) or pos is None:
        cost = abs(target_notional - old) * (inst.spread / 2) / price
        book["equity"] -= cost
        if pos is None or np.sign(target_notional) != np.sign(old):
            msgs.append(f"{sym} swing book -> {target_frac:+.0%} of equity (bias {bias:+.2f})")
        book["positions"][sym] = {"notional": target_notional, "price": price, "frac": target_frac,
                                  "updated": _now()}
    else:
        book["positions"][sym]["price"] = price
    return msgs


def log_equity(st: dict) -> None:
    new = not EQUITY.exists()
    with EQUITY.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "tactical", "swing", "combined", "open_tactical"])
        t, s = st["tactical"]["equity"], st["swing"]["equity"]
        w.writerow([_now(), round(t, 2), round(s, 2), round(t + s, 2), ",".join(st["tactical"]["positions"])])


def status_markdown(st: dict, lives: dict, news: dict | None, forward: list[dict] | None = None) -> str:
    init = st["initial"]
    t, s = st["tactical"], st["swing"]

    def pct(x):
        return f"{x * 100:+.2f}%"
    lines = ["# وضعیت حساب دمو (Paper Trading)", "",
             f"آخرین به‌روزرسانی: `{_now()}` · شروع: `{st['created'][:10]}` · سرمایه اولیه هر دفتر: ${init:,.0f}", "",
             "| دفتر | موجودی | بازده | افت از سقف |", "|---|---|---|---|",
             f"| تاکتیکی ساعتی (آزمایشی) | ${t['equity']:,.2f} | {pct(t['equity'] / init - 1)} | {pct(t['equity'] / t['peak'] - 1)} |",
             f"| سوئینگ روزانه | ${s['equity']:,.2f} | {pct(s['equity'] / init - 1)} | {pct(s['equity'] / s['peak'] - 1)} |",
             f"| **جمع** | **${t['equity'] + s['equity']:,.2f}** | **{pct((t['equity'] + s['equity']) / (2 * init) - 1)}** | |",
             "", "## پوزیشن‌های باز تاکتیکی", ""]
    if t["positions"]:
        lines += ["| نماد | جهت | ورود | حد ضرر | TP1 | TP2 | لات | ریسک |", "|---|---|---|---|---|---|---|---|"]
        for sym, p in t["positions"].items():
            lines.append(f"| {sym} | {p['side']} | {p['entry']} | {p['stop']} | {p['tp1']} | {p['tp2']} | "
                         f"{p['lots']} | ${p['risk_usd']:,.2f} |")
    else:
        lines.append("پوزیشن بازی نیست.")
    lines += ["", "## دفتر سوئینگ", "", "| نماد | موقعیت (٪ سرمایه) |", "|---|---|"]
    for sym, p in s["positions"].items():
        lines.append(f"| {sym} | {p['frac']:+.0%} |")
    lines += ["", "## آخرین تحلیل", "", "| نماد | قیمت | تصمیم | سوگیری روزانه | میز ساعتی | اخبار | دلیل |",
              "|---|---|---|---|---|---|---|"]
    for sym, L in lives.items():
        nv = L.get("news_view") or {}
        why = "؛ ".join(L.get("why_not", []))[:120].replace("|", "\\|")
        lines.append(f"| {sym} | {L['price']} | {L['decision']} | {L['strategic']['bias']:+.2f} | "
                     f"{L['composite']:+.3f} | {nv.get('score', 0):+.2f} | {why} |")
    if news and news.get("theme_fa"):
        lines += ["", f"**جمع‌بندی اخبار ({news.get('model', '')}):** {news['theme_fa']}"]
    if TRADES.exists():
        df = pd.read_csv(TRADES)
        if len(df):
            wins = (df["pnl_usd"] > 0).mean()
            lines += ["", f"## معاملات بسته‌شده: {len(df)} · برد {wins:.0%} · میانگین {df['r_multiple'].mean():+.2f}R · "
                          f"جمع ${df['pnl_usd'].sum():,.2f}", "",
                      "| بسته شد | نماد | جهت | ورود | خروج | نتیجه | سود/زیان |", "|---|---|---|---|---|---|---|"]
            for _, r in df.tail(15).iloc[::-1].iterrows():
                lines.append(f"| {str(r['closed'])[:16]} | {r['symbol']} | {r['side']} | {r['entry']} | {r['exit']} | "
                             f"{r['reason']} {r['r_multiple']:+.2f}R | ${r['pnl_usd']:,.2f} |")
    if forward:
        lines += ["", "## آزمون رو به جلو فرضیه‌ها (فقط ثبت، بدون معامله)", "",
                  "| فرضیه | نماد | معامله | میانگین خالص (bp) | جمع (bp) | برد | t |", "|---|---|---|---|---|---|---|"]
        for h in forward:
            t = f"{h['t']:+.2f}" if h["t"] == h["t"] else "–"
            lines.append(f"| {h['hypothesis']} | {h['symbol']} | {h['n']} | {h['mean_net_bp']:+.2f} | "
                         f"{h['sum_net_bp']:+.1f} | {h['hit']:.0%} | {t} |")
        lines.append("ثبت از ۲۰۲۶-۱۰-۱۲؛ قاعده‌ها پیش از دیدن این داده‌ها ثابت شده‌اند (`aitrader/live/forward.py`).")
    from ..config import DESK_URL
    lines += ["", f"داشبورد زنده (Cloudflare): {DESK_URL} · ربات تلگرام: /status، /gold، /eurusd و پرسش آزاد",
              "", "> حساب کاغذی است؛ هیچ سفارشی به بروکر ارسال نمی‌شود. نتایج گذشته تضمین آینده نیست."]
    return "\n".join(lines) + "\n"
