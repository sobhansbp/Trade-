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
NEWS_CACHE = JOURNAL_DIR / "news_view.json"
STATUS = JOURNAL_DIR / "STATUS.md"


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


def _cached_news(results: dict, every_hours: float) -> dict | None:
    """Score news at most every ``every_hours`` (Groq free tier ~200k tokens/day)."""
    from ..news.analyst import news_context

    try:
        cached = json.loads(NEWS_CACHE.read_text())
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(cached["scored_at"])).total_seconds() / 3600
        if age < every_hours and set(results) <= set(cached["view"]["symbols"]):
            return cached["view"]
    except Exception:
        pass
    view = news_context(results)
    if view:
        NEWS_CACHE.write_text(json.dumps({"scored_at": datetime.now(timezone.utc).isoformat(), "view": view},
                                         ensure_ascii=False, default=str))
    return view


def summary_text(st: dict, lives: dict, news: dict | None) -> str:
    """Short Persian account summary for Telegram."""
    init = st["initial"]
    t, s = st["tactical"]["equity"], st["swing"]["equity"]
    fa = {"LONG": "خرید", "SHORT": "فروش", "WAIT": "صبر"}
    lines = ["📊 گزارش حساب دمو aitrader",
             f"تاکتیکی: ${t:,.2f} ({(t / init - 1) * 100:+.2f}%)",
             f"سوئینگ: ${s:,.2f} ({(s / init - 1) * 100:+.2f}%)",
             f"جمع: ${t + s:,.2f} ({((t + s) / (2 * init) - 1) * 100:+.2f}%)"]
    for sym, p in st["tactical"]["positions"].items():
        lines.append(f"🔹 باز: {sym} {fa.get(p['side'], p['side'])} از {p['entry']} | SL {p['stop']} | TP {p['tp2']}")
    for sym, L in lives.items():
        nv = (L.get("news_view") or {}).get("score")
        lines.append(f"{sym} {L['price']}: {fa.get(L.get('final_decision', L['decision']), L['decision'])} | "
                     f"سوگیری روزانه {L['strategic']['bias']:+.2f} | میز ساعتی {L['composite']:+.3f}"
                     + (f" | اخبار {nv:+.2f}" if nv is not None else ""))
    if news and news.get("theme_fa"):
        lines.append(f"📰 {news['theme_fa']}")
    lines.append("جزئیات: github.com/sobhansbp/Trade-/blob/ccr-131f7ec0-kdgwmt/journal/STATUS.md")
    return "\n".join(lines)


def run_cycle(symbols: list[str], use_ai: bool, execute: bool, equity: float = 10_000.0,
              news_every_hours: float = 3.0, send_summary: bool = False,
              summary_hour_utc: int | None = 21) -> list[str]:
    """One full live cycle: analyse, news, AI desk on trade candidates, paper books, status."""
    from ..ai.claude_desk import run_desk
    from ..config import DEFAULT_CONFIG as cfg
    from ..news.analyst import apply_news_filter
    from ..pipeline import analyze_live
    from ..report.charts import render_png
    from . import paper

    st = paper.load_state(equity)
    results, out = {}, []
    for sym in symbols:
        try:
            results[sym] = analyze_live(sym, equity=st["tactical"]["equity"], use_cache=False)
        except Exception as e:
            out.append(f"[{sym}] analysis error: {e}")
    news = _cached_news(results, news_every_hours) if results else None
    lives = {}
    for sym, res in results.items():
        try:
            live, b = res["live"], res["_bundle"]
            if news:
                apply_news_filter(live, news["symbols"].get(sym))
            final = live["decision"]
            ai = None
            # the LLM desk is a veto on trades, so only spend tokens when there is a trade to veto
            if use_ai and final in ("LONG", "SHORT") and sym not in st["tactical"]["positions"]:
                png = render_png(b.f, live, JOURNAL_DIR / f"{sym}_latest.png")
                ai = run_desk(live, png)
                if ai.get("mode") == "llm":
                    final = ai["decision"]["final_decision"]
                    live["ai_summary_fa"] = ai["decision"].get("summary_fa", "")
            live["final_decision"] = final
            lives[sym] = live
            journal_signal(live, {"final_decision": final})
            msgs = paper.tactical_step(st, live, b.df, final, cfg)
            sb = live["strategic"]
            msgs += paper.swing_step(st, sym, float(b.df["close"].iloc[-1]), sb["bias"],
                                     sb["vol_target_leverage"], len(symbols))
            if final in ("LONG", "SHORT") and any("opened" in m for m in msgs):
                msgs.append(mt5_send(live, execute))
            nv = live.get("news_view")
            head = (f"[{sym}] {live['time_utc'][:16]} UTC price {live['price']} -> {final} "
                    f"(bias {sb['bias']:+.2f}, desk {live['composite']:+.3f}"
                    + (f", news {nv['score']:+.2f}" if nv else "") + ")")
            out.append(head)
            out += ["    " + m for m in msgs]
            if msgs:
                notify_telegram(head + "\n" + "\n".join(msgs) +
                                (f"\n{live.get('ai_summary_fa', '')}" if live.get("ai_summary_fa") else ""))
        except Exception as e:  # keep the loop alive
            out.append(f"[{sym}] error: {e}")
    paper.save_state(st)
    paper.log_equity(st)
    try:   # pre-registered research hypotheses, logged only (never traded)
        from . import forward
        out += forward.update()
        fwd = forward.summary()
    except Exception as e:
        out.append(f"forward test error: {e}")
        fwd = None
    STATUS.write_text(paper.status_markdown(st, lives, news, fwd), encoding="utf-8")
    if send_summary or (summary_hour_utc is not None and datetime.now(timezone.utc).hour == summary_hour_utc):
        ok = notify_telegram(summary_text(st, lives, news))
        out.append(f"telegram summary sent: {ok}")
    t, s = st["tactical"]["equity"], st["swing"]["equity"]
    out.append(f"paper equity: tactical ${t:,.2f} | swing ${s:,.2f} | total ${t + s:,.2f}")
    return out


def run_loop(symbols: list[str], use_ai: bool, execute: bool, once: bool = False, equity: float = 10_000.0,
             news_every_hours: float = 3.0, send_summary: bool = False):
    while True:
        for line in run_cycle(symbols, use_ai, execute, equity, news_every_hours, send_summary):
            print(line, flush=True)
        if once:
            return
        now = datetime.now(timezone.utc)
        nxt = (now + timedelta(hours=1)).replace(minute=2, second=0, microsecond=0)
        time.sleep(max(30, (nxt - now).total_seconds()))
