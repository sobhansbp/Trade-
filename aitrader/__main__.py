"""Command line interface.

  python -m aitrader analyze EURUSD XAUUSD [--ai] [--equity 10000]
  python -m aitrader backtest EURUSD XAUUSD
  python -m aitrader live EURUSD XAUUSD [--ai] [--execute] [--once]
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import INSTRUMENTS, REPORT_DIR


def _csv_map(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.upper()] = v
    return out


def cmd_analyze(args):
    from .ai.claude_desk import run_desk
    from .pipeline import analyze_live
    from .report.charts import plotly_chart, plotly_equity, render_png
    from .report.html import build_dashboard, to_json
    from .research_notes import FINDINGS

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csvs = _csv_map(args.csv)
    results = {sym: analyze_live(sym, csv=csvs.get(sym), equity=args.equity) for sym in args.symbols}
    from .news.analyst import news_context
    news = news_context(results) if not args.no_news else None
    items = []
    for sym, res in results.items():
        live, b = res["live"], res["_bundle"]
        if news:
            from .news.analyst import apply_news_filter
            apply_news_filter(live, news["symbols"].get(sym))
            live["news_headlines"] = [{"title": h["title"], "source": h["source"], "time": h["time"],
                                       "impact": h["impact"].get(sym, 0), "confidence": h["confidence"]}
                                      for h in news["items"] if h["impact"].get(sym, 0)][:12]
            live["news_theme_fa"] = news.get("theme_fa", "")
        png = render_png(b.f, live, REPORT_DIR / f"{sym}_h1.png")
        ai = run_desk(live, png) if args.ai else None
        if ai is None:
            from .ai.claude_desk import offline_brief
            ai = {"mode": "offline", **offline_brief(live)}
        runs = res["_runs"]
        eq = {"production (OOS)": runs["production_oos"]["equity"], "random entries": runs["random_entries"]["equity"],
              "desk-only v1": runs["desk_only_v1"]["equity"]}
        items.append({"live": live, "ai": ai, "chart": plotly_chart(b.f, live),
                      "equity": plotly_equity(eq, f"{sym} equity, $10k start, out-of-sample")})
        (REPORT_DIR / f"{sym}_analysis.json").write_text(to_json({"live": live, "ai": ai}))
        print(f"\n=== {sym} @ {live['price']} ({live['time_utc'][:16]} UTC)")
        print(f"decision: {live['decision']} | daily bias {live['strategic']['bias']:+.2f} | "
              f"desk {live['composite']:+.3f} | regime {live['regime']}"
              + (f" | news {live['news_view']['score']:+.2f}" if live.get("news_view") else ""))
        for w in live["why_not"]:
            print("  why not:", w)
        p = live["plan"]
        print(f"  plan ({p['side']}): entry {p['entry_ref']} stop {p['stop']} tp1 {p['tp1_partial']} tp2 {p['tp2_final']}")
        txt = ai.get("summary_fa") or ai.get("decision", {}).get("summary_fa", "")
        print("  " + txt.replace("\n", "\n  "))
    notes = json.loads(open(args.notes, encoding="utf-8").read()) if args.notes else None
    from .portfolio import account_view
    acc = account_view(results)
    acc_json = {k: v for k, v in acc.items() if not k.startswith("_")}
    (REPORT_DIR / "account_view.json").write_text(to_json(acc_json))
    if "combined" in acc:
        print(f"\naccount: combined sharpe {acc['combined']['sharpe']} | H1 {acc['h1']['sharpe']} | "
              f"swing {acc['swing']['sharpe']} | corr {acc['corr_h1_swing']}")
    from .report.lab import lab_section
    from .research_notes import LAB_CHART_NOTES, LAB_HYPOTHESES, LAB_SOURCES, LAB_SURVEY
    lab = lab_section(LAB_SURVEY, LAB_HYPOTHESES, LAB_CHART_NOTES, LAB_SOURCES)
    html = build_dashboard(items, FINDINGS, standalone=True, ai_commentary=notes, account=acc, lab=lab)
    out = REPORT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    (REPORT_DIR / "dashboard_fragment.html").write_text(build_dashboard(items, FINDINGS, standalone=False,
                                                                        ai_commentary=notes, account=acc, lab=lab),
                                                       encoding="utf-8")
    print(f"\ndashboard -> {out}")


def cmd_backtest(args):
    from .backtest.engine import combine_portfolio
    from .pipeline import research

    csvs = _csv_map(args.csv)
    results, summary = {}, {}
    for sym in args.symbols:
        r = research(sym, csv=csvs.get(sym))
        results[sym] = r["_runs"]["production_oos"]
        summary[sym] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(f"\n=== {sym}  {r['start'][:10]} -> {r['end'][:10]}  holdout from {r['holdout_start'][:10]}")
        for k, s in r["runs"].items():
            print(f"  {k:22s} trades={s.get('trades', 0):4d} win={s.get('win_rate', 0):.2f} avgR={s.get('avg_r', 0):+.3f} "
                  f"PF={s.get('profit_factor', 0)} sharpe={s.get('sharpe', 0)} dd={s.get('max_drawdown', 0)} "
                  f"ret={s.get('total_return', 0):+.4f}")
        print("  swing (daily, 20y):", json.dumps(r["swing"]))
    if len(results) > 1:
        port = combine_portfolio(results, INSTRUMENTS)
        summary["portfolio"] = port["stats"]
        print("\n=== portfolio (correlation-aware):", json.dumps(port["stats"], default=str))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "backtest_summary.json").write_text(json.dumps(summary, indent=1, default=str))


def cmd_live(args):
    from .live.runner import run_loop

    run_loop(args.symbols, use_ai=args.ai, execute=args.execute, once=args.once, equity=args.equity,
             news_every_hours=args.news_every, send_summary=args.summary)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aitrader", description="AI-assisted EURUSD / XAUUSD trading desk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("analyze", cmd_analyze), ("backtest", cmd_backtest), ("live", cmd_live)):
        p = sub.add_parser(name)
        p.add_argument("symbols", nargs="*", default=["EURUSD", "XAUUSD"])
        p.add_argument("--csv", action="append", help="SYMBOL=path/to/broker_export.csv (H1 or lower)")
        p.add_argument("--equity", type=float, default=10_000.0)
        p.add_argument("--risk", type=float, default=None,
                       help="risk per trade as a fraction of equity (default 0.005; >0.01 is not recommended)")
        p.add_argument("--ai", action="store_true", help="run the LLM bull/bear/head-trader desk (Groq or Claude)")
        if name == "analyze":
            p.add_argument("--notes", help="JSON file {SYMBOL: analyst commentary} added to the dashboard")
            p.add_argument("--no-news", action="store_true", help="skip the LLM news analyst")
        if name == "live":
            p.add_argument("--execute", action="store_true", help="route orders to MetaTrader 5")
            p.add_argument("--once", action="store_true", help="run one cycle and exit (for cron / CI)")
            p.add_argument("--summary", action="store_true", help="also send a Telegram account summary")
            p.add_argument("--news-every", type=float, default=3.0,
                           help="hours between LLM news scorings (Groq free tier ~200k tokens/day)")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.symbols = [s.upper() for s in args.symbols]
    if args.risk is not None:
        from . import config
        if not 0 < args.risk <= 0.05:
            ap.error("--risk must be between 0 and 0.05")
        if args.risk > 0.01:
            print(f"WARNING: risk {args.risk:.1%} per trade - see the Monte Carlo table: drawdowns grow fast")
        config.DEFAULT_CONFIG.risk_per_trade = args.risk
        config.DEFAULT_CONFIG.max_risk_per_trade = max(config.DEFAULT_CONFIG.max_risk_per_trade, 2 * args.risk)
    bad = [s for s in args.symbols if s not in INSTRUMENTS]
    if bad:
        ap.error(f"unknown symbols {bad}; available: {list(INSTRUMENTS)}")
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
