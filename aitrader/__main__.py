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
    items = []
    for sym in args.symbols:
        res = analyze_live(sym, csv=csvs.get(sym), equity=args.equity)
        live, b = res["live"], res["_bundle"]
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
              f"desk {live['composite']:+.3f} | regime {live['regime']}")
        for w in live["why_not"]:
            print("  why not:", w)
        p = live["plan"]
        print(f"  plan ({p['side']}): entry {p['entry_ref']} stop {p['stop']} tp1 {p['tp1_partial']} tp2 {p['tp2_final']}")
        print("  " + (ai.get("summary_fa") or ai.get("decision", {}).get("summary_fa", "")).replace("\n", "\n  "))
    notes = json.loads(open(args.notes, encoding="utf-8").read()) if args.notes else None
    html = build_dashboard(items, FINDINGS, standalone=True, ai_commentary=notes)
    out = REPORT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    (REPORT_DIR / "dashboard_fragment.html").write_text(build_dashboard(items, FINDINGS, standalone=False,
                                                                        ai_commentary=notes),
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

    run_loop(args.symbols, use_ai=args.ai, execute=args.execute, once=args.once, equity=args.equity)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aitrader", description="AI-assisted EURUSD / XAUUSD trading desk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("analyze", cmd_analyze), ("backtest", cmd_backtest), ("live", cmd_live)):
        p = sub.add_parser(name)
        p.add_argument("symbols", nargs="*", default=["EURUSD", "XAUUSD"])
        p.add_argument("--csv", action="append", help="SYMBOL=path/to/broker_export.csv (H1 or lower)")
        p.add_argument("--equity", type=float, default=10_000.0)
        p.add_argument("--ai", action="store_true", help="run the Claude bull/bear/head-trader desk")
        if name == "analyze":
            p.add_argument("--notes", help="JSON file {SYMBOL: analyst commentary} added to the dashboard")
        if name == "live":
            p.add_argument("--execute", action="store_true", help="route orders to MetaTrader 5")
            p.add_argument("--once", action="store_true")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.symbols = [s.upper() for s in args.symbols]
    bad = [s for s in args.symbols if s not in INSTRUMENTS]
    if bad:
        ap.error(f"unknown symbols {bad}; available: {list(INSTRUMENTS)}")
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
