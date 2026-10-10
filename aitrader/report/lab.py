"""Research-lab section of the dashboard (round 3).

Everything is read from research/results/ (committed) and reports/*_mtf.png, so the
section can be rebuilt without the 1-minute data cache.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .html import _e

RES = Path("research/results")
IMG = Path("reports")

NAME_FA = {
    "W1 pre-Tokyo fix: USD up (18:00 ET -> 10:00 Tokyo)": "W1 پیش از فیکس توکیو: دلار بالا",
    "W2 post-Tokyo fix: USD down (10:00 Tokyo -> 08:00 London)": "W2 پس از فیکس توکیو: دلار پایین",
    "W3 pre-ECB fix: USD up (08:00 London -> 14:15 Frankfurt)": "W3 پیش از فیکس ECB: دلار بالا",
    "W4 ECB -> 1h before London fix: USD down": "W4 از فیکس ECB تا یک ساعت پیش از فیکس لندن: دلار پایین",
    "W5 into London 4pm fix: USD up (15:00 -> 16:00 London)": "W5 منتهی به فیکس ۴ عصر لندن: دلار بالا",
    "W6 post London fix: USD down (16:00 London -> 16:50 ET)": "W6 پس از فیکس لندن: دلار پایین",
    "Breedon-Ranaldo: European morning (08:00-12:00 London)": "بریدن-رانالدو: صبح اروپا",
    "Breedon-Ranaldo: US afternoon (12:00 ET -> 16:00 ET)": "بریدن-رانالدو: بعدازظهر آمریکا",
    "W5 into London fix, month-end only": "W5 فقط آخر ماه",
    "W6 post London fix, month-end only": "W6 فقط آخر ماه",
    "Last trading day of month long": "خرید در آخرین روز معاملاتی ماه",
    "Friday long": "خرید روز جمعه",
    "Tuesday short": "فروش روز سه‌شنبه",
    "BASELINE: long every day (close to close)": "مبنا: خرید هر روز",
    "London breakout of Asian range (stop opposite, exit 16:00 London)": "شکست لندن از رنج آسیا",
    "London breakout, target 1x range, width filter": "شکست لندن با تارگت یک‌برابر رنج و فیلتر عرض",
    "NY 30-min ORB 08:20-08:50 ET (exit 13:30 ET)": "شکست رنج ۳۰ دقیقه اول نیویورک",
    "ICT Judas swing (Asian range sweep fade, London KZ)": "ICT جوداس سوئینگ (خلاف شکار نقدینگی رنج آسیا)",
    "Williams volatility breakout k=0.5": "شکست نوسان لری ویلیامز (k=0.5)",
    "5-min ORB (Zarattini-Aziz) at 08:20 ET": "ORB پنج‌دقیقه‌ای زاراتینی-عزیز",
    "Intraday momentum (first 30m -> last hour)": "مومنتوم درون‌روزی (۳۰ دقیقه اول به ساعت آخر)",
    "Weekend gap fade (>0.1%) to Mon 12:00 London": "خلاف گپ آخر هفته (ورود در لحظه بازگشایی)",
    "Pre-FOMC drift: long 24h before statement": "رانش پیش از FOMC: خرید ۲۴ ساعت قبل",
    "Pre-FOMC USD-up: short 48h before statement": "دلار بالا پیش از FOMC: فروش ۴۸ ساعت قبل",
    "Post-FOMC 2-day drift long": "رانش دوروزه پس از FOMC (خرید)",
    "Post-FOMC reversal (fade first 30m, hold 24h)": "برگشت پس از FOMC (خلاف ۳۰ دقیقه اول)",
    "NFP continuation (first 15m direction, hold 3h)": "ادامه حرکت NFP",
    "NFP reversal (fade first 15m, hold to NY close)": "برگشت NFP",
    "Gold Asian hours long (19:00 ET -> 03:00 ET)": "خرید طلا در ساعات آسیا",
    "Gold overnight long (13:30 ET -> 08:20 ET)": "خرید شبانه طلا (۱۳:۳۰ تا ۰۸:۲۰ نیویورک)",
    "Gold COMEX day short (08:20 -> 13:30 ET)": "فروش طلا در جلسه روزانه کومکس",
    "Gold into PM fix short (14:00 -> 15:00 London)": "فروش طلا منتهی به فیکس عصر لندن",
    "Gold after PM fix long (15:00 -> 17:00 London)": "خرید طلا پس از فیکس عصر لندن",
    "Gold long in September & November": "خرید طلا در سپتامبر و نوامبر",
}
SHORT = {
    "W3 pre-ECB fix: USD up (08:00 London -> 14:15 Frankfurt)": "W3 pre-ECB fix (KMW)",
    "W1 pre-Tokyo fix: USD up (18:00 ET -> 10:00 Tokyo)": "W1 pre-Tokyo fix (KMW)",
    "London breakout of Asian range (stop opposite, exit 16:00 London)": "London breakout",
    "5-min ORB (Zarattini-Aziz) at 08:20 ET": "5-min ORB (Zarattini-Aziz)",
    "ICT Judas swing (Asian range sweep fade, London KZ)": "ICT Judas swing",
    "BASELINE: long every day (close to close)": "baseline: long every day",
    "Gold overnight long (13:30 ET -> 08:20 ET)": "overnight long",
    "Gold COMEX day short (08:20 -> 13:30 ET)": "COMEX day-session short",
    "Williams volatility breakout k=0.5": "Williams breakout k=0.5",
    "Pre-FOMC drift: long 24h before statement": "pre-FOMC long",
}
CHRONOS_FA = {"amazon/chronos-bolt-small": "Chronos-Bolt (کوچک)", "amazon/chronos-2": "Chronos-2"}


def _load(name: str):
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _plot(fig, height=320) -> str:
    fig.update_layout(height=height, margin=dict(l=56, r=16, t=40, b=10), template="plotly_white",
                      legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    div = fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
    return f'<div class="scroll"><div style="min-width:560px;direction:ltr">{div}</div></div>'


def _equity_fig(compact: dict, sym: str) -> str:
    import plotly.graph_objects as go
    fig = go.Figure()
    for key, c in compact["curves"].items():
        s, name = key.split("|", 1)
        if s != sym:
            continue
        fig.add_trace(go.Scatter(x=c["t"], y=c["net"], mode="lines", name=SHORT.get(name, name[:40])))
    fig.add_vline(x="2019-01", line_dash="dash", line_color="#888")
    fig.add_hline(y=0, line_color="#999", line_width=1)
    fig.update_layout(title=dict(text=f"{sym}: cumulative % after retail costs (dashed = start of out-of-sample)",
                                 font=dict(size=12)))
    return _plot(fig, 360)


def _hours_fig(compact: dict, sym: str) -> str:
    import plotly.graph_objects as go
    h = compact["hours"][sym]
    fig = go.Figure()
    for k, name, col in (("is", "2010-2018 (in-sample)", "#8a96a3"), ("oos", "2019-2026 (out-of-sample)", "#c0392b")):
        fig.add_trace(go.Bar(x=list(range(24)), y=h[k]["mean_bp"], name=name, marker_color=col,
                             customdata=h[k]["t"], hovertemplate="hour %{x} UTC: %{y:.2f} bp (t %{customdata})"))
    fig.update_layout(barmode="group", title=dict(text=f"{sym}: mean return by UTC hour (bp)", font=dict(size=12)),
                      xaxis=dict(dtick=1))
    return _plot(fig, 300)


def _scatter_fig(rows: list[dict]) -> str:
    import plotly.graph_objects as go
    fig = go.Figure()
    for sym, col in (("EURUSD", "#2980b9"), ("XAUUSD", "#b07d2b")):
        r = [x for x in rows if x["symbol"] == sym]
        fig.add_trace(go.Scatter(x=[x["is_net_bp"] for x in r], y=[x["oos_net_bp"] for x in r], mode="markers",
                                 name=sym, marker=dict(color=col, size=8),
                                 text=[x["strategy"] for x in r], hovertemplate="%{text}<br>IS %{x:.2f} / OOS %{y:.2f} bp"))
    fig.add_shape(type="line", x0=-25, y0=-25, x1=25, y1=25, line=dict(dash="dash", color="#999", width=1))
    fig.update_layout(title=dict(text="net bp per trade: in-sample (x) vs out-of-sample (y)", font=dict(size=12)),
                      xaxis=dict(range=[-25, 25], zeroline=True), yaxis=dict(range=[-25, 25], zeroline=True))
    return _plot(fig, 380)


def _rank_corr(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _verdict(r: dict) -> tuple[str, str]:
    if r["strategy"].startswith("BASELINE"):
        return "مرجع", ""
    if r["strategy"].startswith("Weekend gap fade"):
        return "ساخته داده (ورود غیرواقعی)", "bad"
    if r["passes_retail"]:
        return "قبول", "good"
    if r["oos_net_bp"] > 0 and r["same_sign"]:
        return "مثبت ولی غیرمعنادار", ""
    return "رد", "bad"


def _strategy_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        v, cls = _verdict(r)
        name = NAME_FA.get(r["strategy"], r["strategy"])
        out.append(
            f"<tr><td><b>{_e(name)}</b><div class='muted' style='font-size:.78rem'>{_e(r['source'])}</div></td>"
            f"<td class='num'>{_e(r['symbol'])}</td><td class='num'>{r['trades_per_year']:.0f}</td>"
            f"<td class='num'>{r['is_gross_bp']:+.2f}</td><td class='num'>{r['is_net_bp']:+.2f}</td>"
            f"<td class='num'>{r['oos_gross_bp']:+.2f}</td><td class='num'>{r['oos_net_bp']:+.2f}</td>"
            f"<td class='num'>{r['oos_t']:+.2f}</td><td class='num'>{r['oos_q']:.2f}</td>"
            f"<td class='{cls}'>{v}</td></tr>")
    head = ("<thead><tr><th>قاعده و منبع</th><th>نماد</th><th>معامله در سال</th><th>درون‌نمونه ناخالص</th>"
            "<th>درون‌نمونه خالص</th><th>برون‌نمونه ناخالص</th><th>برون‌نمونه خالص</th><th>t خالص</th>"
            "<th>q (FDR)</th><th>نتیجه</th></tr></thead>")
    return f"<div class='scroll'><table>{head}<tbody>{''.join(out)}</tbody></table></div>"


def _long_history(js: dict | None) -> str:
    if not js:
        return "<p class='muted'>اجرای ۱۶ ساله هنوز ثبت نشده است.</p>"
    syms = [s for s in ("EURUSD", "XAUUSD") if s in js]
    cols = syms + (["portfolio"] if "portfolio" in js else [])
    label = {"EURUSD": "EURUSD", "XAUUSD": "XAUUSD", "portfolio": "سبد ترکیبی"}
    years = sorted({y for c in cols for y in js[c].get("by_year", {})})
    body = []
    for y in years:
        cells = []
        for c in cols:
            v = js[c].get("by_year", {}).get(y)
            cells.append("<td class='num'>–</td>" if v is None else
                         f"<td class='num {'good' if v > 0 else 'bad'}'>{v * 100:+.1f}%</td>")
        body.append(f"<tr><td class='num'>{y}</td>{''.join(cells)}</tr>")
    stats_rows = []
    keys = (("trades", "تعداد معامله", "{:.0f}"), ("win_rate", "نرخ برد", "{:.0%}"),
            ("profit_factor", "ضریب سود", "{:.2f}"), ("sharpe", "شارپ", "{:.2f}"),
            ("max_drawdown", "افت حداکثر", "{:+.1%}"), ("cagr", "بازده سالانه مرکب", "{:+.1%}"))
    for k, fa, f in keys:
        cells = []
        for c in cols:
            runs = js[c].get("runs", {})
            st = js[c].get("stats") if c == "portfolio" else runs.get("production_nohalt", runs.get("production_oos", {}))
            v = (st or {}).get(k)
            cells.append(f"<td class='num'>{f.format(v) if isinstance(v, (int, float)) else '–'}</td>")
        stats_rows.append(f"<tr><td>{fa}</td>{''.join(cells)}</tr>")
    month = ""
    pm = js.get("portfolio", {}).get("monthly")
    if pm:
        v = np.array(list(pm.values()), dtype=float)
        month = (f"<p>ماه‌های سبد ترکیبی: میانگین <b class='num'>{v.mean() * 100:+.2f}%</b>، میانه "
                 f"<b class='num'>{np.median(v) * 100:+.2f}%</b>، ماه‌های مثبت <b class='num'>{(v > 0).mean() * 100:.0f}%</b>، "
                 f"بهترین <b class='num'>{v.max() * 100:+.1f}%</b> و بدترین <b class='num'>{v.min() * 100:+.1f}%</b>.</p>")
    oos = js.get(syms[0], {}).get("ml_oos_start", "")[:10] if syms else ""
    head = "<thead><tr><th></th>" + "".join(f"<th>{label[c]}</th>" for c in cols) + "</tr></thead>"
    runs_fa = (("production_nohalt", "استراتژی ربات"), ("daily_bias_only_nohalt", "ورود ساعتی فقط با سوگیری روزانه"),
               ("random_entries_nohalt", "ورود تصادفی با همان مدیریت معامله"))
    cmp_rows = []
    for sym in syms:
        for k, fa in runs_fa:
            st = js[sym].get("runs", {}).get(k) or {}
            if not st.get("trades"):
                continue
            cmp_rows.append(
                f"<tr><td class='num'>{sym}</td><td>{fa}</td><td class='num'>{st['trades']}</td>"
                f"<td class='num'>{st['profit_factor']:.2f}</td><td class='num'>{st['sharpe']:+.2f}</td>"
                f"<td class='num {'good' if st['cagr'] > 0 else 'bad'}'>{st['cagr']:+.1%}</td>"
                f"<td class='num'>{st['max_drawdown']:+.1%}</td></tr>")
    cmp_head = ("<thead><tr><th>نماد</th><th>نسخه</th><th>معامله</th><th>ضریب سود</th><th>شارپ</th>"
                "<th>بازده سالانه</th><th>افت حداکثر</th></tr></thead>")
    return (f"<p class='muted'>کد همان کد ربات است، فقط داده ساعتی به‌جای ۲ سال یاهو، ۱۶ سال HistData است. "
            f"یادگیری ماشین walk-forward است و دوره خارج از نمونه از <span class='num'>{_e(oos)}</span> شروع می‌شود. "
            f"ریسک هر معامله ۰٫۵٪ است و هزینه‌ها کسر شده‌اند. ترمز افت ۱۲٪ در بک‌تست دائمی است (یورو در ۲۰۱۶ و طلا در ۲۰۱۹ "
            f"به آن خوردند و دیگر معامله نکردند)، پس اعداد زیر بدون توقف دائمی‌اند؛ نصف شدن ریسک در افت ۶٪ فعال است.</p>"
            f"<p class='bad'>نتیجه: بخش ساعتی ربات در این ۱۲٫۷ سال سود نداد. از ورود تصادفی بهتر بود، ولی نه به اندازه هزینه معامله.</p>{month}"
            f"<div class='grid2'><div class='scroll'><table>{head}<tbody>{''.join(stats_rows)}</tbody></table></div>"
            f"<div class='scroll'><table>{head}<tbody>{''.join(body)}</tbody></table></div></div>"
            f"<h3 style='margin-top:14px'>در برابر نسخه‌های ساده‌تر</h3>"
            f"<div class='scroll'><table>{cmp_head}<tbody>{''.join(cmp_rows)}</tbody></table></div>")


def _risk_scaling(js: dict | None) -> str:
    if not js:
        return ""
    rows = []
    for risk, st in js.items():
        cls = "bad" if st["max_drawdown"] < -0.35 else ""
        rows.append(
            f"<tr class='{cls}'><td class='num'>{float(risk) * 100:.1f}%</td><td class='num'>{st['cagr']:+.1%}</td>"
            f"<td class='num'>{st['median_month']:+.2%}</td><td class='num'>{st['mean_month']:+.2%}</td>"
            f"<td class='num'>{st['positive_months']:.0%}</td><td class='num'>{st['best_month']:+.1%}</td>"
            f"<td class='num'>{st['worst_month']:+.1%}</td><td class='num'>{st['worst_12m']:+.1%}</td>"
            f"<td class='num'>{st['max_drawdown']:+.1%}</td><td class='num'>{st['months_ge_10pct']} از {st['months']}</td></tr>")
    head = ("<thead><tr><th>ریسک هر معامله</th><th>بازده سالانه</th><th>میانه ماه</th><th>میانگین ماه</th><th>ماه مثبت</th>"
            "<th>بهترین ماه</th><th>بدترین ماه</th><th>بدترین ۱۲ ماه</th><th>افت حداکثر</th><th>ماه‌های ۱۰٪+</th></tr></thead>")
    return ("<h3 style='margin-top:14px'>اگر ریسک هر معامله بیشتر بود</h3>"
            "<p class='muted'>همان معاملات ۱۶ ساله با ریسک بزرگ‌تر دوباره مرکب شده‌اند. ترمز افت سرمایه دوباره شبیه‌سازی نشده، "
            "پس ردیف‌های پرریسک اگر خطایی داشته باشند، خوش‌بینانه‌اند.</p>"
            f"<div class='scroll'><table>{head}<tbody>{''.join(rows)}</tbody></table></div>")


def _swing_leverage(js: dict | None) -> str:
    if not js:
        return ""
    rows = []
    for tv, d in js["targets"].items():
        a = d["all"]
        cls = "bad" if a["max_drawdown"] < -0.35 else ""
        rows.append(
            f"<tr class='{cls}'><td class='num'>{float(tv) * 100:.0f}%</td><td class='num good'>{a['cagr']:+.1%}</td>"
            f"<td class='num'>{a['sharpe']:.2f}</td><td class='num'>{a['max_drawdown']:+.1%}</td>"
            f"<td class='num'>{a['median_month']:+.2%}</td><td class='num'>{a['worst_month']:+.1%}</td>"
            f"<td class='num'>{a['worst_12m']:+.1%}</td><td class='num'>{a['months_ge_10pct']} از {a['months']}</td></tr>")
    head = ("<thead><tr><th>هدف نوسان سالانه</th><th>بازده سالانه</th><th>شارپ</th><th>افت حداکثر</th><th>میانه ماه</th>"
            "<th>بدترین ماه</th><th>بدترین ۱۲ ماه</th><th>ماه‌های ۱۰٪+</th></tr></thead>")
    p0, p1 = js["period"]
    return (f"<p>دفتر سوئینگ (سوگیری روزانه × اهرم هدف‌نوسان، یورو و طلا با سرمایه برابر) از <span class='num'>{p0[:4]}</span> تا "
            f"<span class='num'>{p1[:4]}</span> مثبت بود: شارپ یورو <span class='num'>{js['books_sharpe']['EURUSD']}</span>، طلا "
            f"<span class='num'>{js['books_sharpe']['XAUUSD']}</span> و همبستگی دو دفتر <span class='num'>{js['corr']}</span>. "
            "اجزای این لایه روی ۲۰۰۵ تا ۲۰۲۳ انتخاب شده‌اند، پس فقط ۲۰۲۴ به بعد کاملاً برون‌نمونه است.</p>"
            f"<div class='scroll'><table>{head}<tbody>{''.join(rows)}</tbody></table></div>"
            "<p class='muted'>مسیر واقع‌بینانه سود بیشتر، اهرم بیشتر روی همین لایه است، با پذیرفتن افت بزرگ‌تر. حتی با نوسان ۲۰٪ "
            "میانگین حدود ۱٪ در ماه است.</p>")


def _chronos(js: dict | None) -> str:
    if not js:
        return "<p class='muted'>نتیجه‌ای ثبت نشده است.</p>"
    rows = []
    for key, s in js.items():
        model, sym, freq = key.split("|")
        rows.append(
            f"<tr><td>{_e(CHRONOS_FA.get(model, model))}</td><td class='num'>{sym}</td>"
            f"<td>{'روزانه' if freq == 'daily' else 'ساعتی'}</td><td class='num'>{s['n']}</td>"
            f"<td class='num'>{s['hit_rate']:.1%}</td><td class='num'>{s['base_rate']:.1%}</td>"
            f"<td class='num'>{s['naive_momentum_hit']:.1%}</td><td class='num'>{s['p_hit_vs_50pct']:.2f}</td>"
            f"<td class='num {'good' if s['net_sharpe'] > 0 else 'bad'}'>{s['net_sharpe']:+.2f}</td></tr>")
    head = ("<thead><tr><th>مدل</th><th>نماد</th><th>بازه</th><th>تعداد پیش‌بینی</th><th>دقت جهت</th>"
            "<th>نرخ پایه</th><th>دقت مومنتوم ساده</th><th>p</th><th>شارپ خالص</th></tr></thead>")
    return f"<div class='scroll'><table>{head}<tbody>{''.join(rows)}</tbody></table></div>"


def _img(path: Path, alt: str) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"<div class='figure'><img loading='lazy' alt='{_e(alt)}' src='data:image/png;base64,{b64}'></div>"


def lab_section(survey, hypotheses, chart_notes, sources) -> str:
    compact = _load("round3_compact.json")
    if not compact:
        return ""
    rows = compact["table"]
    n_pass = sum(bool(r["passes_retail"]) for r in rows)
    n_pos = sum(r["oos_net_bp"] > 0 for r in rows)
    rho = _rank_corr([r["is_net_bp"] for r in rows], [r["oos_net_bp"] for r in rows])
    top = sorted([r for r in rows if r["oos_net_bp"] > 0], key=lambda r: -r["oos_t"])[:10]
    kpis = (f"<div class='kpis'>"
            f"<div class='kpi'><b>{len(rows)}</b><span>قاعده آزمون‌شده (۳۰ قاعده × ۲ نماد)</span></div>"
            f"<div class='kpi'><b>{n_pass}</b><span>قاعده‌ای که پس از هزینه و آزمون چندگانه قبول شد</span></div>"
            f"<div class='kpi'><b>{n_pos}</b><span>قاعده با بازده خالص مثبت در ۲۰۱۹ تا ۲۰۲۶ (همه غیرمعنادار)</span></div>"
            f"<div class='kpi'><b>{rho:+.2f}</b><span>همبستگی رتبه‌ای درون‌نمونه و برون‌نمونه</span></div></div>")
    cards = "".join(
        f"<div class='card'><b>{_e(t)}</b><p>{_e(d)}</p><p class='muted' style='font-size:.8rem'>"
        + " · ".join(f"<a href='{_e(u)}' target='_blank' rel='noopener'>{_e(lbl)}</a>" for lbl, u in links)
        + "</p></div>" for t, d, links in survey)
    hyp = "".join(f"<li><b>{_e(t)}</b> {_e(d)}</li>" for t, d in hypotheses)
    notes = ""
    for sym in ("EURUSD", "XAUUSD"):
        paras = "".join(f"<li>{_e(p)}</li>" for p in chart_notes.get(sym, []))
        notes += (f"<div class='card'><h3>{sym} · بسته‌شدن {_e(chart_notes.get('asof', ''))}</h3>"
                  f"<ul class='why' style='color:var(--fg)'>{paras}</ul>"
                  f"<details><summary>نمودار D1 / H4 / H1 / M15</summary>"
                  f"{_img(IMG / f'{sym}_mtf.png', f'{sym} multi-timeframe chart')}</details></div>")
    src = "".join(f"<li><a href='{_e(u)}' target='_blank' rel='noopener'>{_e(t)}</a></li>" for t, u in sources)
    LEAD = ("بیش از ۴۰ جست‌وجو در مقالات، گیت‌هاب، بازار اکسپرت‌ها، پراپ‌فرم‌ها، کانال‌های سیگنال و مسابقه‌های هوش مصنوعی "
            "انجام شد. سپس ۳۰ قاعده پرتکرار با همان پارامترهای منبعشان روی داده یک‌دقیقه‌ای یورو و طلا (۲۰۱۰ تا ۲۰۲۶) آزمون شدند: "
            "۲۰۱۰ تا ۲۰۱۸ درون‌نمونه و ۲۰۱۹ تا ۲۰۲۶ برون‌نمونه، با کسر هزینه معامله. نتیجه: هیچ قاعده‌ای پس از هزینه و کنترل آزمون "
            "چندگانه (Benjamini-Hochberg) قبول نشد. آنچه پشت ربات مانده، یعنی روند روزانه و اختلاف نرخ بهره، همچنان قوی‌ترین لایه است.")
    return f"""
<section class="desk" id="lab">
  <header><h2>آزمایشگاه پژوهش: دور سوم</h2>
    <span class="muted">۱۶ سال داده یک‌دقیقه‌ای · مقالات، ربات‌ها، کانال‌ها و استراتژی‌های محبوب · کنترل آزمون چندگانه</span></header>
  <div class="card ai"><p class="lead">{LEAD}</p></div>
  {kpis}
  <h3>بازار ربات‌ها، سیگنال‌ها و مقالات چه می‌گوید</h3>
  <div class="findings">{cards}</div>
  <div class="card"><h3>آیا لبه درون‌نمونه دوام آورد؟</h3>{_scatter_fig(rows)}
    <p class="muted">اگر قاعده‌ها لبه واقعی داشتند، نقطه‌ها روی خط‌چین قطری و در ربع بالا-راست جمع می‌شدند. بیشترشان زیر صفرند.</p></div>
  <div class="card"><h3>بهترین‌ها در برون‌نمونه (مرتب بر اساس t)</h3>{_strategy_rows(top)}
    <p class="muted">واحد: واحد پایه (bp) در هر معامله. هزینه خرد: ۱٫۲ پیپ برای یورو و ۰٫۴ دلار برای طلا در هر رفت‌وبرگشت.
    q همان p تعدیل‌شده برای ۶۰ آزمون است؛ برای قبول باید زیر ۰٫۰۵ باشد.</p></div>
  <div class="card"><h3>منحنی سود تجمعی چند قاعده شاخص</h3>{_equity_fig(compact, 'EURUSD')}{_equity_fig(compact, 'XAUUSD')}
    <p class="muted">W3 (الگوی فیکس‌های کرون-مولر-ویلان) تا ۲۰۱۸ عالی بود و بعد از انتشار، آرام سقوط کرد. قاعده‌های سودده طلا
    در ۲۰۲۴ تا ۲۰۲۶ همان روند صعودی خود طلا را گرفته‌اند (خط «مبنا»).</p></div>
  <div class="card"><h3>بازده میانگین در هر ساعت</h3>{_hours_fig(compact, 'EURUSD')}{_hours_fig(compact, 'XAUUSD')}
    <p class="muted">میله‌های بزرگ ساعت ۲۱ تا ۲۳ UTC واقعی نیستند: داده فقط قیمت Bid دارد و در زمان رول‌اور ۵ عصر نیویورک
    اسپرد باز می‌شود (افت ساختگی و برگشت بعدی). معامله‌گر خرد در این ساعت‌ها همان اسپرد را می‌پردازد.</p></div>
  <details class="card"><summary>جدول کامل ۶۰ آزمون</summary>{_strategy_rows(rows)}</details>
  <div class="card"><h3>اعتبارسنجی ۱۶ ساله استراتژی اصلی ربات</h3>{_long_history(_load('long_history_backtest.json'))}
    {_risk_scaling(_load('risk_scaling.json'))}</div>
  <div class="card"><h3>تنها لایه با شواهد بلندمدت: دفتر سوئینگ روزانه</h3>{_swing_leverage(_load('swing_leverage.json'))}</div>
  <div class="card"><h3>مدل‌های پایه پیش‌بینی سری زمانی (آمازون Chronos)</h3>
    <p class="muted">فقط روی داده‌های پس از انتشار هر مدل آزمون شد تا داده آموزشی به نتیجه نشت نکند. سیگنال: جهت میانه پیش‌بینی یک‌گام بعد.</p>
    {_chronos(_load('foundation_models.json'))}</div>
  <h3>تحلیل چندزمانه چارت</h3>
  <div class="grid2">{notes}</div>
  <div class="card"><h3>فرضیه‌هایی که رو به جلو ثبت می‌شوند (بدون معامله)</h3><ul class="why" style="color:var(--fg)">{hyp}</ul></div>
  <details class="card"><summary>منابع ({len(sources)})</summary><ul class="src">{src}</ul></details>
</section>"""
