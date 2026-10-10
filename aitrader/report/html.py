"""Persian (RTL) trading-desk dashboard for one or more instruments."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import numpy as np

PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/4.1.1/plotly.min.js"

CSS = """
/* Layout: one reading column of instrument "desks", each opening with a verdict strip,
   then evidence (analysts, chart, levels) and the out-of-sample record. */
:root{
  --bg:#f3f5f8; --surface:#ffffff; --fg:#18212c; --muted:#5f6b7a; --line:#dde2ea;
  --brass:#a8741f; --buy:#1d7f55; --sell:#b83a33; --wait:#6d6a86;
  --buy-bg:#e3f3ea; --sell-bg:#f8e5e3; --wait-bg:#ecebf3; --brass-bg:#f6eedf;
  --f-display:"Vazirmatn","Tahoma",sans-serif; --f-body:"Vazirmatn","Tahoma",sans-serif;
  --f-num:"IBM Plex Mono","Menlo","Consolas",monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f141a; --surface:#171e27; --fg:#e5e9ef; --muted:#97a2b1; --line:#2a3440;
  --brass:#d6a64f; --buy:#4cc28d; --sell:#ef7a70; --wait:#a9a6c6;
  --buy-bg:#14302a; --sell-bg:#3a1e1d; --wait-bg:#252438; --brass-bg:#2d2618; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f141a; --surface:#171e27; --fg:#e5e9ef; --muted:#97a2b1; --line:#2a3440;
  --brass:#d6a64f; --buy:#4cc28d; --sell:#ef7a70; --wait:#a9a6c6;
  --buy-bg:#14302a; --sell-bg:#3a1e1d; --wait-bg:#252438; --brass-bg:#2d2618; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:15px;line-height:1.75;margin:0}
.wrap{max-width:1180px;margin:0 auto;padding-inline:16px;padding-block:28px 64px;display:flex;flex-direction:column;gap:28px}
h1,h2,h3{font-family:var(--f-display);text-wrap:balance;margin:0;line-height:1.35}
h1{font-size:1.75rem;font-weight:800}
h2{font-size:1.3rem;font-weight:800}
h3{font-size:1.02rem;font-weight:700;color:var(--muted)}
.num,td.num,.mono{font-family:var(--f-num);font-variant-numeric:tabular-nums;direction:ltr;unicode-bidi:isolate}
.eyebrow{font-size:.78rem;letter-spacing:.06em;color:var(--brass);font-weight:700}
.muted{color:var(--muted)}
header.top{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:12px;border-bottom:2px solid var(--fg);padding-bottom:14px}
header.top p{margin:4px 0 0;color:var(--muted);max-width:68ch}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px;min-width:0}
.verdict{display:flex;flex-direction:column;gap:12px}
.vhead{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.sym{font-family:var(--f-num);font-weight:600;font-size:1.25rem;letter-spacing:.04em}
.pill{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:2px 12px;font-weight:800;font-size:.9rem}
.pill.LONG{background:var(--buy-bg);color:var(--buy)} .pill.SHORT{background:var(--sell-bg);color:var(--sell)}
.pill.WAIT{background:var(--wait-bg);color:var(--wait)}
.kv{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.kv div{display:flex;flex-direction:column}
.kv span:first-child{font-size:.75rem;color:var(--muted)}
.kv span:last-child{font-family:var(--f-num);font-variant-numeric:tabular-nums;font-weight:600;direction:ltr;text-align:right}
.meter{position:relative;height:10px;border-radius:5px;background:linear-gradient(90deg,var(--buy) 0%,var(--line) 50%,var(--sell) 100%);opacity:.9}
.meter i{position:absolute;top:-4px;width:4px;height:18px;border-radius:2px;background:var(--fg)}
.meter-l{display:flex;justify-content:space-between;font-size:.72rem;color:var(--muted)}
.plan{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;background:var(--brass-bg);border-radius:8px;padding:10px}
.plan div{display:flex;flex-direction:column}
.plan span:first-child{font-size:.72rem;color:var(--muted)}
.plan span:last-child{font-family:var(--f-num);font-weight:600;direction:ltr;text-align:right}
.why{margin:0;padding-inline-start:18px;color:var(--muted);font-size:.88rem}
section.desk{display:flex;flex-direction:column;gap:16px;scroll-margin-top:12px}
section.desk>header{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:8px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th{font-size:.76rem;color:var(--muted);font-weight:700;letter-spacing:.03em;white-space:nowrap}
td.num{text-align:left;white-space:nowrap}
.bar{position:relative;width:110px;height:8px;background:var(--line);border-radius:4px;display:inline-block;vertical-align:middle;direction:ltr}
.bar b{position:absolute;top:0;height:8px;border-radius:4px}
.bar b.pos{left:50%;background:var(--buy)} .bar b.neg{right:50%;background:var(--sell)}
.reasons{font-size:.82rem;color:var(--muted);margin:2px 0 0;padding:0;list-style:none}
.reasons li{direction:ltr;text-align:left}
.ai{border-inline-start:4px solid var(--brass);white-space:pre-line}
.ai .lead{font-size:1rem}
.good{color:var(--buy);font-weight:700} .bad{color:var(--sell);font-weight:700}
tr.bad td{color:var(--sell)}
.chart{min-height:420px;direction:ltr;min-width:640px}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.findings .card p{margin:6px 0 0;font-size:.9rem;color:var(--muted)}
.tag{display:inline-block;font-size:.72rem;border:1px solid var(--line);border-radius:4px;padding:0 6px;color:var(--muted)}
details summary{cursor:pointer;font-weight:700}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:4px}
.kpi b{font-family:var(--f-num);font-size:1.7rem;font-weight:600;direction:ltr;text-align:right;line-height:1.2}
.kpi span{font-size:.8rem;color:var(--muted)}
.figure{background:#fff;border-radius:8px;padding:6px;margin-top:10px;overflow-x:auto}
.figure img{width:100%;min-width:560px;height:auto;display:block}
.src{columns:2 320px;font-size:.85rem;padding-inline-start:18px}
.src li{break-inside:avoid;margin-bottom:4px;direction:ltr;text-align:left}
footer{font-size:.82rem;color:var(--muted);border-top:1px solid var(--line);padding-top:14px}
a{color:var(--brass)} a:focus-visible,summary:focus-visible{outline:2px solid var(--brass);outline-offset:2px}
@media (max-width:520px){.kv{grid-template-columns:repeat(2,minmax(0,1fr))}.plan{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""

FA = {"LONG": "خرید", "SHORT": "فروش", "WAIT": "صبر",
      "trending": "روند‌دار", "ranging": "رنج", "transition": "گذار",
      "high-vol": "نوسان بالا", "low-vol": "نوسان پایین", "normal-vol": "نوسان عادی"}

RUN_FA = {
    "production_dev": "استراتژی نهایی - دوره توسعه (OOS ML)",
    "production_holdout": "استراتژی نهایی - دوره کنار گذاشته (Holdout)",
    "production_oos": "استراتژی نهایی - کل دوره خارج از نمونه",
    "daily_bias_only": "فقط سوگیری روزانه",
    "with_meta_filter": "با فیلتر متا-مدل",
    "desk_only_v1": "نسخه ۱: فقط میز ساعتی (رد شد)",
    "random_entries": "ورود تصادفی (معیار شانس)",
}


def _e(x) -> str:
    return html.escape(str(x))


def _fmt(v, pct=False, d=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v*100:+.1f}%" if pct else f"{v:.{d}f}"


def _bar(score: float) -> str:
    w = min(abs(score), 1) * 50
    cls = "pos" if score >= 0 else "neg"
    return f'<span class="bar"><b class="{cls}" style="width:{w:.0f}%"></b></span>'


def _verdict_card(L: dict, ai: dict | None) -> str:
    st = L.get("strategic", {})
    b = st.get("bias", 0)
    pos = (b + 1) / 2 * 100   # meter is drawn LTR with bullish on the left
    p = L.get("plan", {})
    dec = (ai or {}).get("final_decision", L["decision"])
    why = "".join(f"<li>{_e(w)}</li>" for w in L.get("why_not", []))
    plan_title = "پلن معامله" if dec != "WAIT" else f"پلن مشروط ({FA.get(p.get('side'), '')} در صورت تأیید)"
    return f"""
<article class="card verdict" aria-label="{_e(L['symbol'])}">
  <div class="vhead"><span class="sym">{_e(L['symbol'])}</span>
    <span class="pill {dec}">{FA.get(dec, dec)}</span></div>
  <div class="kv">
    <div><span>قیمت</span><span>{L['price']}</span></div>
    <div><span>برآیند میز ساعتی</span><span>{L['composite']:+.3f}</span></div>
    <div><span>رژیم</span><span style="font-family:var(--f-body)">{FA.get(L['regime'], L['regime'])}</span></div>
  </div>
  <div>
    <div class="meter-l"><span>سوگیری روزانه: <b class="num">{b:+.2f}</b></span><span>{_e(L['time_utc'][:16])} UTC</span></div>
    <div class="meter" style="direction:ltr" role="img" aria-label="daily bias {b:+.2f}"><i style="left:calc({100-pos:.0f}% - 2px)"></i></div>
    <div class="meter-l" style="direction:ltr"><span>bullish +1</span><span>0</span><span>-1 bearish</span></div>
  </div>
  <h3>{plan_title}</h3>
  <div class="plan">
    <div><span>ورود</span><span>{p.get('entry_ref')}</span></div>
    <div><span>حد ضرر</span><span>{p.get('stop')}</span></div>
    <div><span>تارگت ۱ (۵۰٪)</span><span>{p.get('tp1_partial')}</span></div>
    <div><span>تارگت نهایی</span><span>{p.get('tp2_final')}</span></div>
  </div>
  {f'<ul class="why">{why}</ul>' if why else ''}
</article>"""


def _analyst_table(L: dict) -> str:
    rows = []
    for a in sorted(L["analysts"], key=lambda x: -abs(x["score"])):
        rs = "".join(f"<li>{_e(r)}</li>" for r in a["reasons"][:4])
        rows.append(f"<tr><td><b>{_e(a['title_fa'])}</b><ul class='reasons'>{rs}</ul></td>"
                    f"<td class='num'>{a['score']:+.2f}</td><td>{_bar(a['score'])}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>تحلیلگر و دلایل</th><th>امتیاز</th><th>جهت</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def _strategic_table(L: dict) -> str:
    st = L["strategic"]
    rows = "".join(f"<tr><td>{_e(v['label_fa'])}</td><td class='num'>{v['value']:+.2f}</td></tr>"
                   for v in st["components"].values())
    return (f"<div class='scroll'><table><thead><tr><th>جزء سوگیری روزانه (وزن برابر)</th><th>مقدار</th></tr></thead>"
            f"<tbody>{rows}<tr><td><b>سوگیری نهایی</b></td><td class='num'><b>{st['bias']:+.2f}</b></td></tr>"
            f"<tr><td>اهرم هدف‌گذاری نوسان ۱۰٪ (حالت سوئینگ)</td><td class='num'>{st['vol_target_leverage']}×</td></tr>"
            f"<tr><td>موقعیت سوئینگ پیشنهادی (٪ ارزش اسمی به سرمایه)</td><td class='num'>{st['swing_position_pct_equity']:+.1f}%</td></tr>"
            "</tbody></table></div>")


def _levels_table(L: dict) -> str:
    lv = L["levels"]
    rows = []
    for k, v in lv["session"].items():
        rows.append(f"<tr><td class='mono' style='text-align:right'>{_e(k.upper())}</td><td class='num'>{v}</td></tr>")
    for z in lv["zones"]:
        rows.append(f"<tr><td>{_e(z['kind'].replace('_', ' ').upper())} <span class='tag'>{z['touched']} لمس</span></td>"
                    f"<td class='num'>{z['bottom']} – {z['top']}</td></tr>")
    for lvl, s in lv["support_resistance"]:
        rows.append(f"<tr><td>S/R خوشه‌ای <span class='tag'>قدرت {s}</span></td><td class='num'>{lvl}</td></tr>")
    for k, v in lv["fibonacci_last_leg"].items():
        rows.append(f"<tr><td>فیبوناچی {k}</td><td class='num'>{v}</td></tr>")
    for k, (a, b) in lv["harmonic_prz"].items():
        rows.append(f"<tr><td>PRZ هارمونیک {k}</td><td class='num'>{a} – {b}</td></tr>")
    pat = L.get("patterns", {})
    extra = []
    if pat.get("elliott"):
        extra.append(f"الیوت: {pat['elliott']}")
    if pat.get("harmonic_recent"):
        extra.append(f"هارمونیک اخیر: {pat['harmonic_recent']}")
    extra += pat.get("geometry", [])
    ex = "".join(f"<li>{_e(x)}</li>" for x in extra)
    return (f"<div class='scroll'><table><thead><tr><th>سطح</th><th>قیمت</th></tr></thead><tbody>{''.join(rows)}"
            f"</tbody></table></div>{f'<ul class=reasons>{ex}</ul>' if ex else ''}")


def _news_table(L: dict) -> str:
    if not L.get("news"):
        return "<p class='muted'>خبر پراهمیتی در ۷۲ ساعت آینده در تقویم این هفته نیست.</p>"
    rows = "".join(f"<tr><td class='num'>{_e(n['time_utc'][:16])}</td><td>{_e(n['currency'])}</td>"
                   f"<td style='direction:ltr;text-align:left'>{_e(n['title'])}</td><td class='num'>{n['hours_from_now']:+.1f}h</td>"
                   f"<td class='num'>{_e(n.get('forecast') or '')}</td></tr>" for n in L["news"])
    return ("<div class='scroll'><table><thead><tr><th>زمان UTC</th><th>ارز</th><th>رویداد</th><th>فاصله</th><th>پیش‌بینی</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>")


def _news_ai(L: dict) -> str:
    v = L.get("news_view")
    if not v:
        return ""
    rows = "".join(
        f"<tr><td class='num'>{_e(h['time'][11:16])}</td><td style='direction:ltr;text-align:left'>{_e(h['title'])}"
        f" <span class='tag'>{_e(h['source'])}</span></td><td class='num {'good' if h['impact'] > 0 else 'bad'}'>"
        f"{h['impact']:+d}</td></tr>" for h in L.get("news_headlines", []))
    risk = " · <b class='bad'>ریسک رویداد</b>" if v.get("event_risk") else ""
    return (f"<div class='card ai'><div class='eyebrow'>تحلیلگر اخبار (هوش مصنوعی) · امتیاز "
            f"<span class='num'>{v['score']:+.2f}</span> از {v['n']} خبر مرتبط{risk}</div>"
            f"<p class='lead'>{_e(v.get('rationale_fa', ''))}</p>"
            f"<p class='muted'>{_e(L.get('news_theme_fa', ''))}</p>"
            "<p class='muted'>اخبار فقط می‌تواند معامله را متوقف کند (وتو)؛ اثرش روی سود هنوز به‌صورت رو به جلو در حال سنجش است.</p>"
            f"<details><summary>اخبار اثرگذار</summary><div class='scroll'><table><thead><tr><th>UTC</th><th>تیتر</th>"
            f"<th>اثر</th></tr></thead><tbody>{rows}</tbody></table></div></details></div>")


def _bt_table(L: dict) -> str:
    rows = []
    for k, label in RUN_FA.items():
        s = L["backtest"].get(k)
        if not s or not s.get("trades"):
            continue
        ret = s.get("total_return", 0)
        rows.append(f"<tr><td>{_e(label)}</td><td class='num'>{s['trades']}</td><td class='num'>{_fmt(s.get('win_rate'), True)}</td>"
                    f"<td class='num'>{_fmt(s.get('avg_r'))}</td><td class='num'>{_fmt(s.get('profit_factor'))}</td>"
                    f"<td class='num'>{_fmt(s.get('sharpe'))}</td><td class='num'>{_fmt(s.get('max_drawdown'), True)}</td>"
                    f"<td class='num {'good' if ret > 0 else 'bad'}'>{_fmt(ret, True)}</td><td class='num'>{_fmt(s.get('psr'))}</td></tr>")
    sw = L.get("swing_backtest", {})
    srows = "".join(f"<tr><td>{_e(k.replace('_', '–'))}</td><td class='num'>{_fmt(v.get('sharpe'))}</td>"
                    f"<td class='num'>{_fmt(v.get('ann_return'), True)}</td><td class='num'>{_fmt(v.get('max_drawdown'), True)}</td>"
                    f"<td class='num'>{v.get('years', '–')}</td></tr>" for k, v in sw.items() if v)
    return (f"<p class='muted'>هر معامله با ریسک ۰٫۵٪ سرمایه، اسپرد و لغزش واقعی، ورود در کندل بعد. شروع Holdout: "
            f"<span class='num'>{_e(L['holdout_start'][:10])}</span> (خرید و نگهداری در همین دوره: "
            f"<span class='num'>{_fmt(L['buy_and_hold_holdout'], True)}</span>). PSR = احتمال اینکه شارپ واقعی مثبت باشد.</p>"
            "<div class='scroll'><table><thead><tr><th>اجرا</th><th>معاملات</th><th>برد</th><th>میانگین R</th><th>PF</th>"
            f"<th>شارپ</th><th>افت حداکثر</th><th>بازده</th><th>PSR</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            "<h3>حالت سوئینگ روزانه (۲۰+ سال داده)</h3><div class='scroll'><table><thead><tr><th>دوره</th><th>شارپ</th>"
            f"<th>بازده سالانه</th><th>افت حداکثر</th><th>سال</th></tr></thead><tbody>{srows}</tbody></table></div>")


def _ai_block(ai: dict | None) -> str:
    if not ai:
        return ""
    if ai.get("mode") == "llm":
        d = ai["decision"]
        notes = "".join(f"<li>{_e(n)}</li>" for n in d.get("guardrail_notes", []))
        risks = "".join(f"<li>{_e(r)}</li>" for r in d.get("key_risks", []))
        watch = "".join(f"<li>{_e(r)}</li>" for r in d.get("levels_to_watch", []))
        return f"""<div class="card ai"><div class="eyebrow">میز هوش مصنوعی ({_e(ai.get('provider'))}: {_e(ai.get('model'))}) - مناظره خریدار/فروشنده</div>
<p class="lead">{_e(d.get('summary_fa', ''))}</p>
<p class="muted">اطمینان: <span class="num">{d.get('conviction')}</span>/100 · سناریوی اصلی: {_e(d.get('primary_scenario'))}</p>
<p class="muted">سناریوی جایگزین: {_e(d.get('alternative_scenario'))} · ابطال: {_e(d.get('invalidation'))}</p>
{f'<h3>سطوح زیر نظر</h3><ul class="reasons">{watch}</ul>' if watch else ''}
{f'<h3>ریسک‌ها</h3><ul class="reasons">{risks}</ul>' if risks else ''}
{f'<ul class="why">{notes}</ul>' if notes else ''}
<details><summary>استدلال خریدار (Bull)</summary><p style="direction:ltr;text-align:left">{_e(ai.get('bull_case', ''))}</p></details>
<details><summary>استدلال فروشنده (Bear)</summary><p style="direction:ltr;text-align:left">{_e(ai.get('bear_case', ''))}</p></details>
</div>"""
    return f"""<div class="card ai"><div class="eyebrow">جمع‌بندی تحلیلگر</div><p class="lead">{_e(ai.get('summary_fa', ''))}</p></div>"""


def _account_section(acc: dict | None) -> str:
    if not acc or "combined" not in acc:
        return ""
    def pct(x, d=1):
        return "–" if x is None else f"{x*100:+.{d}f}%"
    months = acc["monthly_h1_portfolio"]
    mrows = "".join(f"<tr><td class='num'>{_e(k)}</td><td class='num {'good' if v > 0 else 'bad'}'>{pct(v)}</td></tr>"
                    for k, v in months.items())
    vals = list(months.values())
    pos = sum(1 for v in vals if v > 0) / max(1, len(vals))

    def mc_rows(rows):
        out = []
        for r in rows:
            danger = r["p_drawdown_50pct"] >= 0.05 or r["median_max_dd"] <= -0.3
            out.append(f"<tr{' class=bad' if danger else ''}><td class='num'>{r['target_vol']*100:.0f}%</td>"
                       f"<td class='num'>{r['leverage_x']:.1f}×</td><td class='num'>{pct(r['median_month'])}</td>"
                       f"<td class='num'>{pct(r['mean_month'])}</td><td class='num'>{r['p_month_ge_10pct']*100:.0f}%</td>"
                       f"<td class='num'>{pct(r['median_max_dd'], 0)}</td><td class='num'>{pct(r['p95_max_dd'], 0)}</td>"
                       f"<td class='num'>{r['p_drawdown_50pct']*100:.0f}%</td></tr>")
        return "".join(out)
    head = ("<thead><tr><th>نوسان سالانه هدف</th><th>اهرم</th><th>میانه ماه</th><th>میانگین ماه</th><th>احتمال ماه ≥۱۰٪</th>"
            "<th>میانه افت حداکثر</th><th>افت بد (۵٪ بدترین)</th><th>احتمال نصف شدن حساب</th></tr></thead>")
    k = acc["kelly"]
    c, h1, sw = acc["combined"], acc["h1"], acc["swing"]
    return f"""
<section class="desk" id="account"><header><h2>حساب ترکیبی و هدف ۱۰٪ ماهانه</h2>
<span class="muted num">{_e(acc['window'][0])} → {_e(acc['window'][1])}</span></header>
<div class="card"><p><b class="bad">هشدار:</b> این بخش بر پایه بک‌تست ۲۰ ماه اخیر (یاهو) است. اعتبارسنجی ۱۶ ساله نشان داد بخش ساعتی ربات در ۲۰۱۴ تا ۲۰۲۶ پس از هزینه سود نداد (ضریب سود ۰٫۹۵ تا ۱٫۰۰) و فقط دفتر سوئینگ روزانه شواهد مثبت بلندمدت دارد (شارپ حدود ۰٫۶۷). پس اعداد زیر خوش‌بینانه‌اند؛ جدول واقع‌بینانه در بخش <a href="#lab">آزمایشگاه پژوهش</a> است.</p></div>
<div class="grid2">
 <div class="card"><h3>دو استراتژی با هم</h3>
  <div class="scroll"><table><thead><tr><th>بخش</th><th>شارپ</th><th>بازده سالانه</th><th>نوسان</th><th>افت حداکثر</th></tr></thead><tbody>
  <tr><td>تاکتیکی ساعتی (ریسک ۰٫۵٪)</td><td class="num">{h1['sharpe']}</td><td class="num">{pct(h1['ann_return'])}</td><td class="num">{pct(h1['vol'])}</td><td class="num">{pct(h1['max_dd'])}</td></tr>
  <tr><td>سوئینگ روزانه (نوسان ۱۰٪ تقسیم بر دو نماد)</td><td class="num">{sw['sharpe']}</td><td class="num">{pct(sw['ann_return'])}</td><td class="num">{pct(sw['vol'])}</td><td class="num">{pct(sw['max_dd'])}</td></tr>
  <tr><td><b>ترکیب با ریسک برابر</b></td><td class="num"><b>{c['sharpe']}</b></td><td class="num">{pct(c['ann_return'])}</td><td class="num">{pct(c['vol'])}</td><td class="num">{pct(c['max_dd'])}</td></tr>
  <tr><td>سوئینگ، ۲۰۰۵ تا ۲۰۲۳ (بلندمدت)</td><td class="num">{acc['swing_20y']['sharpe']}</td><td class="num">{pct(acc['swing_20y']['ann_return'])}</td><td class="num">{pct(acc['swing_20y']['vol'])}</td><td class="num">{pct(acc['swing_20y']['max_dd'])}</td></tr>
  </tbody></table></div>
  <p class="muted">همبستگی دو استراتژی: <span class="num">{acc['corr_h1_swing']}</span>. چون کم است، ترکیب شارپ را بالا می‌برد و افت را کم می‌کند. افزودن جفت‌ارزهای دیگر کمکی نکرد (همه تابع دلارند).</p>
 </div>
 <div class="card"><h3>سود ماهانه پرتفوی ساعتی (ریسک ۰٫۵٪)</h3>
  <p class="muted">{len(vals)} ماه · ماه‌های مثبت <span class="num">{pos*100:.0f}%</span> · میانگین <span class="num">{pct(float(np.mean(vals)), 2)}</span></p>
  <details><summary>جدول ماه‌به‌ماه</summary><div class="scroll"><table><thead><tr><th>ماه</th><th>بازده</th></tr></thead><tbody>{mrows}</tbody></table></div></details>
 </div>
</div>
<div class="card"><h3>با چه ریسکی به ۱۰٪ ماهانه می‌رسیم؟ (مونت‌کارلو ۴۰۰۰ مسیر یک‌ساله)</h3>
 <p>طبق فرمول کلی، حداکثر رشد مرکب ممکن حتی با اهرم بهینه حدود SR²/2 در سال است. با شارپ بک‌تست <span class="num">{k['backtested']['sharpe']:.2f}</span>
 سقف نظری <span class="num">{k['backtested']['max_growth_month']*100:.1f}%</span> در ماه است و با شارپ واقع‌بینانه ۰٫۹ حدود <span class="num">{k['conservative']['max_growth_month']*100:.1f}%</span>.
 برای ۱۰٪ ماهانه شارپ پایدار حدود <span class="num">{k['backtested']['sharpe_needed_for_10pct_month']:.1f}</span> لازم است که در عمل برای معامله‌گر خرد دست‌نیافتنی است.
 «میانه» یعنی ماه معمولی؛ «میانگین» را چند مسیر خوش‌شانس بالا می‌کشد. ردیف‌های قرمز یعنی ریسک نابودی جدی.</p>
 <h3>اگر آینده مثل بک‌تست باشد</h3><div class="scroll"><table>{head}<tbody>{mc_rows(acc['mc_backtested'])}</tbody></table></div>
 <h3>اگر لبه واقعی نصف بک‌تست باشد (معمول در اجرای زنده)</h3><div class="scroll"><table>{head}<tbody>{mc_rows(acc['mc_haircut'])}</tbody></table></div>
 <p class="muted">پیشنهاد: ریسک ۰٫۵٪ تا حداکثر ۱٪ در هر معامله (نوسان ۱۰ تا ۲۰٪). تنظیم با <span class="mono">--risk 0.01</span>.</p>
</div>
</section>"""


def build_dashboard(items: list[dict], findings: list[tuple[str, str]], standalone: bool = True,
                    ai_commentary: dict | None = None, account: dict | None = None, lab: str = "") -> str:
    """items: [{"live": live_dict, "ai": ai_dict, "chart": plotly_div, "equity": plotly_div}]"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards = "".join(_verdict_card(it["live"], it.get("ai")) for it in items)
    desks = []
    for it in items:
        L = it["live"]
        extra = ""
        if ai_commentary and L["symbol"] in ai_commentary:
            extra = f"<div class='card ai'><div class='eyebrow'>یادداشت پژوهشگر</div><p class='lead'>{_e(ai_commentary[L['symbol']])}</p></div>"
        desks.append(f"""
<section class="desk" id="{_e(L['symbol'].lower())}">
  <header><h2>میز {_e(L['symbol'])}</h2><span class="muted">منبع داده: <span class="mono">{_e(L['data_source'])}</span> · ATR(H1) <span class="num">{L['atr']}</span> · {FA.get(L['vol_state'], '')}</span></header>
  {extra}
  {_ai_block(it.get('ai'))}
  {_news_ai(L)}
  <div class="grid2">
    <div class="card"><h3>لایه راهبردی روزانه</h3>{_strategic_table(L)}</div>
    <div class="card"><h3>اخبار پراهمیت پیش‌رو</h3>{_news_table(L)}</div>
  </div>
  <div class="card"><h3>نمودار H1 با نواحی اسمارت‌مانی، سطوح و پلن</h3><div class="scroll"><div class="chart">{it.get('chart', '')}</div></div></div>
  <div class="grid2">
    <div class="card"><h3>میز ۱۲ تحلیلگر (H1)</h3>{_analyst_table(L)}</div>
    <div class="card"><h3>سطوح کلیدی</h3>{_levels_table(L)}</div>
  </div>
  <div class="card"><h3>کارنامه خارج از نمونه</h3>{_bt_table(L)}<div class="scroll">{it.get('equity', '')}</div></div>
</section>""")
    fnd = "".join(f"<div class='card'><b>{_e(t)}</b><p>{_e(d)}</p></div>" for t, d in findings)
    body = f"""
<title>میز معاملاتی یورو و طلا</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;800&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<script src="{PLOTLY_CDN}"></script>
<div class="wrap" dir="rtl" lang="fa">
  <header class="top">
    <div><div class="eyebrow">AI TRADING DESK · EURUSD · XAUUSD</div>
      <h1>میز معاملاتی یورو و طلا</h1>
      <p>سوگیری روزانه (آزموده‌شده روی ۲۰ سال داده) + میز ۱۲ تحلیلگر ساعتی + یادگیری ماشین walk-forward + مدیریت ریسک. تحلیل است، توصیه سرمایه‌گذاری نیست.</p></div>
    <div class="muted num">{now}</div>
  </header>
  <div class="summary">{cards}</div>
  {_account_section(account)}
  <section class="desk"><header><h2>یافته‌های پژوهش</h2></header><div class="findings">{fnd}</div></section>
  {''.join(desks)}
  {lab}
  <footer>این گزارش خودکار توسط موتور aitrader تولید شده است. نتایج گذشته تضمینی برای آینده نیست؛ همه آمارها پس از هزینه معامله و خارج از نمونه‌اند مگر خلافش ذکر شده باشد. قیمت طلا از قرارداد آتی COMEX (GC=F) است و با قیمت اسپات بروکر چند دلار اختلاف دارد.</footer>
</div>
<script>
(function(){{
  function themed(){{
    var cs=getComputedStyle(document.documentElement);
    var fg=cs.getPropertyValue('--fg').trim(), line=cs.getPropertyValue('--line').trim();
    document.querySelectorAll('.js-plotly-plot').forEach(function(el){{
      try{{
        var upd={{'font.color':fg,'legend.font.color':fg}};
        Object.keys(el.layout||{{}}).forEach(function(k){{ if(/^[xy]axis\\d*$/.test(k)){{upd[k+'.gridcolor']=line; upd[k+'.linecolor']=line; upd[k+'.zerolinecolor']=line;}} }});
        Plotly.relayout(el,upd);
      }}catch(e){{}}
    }});
  }}
  window.addEventListener('load',themed);
  try{{ window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',themed);
       new MutationObserver(themed).observe(document.documentElement,{{attributes:true,attributeFilter:['data-theme']}}); }}catch(e){{}}
}})();
</script>
"""
    if standalone:
        return ("<!doctype html><html lang='fa' dir='rtl'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1,viewport-fit=cover'></head><body>"
                + body + "</body></html>")
    return body


def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1,
                      default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
