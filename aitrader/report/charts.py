"""Chart rendering: static PNG (for the AI vision input) and interactive Plotly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ZONE_COLORS = {"bull_ob": "rgba(38,166,91,0.18)", "bear_ob": "rgba(214,69,65,0.18)",
               "bull_fvg": "rgba(52,152,219,0.16)", "bear_fvg": "rgba(155,89,182,0.16)"}
ZONE_MPL = {"bull_ob": ("#26a65b", 0.18), "bear_ob": ("#d64541", 0.18),
            "bull_fvg": ("#3498db", 0.15), "bear_fvg": ("#9b59b6", 0.15)}


def render_png(f: pd.DataFrame, live: dict, path: str | Path, bars: int = 180) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = f.tail(bars)
    x = np.arange(len(d))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1]}, dpi=110)
    up = d["close"] >= d["open"]
    ax.vlines(x, d["low"], d["high"], color=np.where(up, "#26a65b", "#d64541"), lw=0.8)
    ax.bar(x, (d["close"] - d["open"]).abs(), bottom=np.minimum(d["open"], d["close"]),
           color=np.where(up, "#26a65b", "#d64541"), width=0.7)
    for col, c, lw in (("ema20", "#f39c12", 1.0), ("ema50", "#2980b9", 1.0), ("ema200", "#7f8c8d", 1.4)):
        if col in d:
            ax.plot(x, d[col], color=c, lw=lw, label=col.upper())
    t0 = d.index[0]
    p = live.get("plan")
    lo = min(d["low"].min(), p["stop"] if p else np.inf, p["tp2_final"] if p else np.inf)
    hi = max(d["high"].max(), p["stop"] if p else -np.inf, p["tp2_final"] if p else -np.inf)
    pad = (hi - lo) * 0.05
    for z in live["levels"]["zones"]:
        if z["bottom"] > hi + pad or z["top"] < lo - pad:
            continue
        start = pd.Timestamp(z["start"])
        xs = max(0, int(np.searchsorted(d.index, start)))
        col, al = ZONE_MPL[z["kind"]]
        ax.fill_between([xs, len(d) + 3], z["bottom"], z["top"], color=col, alpha=al, lw=0)
        ax.text(len(d) + 3, (z["top"] + z["bottom"]) / 2, z["kind"].replace("_", " ").upper(), fontsize=7,
                color=col, clip_on=True)
    for lv, s in live["levels"]["support_resistance"]:
        ax.axhline(lv, color="#555", lw=0.6, ls=":", alpha=0.7)
    ses = live["levels"]["session"]
    for k, c in (("pdh", "#8e44ad"), ("pdl", "#8e44ad"), ("asia_high", "#16a085"), ("asia_low", "#16a085")):
        if k in ses:
            ax.axhline(ses[k], color=c, lw=0.8, ls="--", alpha=0.8)
            ax.text(1, ses[k], k.upper(), fontsize=7, color=c, va="bottom", clip_on=True)
    if p:
        for k, c in (("entry_ref", "#2c3e50"), ("stop", "#c0392b"), ("tp1_partial", "#27ae60"), ("tp2_final", "#1e8449")):
            ax.axhline(p[k], color=c, lw=1.2, alpha=0.9)
            ax.text(len(d) - 1, p[k], f" {k} {p[k]}", fontsize=8, color=c, va="bottom", ha="right", clip_on=True)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-1, len(d) + 12)
    ax.legend(loc="upper left", fontsize=8)
    st = live.get("strategic", {})
    ax.set_title(f"{live['symbol']} H1  |  {live['time_utc'][:16]} UTC  |  decision {live['decision']}  |  "
                 f"daily bias {st.get('bias', 0):+.2f}  desk {live['composite']:+.3f}  |  regime {live['regime']}",
                 fontsize=10)
    ax.grid(alpha=0.2)
    ax2.plot(x, d["rsi"], color="#8e44ad", lw=1)
    ax2.axhline(70, color="#999", lw=0.6, ls="--")
    ax2.axhline(30, color="#999", lw=0.6, ls="--")
    ax2.set_ylabel("RSI14")
    ticks = np.linspace(0, len(d) - 1, 8).astype(int)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([d.index[i].strftime("%m-%d %H:%M") for i in ticks], fontsize=8)
    _ = t0
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plotly_chart(f: pd.DataFrame, live: dict, bars: int = 300) -> str:
    """Interactive chart as an HTML <div> (plotly.js loaded by the page)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    d = f.tail(bars)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.19, 0.19],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="H1", increasing_line_color="#26a65b", decreasing_line_color="#d64541"), 1, 1)
    for col, c in (("ema20", "#f39c12"), ("ema50", "#2980b9"), ("ema200", "#95a5a6"), ("st_line", "#e67e22")):
        if col in d:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col.upper(), line=dict(color=c, width=1),
                                     opacity=0.9), 1, 1)
    x_end = d.index[-1] + pd.Timedelta(hours=12)
    span = d["high"].max() - d["low"].min()
    vis_lo, vis_hi = d["low"].min() - 0.3 * span, d["high"].max() + 0.3 * span
    for z in live["levels"]["zones"]:
        if z["bottom"] > vis_hi or z["top"] < vis_lo:
            continue
        fig.add_shape(type="rect", x0=max(pd.Timestamp(z["start"]), d.index[0]), x1=x_end, y0=z["bottom"],
                      y1=z["top"], fillcolor=ZONE_COLORS[z["kind"]], line_width=0, row=1, col=1)
        fig.add_annotation(x=x_end, y=z["top"], text=z["kind"].replace("_", " ").upper(), showarrow=False,
                           font=dict(size=9), xanchor="left", row=1, col=1)
    for lv, s in live["levels"]["support_resistance"]:
        fig.add_hline(y=lv, line=dict(color="#888", width=0.7, dash="dot"), row=1, col=1)
    ses = live["levels"]["session"]
    for k, c in (("pdh", "#8e44ad"), ("pdl", "#8e44ad"), ("asia_high", "#16a085"), ("asia_low", "#16a085"),
                 ("pwh", "#34495e"), ("pwl", "#34495e")):
        if k in ses:
            fig.add_hline(y=ses[k], line=dict(color=c, width=1, dash="dash"), annotation_text=k.upper(),
                          annotation_font_size=9, row=1, col=1)
    p = live.get("plan")
    if p:
        for k, c in (("entry_ref", "#2c3e50"), ("stop", "#c0392b"), ("tp1_partial", "#27ae60"),
                     ("tp2_final", "#1e8449")):
            fig.add_hline(y=p[k], line=dict(color=c, width=1.5), annotation_text=f"{k} {p[k]}",
                          annotation_font_size=10, row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI14", line=dict(color="#8e44ad", width=1)), 2, 1)
    fig.add_hline(y=70, line=dict(color="#aaa", width=0.6, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#aaa", width=0.6, dash="dash"), row=2, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["macd_hist"], name="MACD hist (ATR)",
                         marker_color=np.where(d["macd_hist"] >= 0, "#26a65b", "#d64541")), 3, 1)
    fig.update_layout(height=760, margin=dict(l=10, r=60, t=30, b=10), xaxis_rangeslider_visible=False,
                      template="plotly_white", legend=dict(orientation="h", y=1.04, font=dict(size=10)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def plotly_equity(curves: dict[str, pd.Series], title: str) -> str:
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, s in curves.items():
        if s is None or len(s) == 0:
            continue
        s = s.dropna()
        s = s.resample("1D").last().dropna() if len(s) > 800 else s
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=36, b=10), title=dict(text=title, font=dict(size=13)),
                      template="plotly_white", legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
