"""End-to-end orchestration: research (walk-forward backtest) and live analysis."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysts.ensemble import explain_latest, run_ensemble
from .backtest.engine import desk_only_signal, performance, run_backtest, signal_mask
from .config import DEFAULT_CONFIG, INSTRUMENTS, Instrument, StrategyConfig
from .data import loader
from .features import build_features, ml_columns
from .indicators.patterns import describe_geometry, harmonic_prz
from .indicators.structure import sr_levels_at
from .ml.models import DirectionModel, MetaModel
from .risk.manager import label_outcomes, position_size, risk_fraction, stop_distances
from .strategic import LABELS_FA, attach_bias, daily_bias, load_daily

warnings.filterwarnings("ignore")


@dataclass
class Bundle:
    inst: Instrument
    df: pd.DataFrame
    f: pd.DataFrame
    state: object
    stops: pd.DataFrame
    labels: pd.DataFrame
    ens: pd.DataFrame | None = None
    meta_p: pd.Series | None = None
    dir_model: DirectionModel | None = None
    meta_model: MetaModel | None = None
    extras: dict = field(default_factory=dict)


def load_inputs(symbol: str, csv: str | None = None, use_cache: bool = True):
    inst = INSTRUMENTS[symbol]
    df = loader.load_instrument(inst, "1h", csv=csv, use_cache=use_cache)
    try:
        peer = loader.fetch_yahoo(inst.smt_peer, "1h", use_cache=use_cache)
    except Exception:
        peer = None
    try:
        macro = loader.load_macro_daily()
    except Exception:
        macro = None
    try:
        cot = loader.fetch_cot(inst.cot_code)
    except Exception:
        cot = None
    return inst, df, peer, macro, cot


def prepare(symbol: str, cfg: StrategyConfig = DEFAULT_CONFIG, csv: str | None = None,
            use_cache: bool = True) -> Bundle:
    inst, df, peer, macro, cot = load_inputs(symbol, csv, use_cache)
    f, state = build_features(df, inst, peer, macro, cot, return_state=True)
    daily = load_daily(symbol, inst.yahoo)
    bias = daily_bias(symbol, daily, macro, target_vol=cfg.swing_target_vol)
    f = f.join(attach_bias(f.index, bias.drop(columns=["close"])))
    stops = stop_distances(f, cfg)
    labels = label_outcomes(f, stops, inst, cfg)
    return Bundle(inst, df, f, state, stops, labels,
                  extras={"daily_bias": bias, "daily_close": daily, "macro": macro})


def research(symbol: str, cfg: StrategyConfig = DEFAULT_CONFIG, csv: str | None = None,
             bundle: Bundle | None = None) -> dict:
    b = bundle or prepare(symbol, cfg, csv)
    f = b.f.copy()
    cols = ml_columns(f)
    dm = DirectionModel(cols)
    f = f.join(dm.walk_forward(f, b.labels, cfg))
    ens = run_ensemble(f, symbol)
    mm = MetaModel(cols)
    meta_cfg = StrategyConfig(**{**cfg.__dict__, "wf_initial_frac": min(0.85, cfg.wf_initial_frac + 0.2)})
    meta_p = mm.walk_forward(f, ens, b.labels, meta_cfg)
    b.f, b.ens, b.meta_p, b.dir_model, b.meta_model = f, ens, meta_p, dm, mm

    n = len(f)
    ml_start = int(n * cfg.wf_initial_frac)
    meta_start = int(n * meta_cfg.wf_initial_frac)
    dev_end = int(n * 0.75)          # research holdout boundary (see docs/RESEARCH.md)
    sig = signal_mask(f, ens, cfg, meta_p)
    sig_meta = signal_mask(f, ens, StrategyConfig(**{**cfg.__dict__, "use_meta": True}), meta_p)
    sig_desk = desk_only_signal(f, ens, cfg)
    bias = f["sb_bias"].fillna(0)
    sig_bias = pd.Series(np.where((bias.abs() >= cfg.bias_min) & ~f["hour"].isin(cfg.avoid_hours_utc),
                                  np.sign(bias), 0.0), index=f.index)

    def _until(s_, i):
        s_ = s_.copy()
        s_.iloc[i:] = 0.0
        return s_

    out = {"symbol": symbol, "bars": n, "start": str(f.index[0]), "end": str(f.index[-1]),
           "ml_oos_start": str(f.index[ml_start]), "meta_oos_start": str(f.index[meta_start]),
           "holdout_start": str(f.index[dev_end]),
           "direction_model": {k: v.summary() for k, v in dm.reports.items()},
           "meta_model": mm.report.summary() if mm.report else {}}
    runs = {
        "production_oos": run_backtest(f, sig, b.stops, b.inst, cfg, start=ml_start),
        "production_dev": run_backtest(f, _until(sig, dev_end), b.stops, b.inst, cfg, start=ml_start),
        "production_holdout": run_backtest(f, sig, b.stops, b.inst, cfg, start=dev_end),
        "with_meta_filter": run_backtest(f, sig_meta, b.stops, b.inst, cfg, meta_p, start=meta_start),
        "daily_bias_only": run_backtest(f, sig_bias, b.stops, b.inst, cfg, start=ml_start),
        "desk_only_v1": run_backtest(f, sig_desk, b.stops, b.inst, cfg, start=ml_start),
    }
    rng = np.random.default_rng(42)
    rnd = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.02, 0.96, 0.02]), index=f.index)
    runs["random_entries"] = run_backtest(f, rnd, b.stops, b.inst, cfg, start=ml_start)
    out["buy_and_hold_return"] = round(float(f["close"].iloc[-1] / f["close"].iloc[ml_start] - 1), 4)
    out["buy_and_hold_holdout"] = round(float(f["close"].iloc[-1] / f["close"].iloc[dev_end] - 1), 4)
    out["runs"] = {k: v["stats"] for k, v in runs.items()}
    out["swing"] = swing_stats(b)
    out["_runs"] = runs
    out["_bundle"] = b
    out["feature_importance"] = []
    return out


def swing_returns(b: Bundle) -> pd.Series:
    """Daily P&L of the swing mode: position = bias x vol-target leverage."""
    bias = b.extras["daily_bias"]
    c = b.extras["daily_close"].copy()
    idx = pd.DatetimeIndex(c.index)
    c.index = (idx.tz_convert(None) if idx.tz is not None else idx).normalize()
    c = c[~c.index.duplicated(keep="last")]
    r = c.pct_change().reindex(bias.index)
    pos = (bias["bias"] * bias["leverage"]).fillna(0.0)
    return (pos * r - pos.diff().abs() * b.inst.spread / c.reindex(bias.index)).dropna()


def swing_stats(b: Bundle) -> dict:
    pr = swing_returns(b)

    def st(x):
        if len(x) < 60 or x.std() == 0:
            return {}
        eq = (1 + x).cumprod()
        return {"sharpe": round(float(x.mean() / x.std() * np.sqrt(252)), 2),
                "ann_return": round(float(x.mean() * 252), 4),
                "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
                "years": round(len(x) / 252, 1)}
    return {"2005_2023": st(pr["2005":"2023"]), "2024_now": st(pr["2024":]),
            "2005_2011": st(pr["2005":"2011"]), "2012_2017": st(pr["2012":"2017"]),
            "2018_2023": st(pr["2018":"2023"])}


# ------------------------------------------------------------------ live

def _fib_levels(lo: float, hi: float, leg_dir: float) -> dict:
    ratios = [0.236, 0.382, 0.5, 0.618, 0.705, 0.786]
    if leg_dir > 0:   # up-leg: retracements measured down from the high
        lv = {f"{r:.3f}": hi - r * (hi - lo) for r in ratios}
        lv.update({"ext_1.272": lo + 1.272 * (hi - lo), "ext_1.618": lo + 1.618 * (hi - lo)})
    else:
        lv = {f"{r:.3f}": lo + r * (hi - lo) for r in ratios}
        lv.update({"ext_1.272": hi - 1.272 * (hi - lo), "ext_1.618": hi - 1.618 * (hi - lo)})
    return lv


def _plan(side: int, row: pd.Series, stop_d: float, inst: Instrument, cfg: StrategyConfig,
          equity: float, risk_frac: float) -> dict:
    entry = float(row["close"])
    stop = entry - side * stop_d
    tp1 = entry + side * cfg.partial_at_r * stop_d
    tp2 = entry + side * cfg.rr_target * stop_d
    liq = []
    keys_up = ["res", "pdh", "eqh", "asia_high", "pwh", "last_sh"]
    keys_dn = ["sup", "pdl", "eql", "asia_low", "pwl", "last_sl"]
    for k in (keys_up if side > 0 else keys_dn):
        v = row.get(k, np.nan)
        if np.isfinite(v) and 0.3 * row["atr"] < (v - entry) * side < 8 * row["atr"]:
            liq.append((k, float(v)))
    liq = sorted(liq, key=lambda x: abs(x[1] - entry))[:4]
    lots = position_size(equity, risk_frac, stop_d, inst)
    d = inst.digits
    return {"side": "LONG" if side > 0 else "SHORT", "entry_ref": round(entry, d),
            "stop": round(stop, d), "tp1_partial": round(tp1, d), "tp2_final": round(tp2, d),
            "risk_reward_final": cfg.rr_target, "stop_distance": round(stop_d, d),
            "stop_distance_pips": round(stop_d / inst.pip, 1),
            "liquidity_targets": [(k, round(v, d)) for k, v in liq],
            "risk_fraction": round(risk_frac, 4),
            "lots_for_equity": round(lots, 2), "equity_assumed": equity,
            "management": f"close {int(cfg.partial_fraction*100)}% at TP1 (+{cfg.partial_at_r}R), move stop to "
                          f"breakeven, hold rest to TP2 (+{cfg.rr_target}R) or {cfg.time_stop_bars}h time stop"}


def analyze_live(symbol: str, cfg: StrategyConfig = DEFAULT_CONFIG, csv: str | None = None,
                 equity: float = 10_000.0, use_cache: bool = True) -> dict:
    res = research(symbol, cfg, csv)
    b: Bundle = res["_bundle"]
    f, ens = b.f, b.ens
    # refresh the most recent predictions with models trained on all completed labels
    cols = b.dir_model.cols
    dm = DirectionModel(cols)
    last = dm.fit_and_predict_last(f, b.labels)
    for c in last.columns:
        f.loc[f.index[-200:], c] = last[c].iloc[-200:]
    ens = run_ensemble(f, symbol)
    mm = MetaModel(cols)
    mp_last = mm.fit_and_predict_last(f, ens, b.labels)
    meta_p = b.meta_p.copy()
    meta_p.iloc[-200:] = mp_last.iloc[-200:]
    b.ens, b.meta_p = ens, meta_p
    res["feature_importance"] = dm.importance()

    row, e = f.iloc[-1], ens.iloc[-1]
    inst = b.inst
    sig = signal_mask(f, ens, cfg, meta_p)
    side = int(sig.iloc[-1])
    comp = float(e["composite"])
    sb = float(row.get("sb_bias", 0.0))
    bias = int(np.sign(sb)) if sb != 0 else (1 if comp > 0 else -1)

    # news blackout
    now = f.index[-1] + pd.Timedelta("1h")
    try:
        events = loader.high_impact_events(inst.news_currencies, now, 72)
    except Exception:
        events = pd.DataFrame()
    blackout = False
    ev_list = []
    for _, ev in events.iterrows():
        dt = (ev["date"] - now).total_seconds() / 3600
        ev_list.append({"time_utc": str(ev["date"]), "currency": ev["country"], "title": ev["title"],
                        "forecast": ev.get("forecast", ""), "previous": ev.get("previous", ""),
                        "hours_from_now": round(dt, 1)})
        if -0.5 <= dt <= 1.0:
            blackout = True

    p_meta = float(meta_p.iloc[-1]) if np.isfinite(meta_p.iloc[-1]) else None
    dd = 0.0
    rf = risk_fraction(cfg, p_meta if cfg.use_meta else None, dd)
    stop_d = float(b.stops["stop_long" if bias > 0 else "stop_short"].iloc[-1])
    plan = _plan(bias, row, stop_d, inst, cfg, equity, rf)

    if side != 0 and not blackout:
        decision = "LONG" if side > 0 else "SHORT"
    else:
        decision = "WAIT"
    reasons_block = []
    sbias = float(row.get("sb_bias", 0.0))
    if side == 0:
        if abs(sbias) < cfg.bias_min:
            reasons_block.append(f"daily strategic bias {sbias:+.2f} is not strong enough (needs |bias| >= {cfg.bias_min})")
        elif comp * np.sign(sbias) <= cfg.comp_align:
            reasons_block.append(f"H1 desk composite {comp:+.3f} does not confirm the daily bias {sbias:+.2f}")
        if cfg.use_meta and p_meta is not None and p_meta < cfg.meta_threshold:
            reasons_block.append(f"meta-model P(win) {p_meta:.0%} < {cfg.meta_threshold:.0%}")
        if row["hour"] in cfg.avoid_hours_utc:
            reasons_block.append("rollover/thin-liquidity hour")
    if blackout:
        reasons_block.append("high-impact news within the blackout window")

    # key levels
    st = b.state
    zones, seen = [], set()
    for z in reversed([z for z in st.zones if z.active]):
        key = (z.kind, round(z.top, 6), round(z.bottom, 6))
        if key not in seen and abs((z.top + z.bottom) / 2 - row["close"]) < 25 * row["atr"]:
            seen.add(key)
            zones.append(z.as_dict(f.index))
    zones = zones[:10][::-1]
    t = len(f) - 1
    sr = sr_levels_at(b.df, f[["ph", "pl"]], t, row["atr"])
    sr = sorted(sr, key=lambda x: abs(x[0] - row["close"]))[:8]
    swings = st.swings[-12:]
    fib = {}
    if np.isfinite(row.get("last_sh", np.nan)) and np.isfinite(row.get("last_sl", np.nan)):
        fib = _fib_levels(row["last_sl"], row["last_sh"], row.get("leg_dir", 1) or 1)
    prz = {}
    piv = [(p, k, v) for p, k, v, _ in swings]
    if len(piv) >= 4:
        X, A, B, C = (x[2] for x in piv[-4:])
        prz = harmonic_prz(X, A, B, C)
    geometry = describe_geometry(piv, row["atr"])
    elliott = next((lab for lab in reversed(f["elliott_label"].tail(60).tolist()) if lab), "")
    harm = next((lab for lab in reversed(f["harmonic_name"].tail(24).tolist()) if lab), "")

    d = inst.digits
    live = {
        "symbol": symbol, "time_utc": str(f.index[-1]), "price": round(float(row["close"]), d),
        "data_source": inst.yahoo if not csv else csv,
        "decision": decision, "bias": "BULLISH" if bias > 0 else "BEARISH",
        "composite": round(comp, 3), "threshold": round(float(e["threshold"]), 3),
        "agreement": round(float(e["agreement"]), 3), "confidence": round(float(e["confidence"]), 3),
        "meta_p_win": None if p_meta is None else round(p_meta, 3),
        "ml_p_long": round(float(row.get("ml_p_long", np.nan)), 3),
        "ml_p_short": round(float(row.get("ml_p_short", np.nan)), 3),
        "strategic": {
            "bias": round(sb, 3),
            "components": {k: {"value": float(row.get(f"sb_{k}", np.nan)), "label_fa": LABELS_FA.get(k, k)}
                           for k in b.extras["daily_bias"].attrs.get("components", [])},
            "vol_target_leverage": round(float(row.get("sb_leverage", np.nan)), 2),
            "swing_position_pct_equity": round(float(sb * row.get("sb_leverage", 0.0)) * 100, 1),
            "realized_vol_annual": round(float(row.get("sb_realized_vol", np.nan)), 3),
        },
        "swing_backtest": res["swing"], "holdout_start": res["holdout_start"],
        "buy_and_hold_holdout": res["buy_and_hold_holdout"],
        "regime": e["regime"], "trend_strength": round(float(e["trend_strength"]), 3),
        "vol_state": e["vol_state"], "atr": round(float(row["atr"]), d),
        "why_not": reasons_block, "plan": plan, "analysts": explain_latest(f, ens, symbol),
        "levels": {
            "support_resistance": [(round(lv, d), round(s, 1)) for lv, s in sr],
            "zones": [{**z, "top": round(z["top"], d), "bottom": round(z["bottom"], d),
                       "start": str(z["start"]), "created": str(z["created"])} for z in zones],
            "session": {k: round(float(row[k]), d) for k in
                        ("pdh", "pdl", "pdc", "pwh", "pwl", "asia_high", "asia_low", "day_open",
                         "week_open", "pivot", "r1", "s1", "r2", "s2", "poc", "vah", "val", "eqh", "eql")
                        if k in row and np.isfinite(row[k]) and abs(row[k] - row["close"]) < 30 * row["atr"]},
            "fibonacci_last_leg": {k: round(v, d) for k, v in fib.items()},
            "harmonic_prz": {k: (round(a, d), round(bb, d)) for k, (a, bb) in prz.items()},
            "swings": [(str(f.index[p]) if p < len(f) else "", k, round(v, d), lab) for p, k, v, lab in swings],
        },
        "patterns": {"elliott": elliott, "harmonic_recent": harm, "geometry": geometry},
        "htf": {"h4_trend": float(row.get("h4_ms_trend", 0)), "d1_trend": float(row.get("d1_ms_trend", 0)),
                "h4_rsi": round(float(row.get("h4_rsi", np.nan)), 1),
                "d1_rsi": round(float(row.get("d1_rsi", np.nan)), 1)},
        "news": ev_list[:12], "news_blackout": blackout,
        "backtest": res["runs"], "buy_and_hold_return": res["buy_and_hold_return"],
        "ml_reports": {"direction": res["direction_model"], "meta": res["meta_model"]},
        "feature_importance": res["feature_importance"],
    }
    res["live"] = live
    return res
