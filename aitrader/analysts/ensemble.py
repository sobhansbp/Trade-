"""Specialist analysts + regime-aware ensemble ("trading desk").

Each analyst turns the causal feature matrix into a score in [-1, 1]
(+1 strongly bullish) and can explain its view for the latest bar.
The ensemble blends them with weights that morph between a *trend* and a
*range* profile according to a continuous regime estimate, then measures how
much the desk agrees before anything becomes a trade candidate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _t(x, k=1.0):
    return np.tanh(np.asarray(x, dtype=float) / k)


def _s(f: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in f:
        return f[col].astype(float).fillna(default)
    return pd.Series(default, index=f.index)


@dataclass
class Analyst:
    name: str
    title_fa: str

    def score(self, f: pd.DataFrame) -> pd.Series:  # pragma: no cover - interface
        raise NotImplementedError

    def explain(self, row: pd.Series) -> list[str]:
        return []


class TrendAnalyst(Analyst):
    def __init__(self):
        super().__init__("trend", "روند چندزمانی")

    def score(self, f):
        h1 = (_s(f, "ema_stack") + _s(f, "st_dir") + _s(f, "ichi_pos") + 0.5 * _s(f, "ichi_tk")
              + _t(_s(f, "ema20_slope"), 0.5) + 0.5 * _s(f, "psar_dir")) / 5.0
        h4 = (_s(f, "h4_ema_stack") + _s(f, "h4_st_dir") + _s(f, "h4_ms_trend")
              + _t(_s(f, "h4_ema_fast_slope"), 0.5)) / 4.0
        d1 = (_s(f, "d1_ema_stack") + _s(f, "d1_st_dir") + _s(f, "d1_ms_trend")
              + _t(_s(f, "d1_ema_fast_slope"), 0.5)) / 4.0
        strength = 0.6 + 0.4 * ((_s(f, "adx", 20) - 15) / 20).clip(0, 1)
        return (0.35 * h1 + 0.35 * h4 + 0.30 * d1).clip(-1, 1) * strength

    def explain(self, r):
        def word(x):
            return "bullish" if x > 0 else "bearish" if x < 0 else "flat"
        return [f"H1 EMA stack {word(r.get('ema_stack', 0))}, Supertrend {word(r.get('st_dir', 0))}, "
                f"Ichimoku {'above' if r.get('ichi_pos', 0) > 0 else 'below' if r.get('ichi_pos', 0) < 0 else 'inside'} cloud",
                f"H4 structure {word(r.get('h4_ms_trend', 0))}, D1 structure {word(r.get('d1_ms_trend', 0))}",
                f"ADX {r.get('adx', np.nan):.1f} (+DI-−DI {r.get('di_diff', 0)*100:+.1f})"]


class MomentumAnalyst(Analyst):
    def __init__(self):
        super().__init__("momentum", "مومنتوم")

    def score(self, f):
        s = (0.25 * _t((_s(f, "rsi", 50) - 50), 15)
             + 0.20 * _t(_s(f, "macd_hist"), 0.25)
             + 0.10 * np.sign(_s(f, "stoch_k", 50) - _s(f, "stoch_d", 50))
             + 0.15 * _t(_s(f, "roc10"), 1.5)
             + 0.15 * _t(_s(f, "tsmom_120"), 1.5)
             + 0.15 * _t(_s(f, "tsmom_480"), 1.5))
        return pd.Series(s, index=f.index).clip(-1, 1)

    def explain(self, r):
        return [f"RSI14 {r.get('rsi', np.nan):.1f}, MACD hist {r.get('macd_hist', 0):+.2f} ATR, "
                f"Stoch {r.get('stoch_k', np.nan):.0f}/{r.get('stoch_d', np.nan):.0f}",
                f"time-series momentum 5d {r.get('tsmom_120', 0):+.2f}σ, 20d {r.get('tsmom_480', 0):+.2f}σ"]


class MeanReversionAnalyst(Analyst):
    def __init__(self):
        super().__init__("mean_reversion", "بازگشت به میانگین")

    def score(self, f):
        bbz = _s(f, "bb_z")
        rsi3 = _s(f, "rsi3", 50)
        vw = _s(f, "px_vs_vwap")
        raw = -(0.45 * _t(bbz, 1.5) + 0.30 * _t(rsi3 - 50, 30) + 0.25 * _t(vw, 2.0))
        extreme = ((bbz.abs() > 1.6) | (rsi3 > 88) | (rsi3 < 12)).astype(float)
        hurst_mr = (0.55 - _s(f, "hurst", 0.5)).clip(0, 0.15) / 0.15
        return pd.Series(raw, index=f.index) * (0.35 + 0.65 * extreme) * (0.6 + 0.4 * hurst_mr)

    def explain(self, r):
        return [f"Bollinger z {r.get('bb_z', 0):+.2f}, RSI3 {r.get('rsi3', np.nan):.0f}, "
                f"distance to VWAP {r.get('px_vs_vwap', 0):+.2f} ATR, Hurst {r.get('hurst', np.nan):.2f}"]


class SMCAnalyst(Analyst):
    """Market structure, order blocks, FVGs, premium/discount, OTE."""

    def __init__(self):
        super().__init__("smc", "اسمارت‌مانی / ساختار")

    def score(self, f):
        ms = _s(f, "ms_trend")
        rp = _s(f, "range_pos", 0.5).clip(-0.5, 1.5)
        lr = _s(f, "leg_retr", 0.0)
        ld = _s(f, "leg_dir", 0.0)
        ote = ((lr >= 0.618) & (lr <= 0.79)).astype(float)
        pd_bias = np.where(ms > 0, (0.5 - rp).clip(-0.5, 0.5), np.where(ms < 0, (0.5 - rp).clip(-0.5, 0.5), 0))
        s = (0.25 * ms
             + 0.30 * (_s(f, "choch_up_d") - _s(f, "choch_dn_d"))
             + 0.15 * (_s(f, "bos_up_d") - _s(f, "bos_dn_d"))
             + (0.35 + 0.15 * _s(f, "ob_strength")) * (_s(f, "in_bull_ob") - _s(f, "in_bear_ob"))
             + 0.20 * (_s(f, "in_bull_fvg") - _s(f, "in_bear_fvg"))
             + 0.30 * pd_bias
             + 0.25 * ote * ld)
        h4 = _s(f, "h4_ms_trend")
        s = s * (1 + 0.25 * h4 * np.sign(s))
        return pd.Series(s, index=f.index).clip(-1, 1)

    def explain(self, r):
        out = []
        ms = r.get("ms_trend", 0)
        out.append(f"H1 market structure {'bullish (HH/HL)' if ms > 0 else 'bearish (LH/LL)' if ms < 0 else 'undefined'}"
                   f"; dealing-range position {r.get('range_pos', np.nan):.0%} "
                   f"({'discount' if r.get('range_pos', .5) < .5 else 'premium'})")
        if r.get("choch_up_d", 0) > 0.2:
            out.append("recent bullish CHoCH (change of character)")
        if r.get("choch_dn_d", 0) > 0.2:
            out.append("recent bearish CHoCH (change of character)")
        if r.get("bos_up_d", 0) > 0.2:
            out.append("recent bullish BOS (continuation)")
        if r.get("bos_dn_d", 0) > 0.2:
            out.append("recent bearish BOS (continuation)")
        if r.get("in_bull_ob", 0):
            out.append("price is reacting inside a bullish order block")
        if r.get("in_bear_ob", 0):
            out.append("price is reacting inside a bearish order block")
        if r.get("in_bull_fvg", 0):
            out.append("price is inside a bullish fair value gap")
        if r.get("in_bear_fvg", 0):
            out.append("price is inside a bearish fair value gap")
        lr = r.get("leg_retr", np.nan)
        if 0.618 <= lr <= 0.79:
            out.append(f"in the OTE zone (retracement {lr:.0%} of last leg)")
        return out


class LiquidityAnalyst(Analyst):
    """Stop hunts (sweeps of swing/equal/session/day/week levels), SMT divergence."""

    def __init__(self):
        super().__init__("liquidity", "نقدینگی / استاپ‌هانت")

    def score(self, f):
        return (0.65 * _s(f, "sweep_score") + 0.45 * _s(f, "smt")).clip(-1, 1)

    def explain(self, r):
        out = []
        names = {"swing_low": "swing low", "eql": "equal lows", "pdl": "previous-day low",
                 "asia_low": "Asian-range low", "pwl": "previous-week low",
                 "swing_high": "swing high", "eqh": "equal highs", "pdh": "previous-day high",
                 "asia_high": "Asian-range high", "pwh": "previous-week high"}
        for k, v in names.items():
            val = r.get(f"sweep_{k}", 0)
            if val > 0:
                out.append(f"sell-side liquidity swept below {v} (bullish stop-hunt)")
            elif val < 0:
                out.append(f"buy-side liquidity swept above {v} (bearish stop-hunt)")
        if abs(r.get("sweep_score", 0)) > 0.1 and not out:
            out.append(f"recent liquidity sweep echo ({r.get('sweep_score', 0):+.2f})")
        if r.get("smt", 0) > 0.2:
            out.append("bullish SMT divergence vs correlated market")
        elif r.get("smt", 0) < -0.2:
            out.append("bearish SMT divergence vs correlated market")
        return out


class PatternAnalyst(Analyst):
    def __init__(self):
        super().__init__("patterns", "الگوها (کندل/کلاسیک/هارمونیک/الیوت)")

    def score(self, f):
        near_sup = ((_s(f, "dist_sup", 9) < 0.6) | (_s(f, "in_bull_ob") > 0) | (_s(f, "in_bull_fvg") > 0)
                    | (_s(f, "d_pdl", 9).abs() < 0.5)).astype(float)
        near_res = ((_s(f, "dist_res", 9) < 0.6) | (_s(f, "in_bear_ob") > 0) | (_s(f, "in_bear_fvg") > 0)
                    | (_s(f, "d_pdh", 9).abs() < 0.5)).astype(float)
        cs = _s(f, "candle_score")
        loc = np.where(cs > 0, 0.4 + 0.6 * near_sup, 0.4 + 0.6 * near_res)
        s = (0.45 * cs * loc + 0.45 * _s(f, "div_reg") + 0.25 * _s(f, "div_hid")
             + 0.45 * _s(f, "double_tb") + 0.55 * _s(f, "head_shoulders")
             + 0.45 * _s(f, "harmonic") + 0.35 * _s(f, "elliott"))
        return pd.Series(s, index=f.index).clip(-1, 1)

    def explain(self, r):
        out = []
        for k, label in [("bull_engulf", "bullish engulfing"), ("bear_engulf", "bearish engulfing"),
                         ("hammer", "hammer/pin bar"), ("shooting_star", "shooting star"),
                         ("morning_star", "morning star"), ("evening_star", "evening star"),
                         ("three_soldiers", "three white soldiers"), ("three_crows", "three black crows")]:
            if r.get(k, 0):
                out.append(f"candle: {label}")
        for k, pos, neg in [("div_reg", "bullish RSI divergence", "bearish RSI divergence"),
                            ("div_hid", "hidden bullish divergence", "hidden bearish divergence"),
                            ("double_tb", "double bottom neckline break", "double top neckline break"),
                            ("head_shoulders", "inverse head & shoulders break", "head & shoulders break")]:
            v = r.get(k, 0)
            if v > 0.15:
                out.append(pos)
            elif v < -0.15:
                out.append(neg)
        if abs(r.get("harmonic", 0)) > 0.15:
            out.append(f"harmonic pattern completed ({'bullish' if r['harmonic'] > 0 else 'bearish'})")
        if abs(r.get("elliott", 0)) > 0.15:
            out.append(f"Elliott count: {'bullish' if r['elliott'] > 0 else 'bearish'} implication")
        return out


class BreakoutAnalyst(Analyst):
    def __init__(self):
        super().__init__("breakout", "شکست / فشردگی نوسان")

    def score(self, f):
        c = _s(f, "close")
        rel = _s(f, "squeeze_release") * np.sign(c - _s(f, "bb_mid"))
        asia_brk = np.where((_s(f, "sess_london") > 0) & (_s(f, "d_asia_high", 0) > 0.1), 1.0,
                            np.where((_s(f, "sess_london") > 0) & (_s(f, "d_asia_low", 0) < -0.1), -1.0, 0.0))
        er_gate = (_s(f, "er", 0.2) > 0.3).astype(float)
        s = (0.40 * rel + 0.35 * _s(f, "don_break") * (0.5 + 0.5 * er_gate)
             + 0.20 * asia_brk + 0.25 * _s(f, "displacement"))
        return pd.Series(s, index=f.index).clip(-1, 1)

    def explain(self, r):
        out = []
        if r.get("squeeze", 0):
            out.append("volatility squeeze (Bollinger inside Keltner) - energy building")
        if r.get("squeeze_release", 0):
            out.append("squeeze just released")
        if r.get("don_break", 0) > 0:
            out.append("20-bar Donchian breakout up")
        elif r.get("don_break", 0) < 0:
            out.append("20-bar Donchian breakdown")
        if r.get("displacement", 0):
            out.append("displacement candle (institutional-size body)")
        return out


class LevelsAnalyst(Analyst):
    """Clustered S/R, prior-day value area, floor pivots."""

    def __init__(self):
        super().__init__("levels", "سطوح حمایت/مقاومت و پروفایل")

    def score(self, f):
        sup = np.exp(-_s(f, "dist_sup", 9).clip(0, None) / 0.5) * _t(_s(f, "sup_str"), 4)
        res = np.exp(-_s(f, "dist_res", 9).clip(0, None) / 0.5) * _t(_s(f, "res_str"), 4)
        va = -0.3 * _s(f, "va_pos")
        piv = 0.2 * (np.exp(-_s(f, "d_s1", 9).abs() / 0.4) - np.exp(-_s(f, "d_r1", 9).abs() / 0.4))
        return (0.6 * (sup - res) + va + piv).clip(-1, 1)

    def explain(self, r):
        out = []
        if np.isfinite(r.get("sup", np.nan)):
            out.append(f"nearest support {r['sup']:.5g} ({r.get('dist_sup', np.nan):.2f} ATR away, strength {r.get('sup_str', 0):.1f})")
        if np.isfinite(r.get("res", np.nan)):
            out.append(f"nearest resistance {r['res']:.5g} ({r.get('dist_res', np.nan):.2f} ATR away, strength {r.get('res_str', 0):.1f})")
        vp = r.get("va_pos", 0)
        out.append("price above prior-day value area" if vp > 0 else "price below prior-day value area" if vp < 0
                   else "price inside prior-day value area")
        return out


class MacroAnalyst(Analyst):
    def __init__(self, symbol: str):
        super().__init__("macro", "بین‌بازاری / کلان")
        self.symbol = symbol

    def score(self, f):
        if self.symbol == "XAUUSD":
            s = (-0.30 * _t(_s(f, "mac_real10y_chg20"), 1.5) - 0.15 * _t(_s(f, "mac_real10y_chg5"), 1.5)
                 - 0.20 * _t(_s(f, "mac_dxy_ret20"), 1.5) - 0.10 * _t(_s(f, "mac_dxy_ret5"), 1.5)
                 + 0.10 * _t(_s(f, "mac_silver_ret5"), 1.5) + 0.10 * _t(_s(f, "mac_vix_chg5"), 0.2)
                 + 0.15 * _t(_s(f, "peer_ret24"), 2.0))
        else:
            s = (-0.30 * _t(_s(f, "mac_dxy_ret20"), 1.5) - 0.15 * _t(_s(f, "mac_dxy_ret5"), 1.5)
                 - 0.25 * _t(_s(f, "mac_us2y_chg20"), 1.5) - 0.10 * _t(_s(f, "mac_us2y_chg5"), 1.5)
                 + 0.05 * _t(_s(f, "mac_spx_ret5"), 1.5) - 0.15 * _t(_s(f, "peer_ret24"), 2.0))
        return pd.Series(s, index=f.index).clip(-1, 1)

    def explain(self, r):
        if self.symbol == "XAUUSD":
            return [f"10y real yield 20d change {r.get('mac_real10y_chg20', 0):+.2f}σ (rising real yields weigh on gold)",
                    f"DXY 20d {r.get('mac_dxy_ret20', 0):+.2f}σ, silver 5d {r.get('mac_silver_ret5', 0):+.2f}σ, "
                    f"VIX z {r.get('mac_vix_z', 0):+.2f}"]
        return [f"DXY 20d {r.get('mac_dxy_ret20', 0):+.2f}σ, 5d {r.get('mac_dxy_ret5', 0):+.2f}σ",
                f"US 2y yield 20d change {r.get('mac_us2y_chg20', 0):+.2f}σ (rate-differential proxy)"]


class SeasonalityAnalyst(Analyst):
    def __init__(self):
        super().__init__("seasonality", "فصلی‌بودن ساعتی/روزانه")

    def score(self, f):
        return pd.Series(0.7 * _t(_s(f, "seas_t"), 2.5) + 0.3 * _t(_s(f, "dow_t"), 3.0), index=f.index)

    def explain(self, r):
        return [f"next-6h hour-of-day drift t-stat {r.get('seas_t', 0):+.2f}, day-of-week t {r.get('dow_t', 0):+.2f}"]


class PositioningAnalyst(Analyst):
    def __init__(self):
        super().__init__("positioning", "موقعیت‌گیری COT")

    def score(self, f):
        ci = _s(f, "cot_cot_index", 0.5)
        contra = np.where(ci > 0.85, -(ci - 0.85) / 0.15, np.where(ci < 0.15, (0.15 - ci) / 0.15, 0.0))
        flow = _t(_s(f, "cot_spec_net_chg4"), 0.05)
        return pd.Series(0.6 * contra + 0.4 * flow, index=f.index).clip(-1, 1)

    def explain(self, r):
        return [f"COT speculator index {r.get('cot_cot_index', np.nan):.0%} of 3y range, "
                f"4w flow {r.get('cot_spec_net_chg4', 0)*100:+.1f}% of OI"]


class MLAnalyst(Analyst):
    """Wraps out-of-sample probabilities from the LightGBM direction model."""

    def __init__(self):
        super().__init__("ml", "یادگیری ماشین (LightGBM)")

    def score(self, f):
        return _s(f, "ml_score")

    def explain(self, r):
        return [f"walk-forward ML: P(long wins) {r.get('ml_p_long', np.nan):.0%}, "
                f"P(short wins) {r.get('ml_p_short', np.nan):.0%}"]


def make_analysts(symbol: str) -> list[Analyst]:
    return [TrendAnalyst(), MomentumAnalyst(), MeanReversionAnalyst(), SMCAnalyst(),
            LiquidityAnalyst(), PatternAnalyst(), BreakoutAnalyst(), LevelsAnalyst(),
            MacroAnalyst(symbol), SeasonalityAnalyst(), PositioningAnalyst(), MLAnalyst()]


W_TREND = {"trend": 1.4, "momentum": 1.0, "mean_reversion": 0.2, "smc": 1.2, "liquidity": 0.6,
           "patterns": 0.6, "breakout": 1.0, "levels": 0.4, "macro": 0.7, "seasonality": 0.4,
           "positioning": 0.2, "ml": 1.3}
W_RANGE = {"trend": 0.4, "momentum": 0.4, "mean_reversion": 1.2, "smc": 0.9, "liquidity": 1.2,
           "patterns": 0.9, "breakout": 0.4, "levels": 1.1, "macro": 0.5, "seasonality": 0.5,
           "positioning": 0.2, "ml": 1.3}


def regime(f: pd.DataFrame) -> pd.DataFrame:
    adx_c = ((_s(f, "adx", 20) - 18) / 15).clip(0, 1)
    er_c = ((_s(f, "er", 0.3) - 0.15) / 0.30).clip(0, 1)
    hu_c = ((_s(f, "hurst", 0.5) - 0.42) / 0.16).clip(0, 1)
    htf = ((_s(f, "h4_ms_trend") == _s(f, "d1_ms_trend")) & (_s(f, "h4_ms_trend") != 0)).astype(float)
    chop_c = ((61.8 - _s(f, "chop", 50)) / 23.6).clip(0, 1)
    ts = (0.30 * adx_c + 0.20 * er_c + 0.15 * hu_c + 0.20 * htf + 0.15 * chop_c)
    out = pd.DataFrame(index=f.index)
    out["trend_strength"] = ts.ewm(span=6).mean()
    vol = _s(f, "atr_pct", 0.5)
    lab = np.where(out["trend_strength"] > 0.55, "trending",
                   np.where(out["trend_strength"] < 0.32, "ranging", "transition"))
    out["regime"] = lab
    out["vol_state"] = np.where(vol > 0.8, "high-vol", np.where(vol < 0.2, "low-vol", "normal-vol"))
    return out


def run_ensemble(f: pd.DataFrame, symbol: str, entry_q: float = 0.85, floor: float = 0.12,
                 analysts: list[Analyst] | None = None) -> pd.DataFrame:
    analysts = analysts or make_analysts(symbol)
    scores = pd.DataFrame({a.name: a.score(f) for a in analysts}, index=f.index).fillna(0.0)
    reg = regime(f)
    ts = reg["trend_strength"].fillna(0.4).values[:, None]
    names = list(scores.columns)
    wt = np.array([W_TREND[n] for n in names])[None, :]
    wr = np.array([W_RANGE[n] for n in names])[None, :]
    if "ml_score" not in f or f["ml_score"].abs().sum() == 0:
        ml_i = names.index("ml")
        wt = wt.copy()
        wr = wr.copy()
        wt[0, ml_i] = wr[0, ml_i] = 0.0
    W = ts * wt + (1 - ts) * wr
    S = scores.values
    comp = (W * S).sum(axis=1) / W.sum(axis=1)
    active = np.abs(S) > 0.15
    sign = np.sign(comp)[:, None]
    agree_w = (W * active * (np.sign(S) == sign)).sum(axis=1)
    act_w = (W * active).sum(axis=1)
    agreement = np.where(act_w > 0, agree_w / np.where(act_w > 0, act_w, 1), 0.0)

    out = scores.add_prefix("a_")
    out["composite"] = comp
    out["agreement"] = agreement
    out = pd.concat([out, reg], axis=1)
    thr = pd.Series(np.abs(comp), index=f.index).rolling(1500, min_periods=300).quantile(entry_q).shift(1)
    out["threshold"] = np.maximum(thr.fillna(np.inf if len(f) > 300 else floor), floor)
    out["confidence"] = (np.abs(comp) / out["threshold"]).clip(0, 2) / 2 * agreement
    return out


def explain_latest(f: pd.DataFrame, ens: pd.DataFrame, symbol: str) -> list[dict]:
    row = f.iloc[-1]
    e = ens.iloc[-1]
    out = []
    for a in make_analysts(symbol):
        out.append({"name": a.name, "title_fa": a.title_fa, "score": float(e[f"a_{a.name}"]),
                    "reasons": a.explain(row)})
    return out
