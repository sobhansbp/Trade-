"""Strategic (daily) bias layer - the part of the system with the strongest
evidence behind it.

Research summary (see docs/RESEARCH.md): on 2005-2023 daily data, simple
medium-term trend signals and a rate-differential signal were positive in all
three sub-periods for EURUSD; for gold, medium-term trend plus the real-yield
trend and a structural long tilt. Short-term daily mean reversion lost money on
both. Components are blended with *equal* weights (no optimisation).

Every component for date D uses information available before D's session:
prices through D-1 and macro series lagged an extra day (FRED publication lag).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import loader
from .indicators import core as ta


def _naive_daily(s: pd.Series) -> pd.Series:
    s = s.copy()
    idx = pd.DatetimeIndex(s.index)
    s.index = (idx.tz_convert(None) if idx.tz is not None else idx).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _donchian_state(c: pd.Series, n: int) -> pd.Series:
    hi, lo = c.rolling(n).max(), c.rolling(n).min()
    st = pd.Series(np.nan, index=c.index)
    st[c >= hi] = 1.0
    st[c <= lo] = -1.0
    return st.ffill().fillna(0.0)


COMPONENTS = {
    "EURUSD": ["tsmom_ens", "ema20_100", "rates_2y_inv", "dxy_tsmom20_inv"],
    "XAUUSD": ["tsmom_ens", "ema50_200", "real_yield_inv", "structural_long"],
}

LABELS_FA = {
    "tsmom_ens": "مومنتوم سری زمانی (۲۰/۶۰/۱۲۰/۲۵۰ روز)",
    "ema20_100": "تقاطع EMA20/100 روزانه",
    "ema50_200": "تقاطع EMA50/200 روزانه",
    "rates_2y_inv": "اختلاف نرخ بهره (روند بازده ۲ساله آمریکا، معکوس)",
    "dxy_tsmom20_inv": "روند ۲۰روزه شاخص دلار (معکوس)",
    "real_yield_inv": "روند بازده واقعی ۱۰ساله (معکوس)",
    "structural_long": "تمایل ساختاری خرید طلا (خرید بانک‌های مرکزی)",
}


def daily_bias(symbol: str, close: pd.Series, macro: pd.DataFrame | None = None,
               target_vol: float = 0.10, long_tilt: float = 1.0) -> pd.DataFrame:
    """Daily bias in [-1, 1] plus a volatility-target leverage, indexed by the
    date on which it may be *used*."""
    c = _naive_daily(close).dropna()
    lc = np.log(c)
    comp = pd.DataFrame(index=c.index)
    comp["tsmom_ens"] = sum(np.sign(lc.diff(L)) for L in (20, 60, 120, 250)) / 4
    comp["ema20_100"] = np.sign(ta.ema(c, 20) - ta.ema(c, 100))
    comp["ema50_200"] = np.sign(ta.ema(c, 50) - ta.ema(c, 200))
    comp["donchian55"] = _donchian_state(c, 55)
    comp["structural_long"] = long_tilt

    if macro is not None and not macro.empty:
        m = macro.copy()
        m.index = pd.DatetimeIndex(m.index).tz_convert(None) if m.index.tz is not None else m.index
        m = m[~m.index.duplicated(keep="last")].reindex(c.index.union(m.index)).ffill().reindex(c.index)
        if "us2y" in m:
            comp["rates_2y_inv"] = -np.sign(m["us2y"].shift(1).diff(60))
        if "dxy" in m:
            # extra day: Yahoo FX daily bars are stamped at 00:00 UTC of the *next* day
            comp["dxy_tsmom20_inv"] = -np.sign(np.log(m["dxy"].shift(1)).diff(20))
        if "real10y" in m:
            comp["real_yield_inv"] = -np.sign(m["real10y"].shift(1).diff(20))
    use = [k for k in COMPONENTS[symbol] if k in comp]
    comp = comp.fillna(0.0)
    comp["bias"] = comp[use].mean(axis=1)
    r = c.pct_change()
    vol = r.ewm(span=60, min_periods=20).std() * np.sqrt(252)
    comp["leverage"] = (target_vol / vol).clip(0, 4)
    comp["realized_vol"] = vol
    comp["close"] = c
    # usable from the next day onwards
    out = comp.shift(1)
    out.attrs["components"] = use
    return out


def load_daily(symbol: str, yahoo: str) -> pd.Series:
    return loader.fetch_yahoo(yahoo, "1d", "max", max_age_s=3 * 3600)["close"]


def attach_bias(f_index: pd.DatetimeIndex, bias: pd.DataFrame) -> pd.DataFrame:
    """Map daily bias onto intraday bars by UTC date."""
    day = pd.DatetimeIndex(f_index).tz_convert(None).normalize()
    b = bias.reindex(bias.index.union(day.unique())).ffill()
    out = b.reindex(day)
    out.index = f_index
    return out.add_prefix("sb_")
