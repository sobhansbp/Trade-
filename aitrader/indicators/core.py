"""Classic technical indicators. Every function is causal (uses only past bars)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                     axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(s: pd.Series, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def stochastic(df: pd.DataFrame, k=14, d=3, smooth=3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    raw = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    kline = raw.rolling(smooth).mean()
    return kline, kline.rolling(d).mean()


def williams_r(df: pd.DataFrame, n=14) -> pd.Series:
    hi = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    return -100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)


def cci(df: pd.DataFrame, n=20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def adx(df: pd.DataFrame, n=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(df), n)
    pdi = 100 * rma(pd.Series(plus_dm, index=df.index), n) / tr
    mdi = 100 * rma(pd.Series(minus_dm, index=df.index), n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx.fillna(0), n), pdi, mdi


def bollinger(s: pd.Series, n=20, k=2.0):
    m = sma(s, n)
    sd = s.rolling(n).std(ddof=0)
    return m, m + k * sd, m - k * sd


def keltner(df: pd.DataFrame, n=20, k=1.5):
    m = ema(df["close"], n)
    a = atr(df, n)
    return m, m + k * a, m - k * a


def donchian(df: pd.DataFrame, n=20):
    return df["high"].rolling(n).max(), df["low"].rolling(n).min()


def supertrend(df: pd.DataFrame, n=10, mult=3.0):
    a = atr(df, n).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    close = df["close"].values
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    fu, fl = upper.copy(), lower.copy()
    direction = np.ones(len(df))
    line = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        if np.isnan(a[i]):
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < fl[i] else 1
        else:
            direction[i] = 1 if close[i] > fu[i] else -1
        line[i] = fl[i] if direction[i] == 1 else fu[i]
    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)


def ichimoku(df: pd.DataFrame, t=9, k=26, s=52):
    """Cloud values *as visible at bar t* (span A/B are plotted k bars ahead,
    so the cloud at t was computed from data at t-k)."""
    tenkan = (df["high"].rolling(t).max() + df["low"].rolling(t).min()) / 2
    kijun = (df["high"].rolling(k).max() + df["low"].rolling(k).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(k)
    span_b = ((df["high"].rolling(s).max() + df["low"].rolling(s).min()) / 2).shift(k)
    return tenkan, kijun, span_a, span_b


def psar(df: pd.DataFrame, step=0.02, max_step=0.2) -> pd.Series:
    high, low = df["high"].values, df["low"].values
    out = np.full(len(df), np.nan)
    if len(df) < 3:
        return pd.Series(out, index=df.index)
    bull, af, ep, sar = True, step, high[0], low[0]
    for i in range(1, len(df)):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < sar:
                bull, sar, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_step)
        else:
            sar = max(sar, high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > sar:
                bull, sar, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_step)
        out[i] = sar
    return pd.Series(out, index=df.index)


def efficiency_ratio(s: pd.Series, n=20) -> pd.Series:
    change = (s - s.shift(n)).abs()
    vol = s.diff().abs().rolling(n).sum()
    return change / vol.replace(0, np.nan)


def kama(s: pd.Series, n=10, fast=2, slow=30) -> pd.Series:
    er = efficiency_ratio(s, n).fillna(0).values
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    v = s.values
    out = np.full(len(s), np.nan)
    if len(s) > n:
        out[n] = v[n]
        for i in range(n + 1, len(s)):
            out[i] = out[i - 1] + sc[i] * (v[i] - out[i - 1])
    return pd.Series(out, index=s.index)


def choppiness(df: pd.DataFrame, n=14) -> pd.Series:
    tr_sum = true_range(df).rolling(n).sum()
    rng = df["high"].rolling(n).max() - df["low"].rolling(n).min()
    return 100 * np.log10(tr_sum / rng.replace(0, np.nan)) / np.log10(n)


def hurst(s: pd.Series, n=128, lags=(2, 4, 8, 16, 32)) -> pd.Series:
    """Rolling Hurst exponent from the scaling of lagged differences of log price.
    H>0.5 trending/persistent, H<0.5 mean reverting."""
    lp = np.log(s)
    lags = np.array(lags)
    stds = []
    for L in lags:
        stds.append((lp - lp.shift(int(L))).rolling(n, min_periods=n // 2).std())
    m = np.log(np.vstack([x.values for x in stds]).T)
    x = np.log(lags)
    xc = x - x.mean()
    slope = (m - m.mean(axis=1, keepdims=True)) @ xc / (xc ** 2).sum()
    return pd.Series(slope, index=s.index)


def linreg_slope(s: pd.Series, n=20) -> pd.Series:
    x = np.arange(n)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    return s.rolling(n).apply(lambda y: (xc * (y - y.mean())).sum() / denom, raw=True)


def zscore(s: pd.Series, n=50) -> pd.Series:
    return (s - s.rolling(n).mean()) / s.rolling(n).std().replace(0, np.nan)


def pct_rank(s: pd.Series, n=500) -> pd.Series:
    """Percentile of the current value within its trailing window (0..1)."""
    return s.rolling(n, min_periods=n // 4).rank(pct=True)


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def mfi(df: pd.DataFrame, n=14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP anchored at each UTC day; falls back to TWAP when volume is absent."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"] if df["volume"].sum() > 0 else pd.Series(1.0, index=df.index)
    vol = vol.replace(0, np.nan).fillna(vol[vol > 0].median() if (vol > 0).any() else 1.0)
    day = df.index.floor("D")
    return (tp * vol).groupby(day).cumsum() / vol.groupby(day).cumsum()
