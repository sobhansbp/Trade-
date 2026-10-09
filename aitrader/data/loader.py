"""Market, macro, positioning and calendar data with on-disk caching.

All OHLC frames are returned with a UTC ``DatetimeIndex`` (bar *open* time) and
lower-case ``open, high, low, close, volume`` columns.
"""
from __future__ import annotations

import io
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import CACHE_DIR, FRED_SERIES, INTERMARKET_YAHOO, Instrument

warnings.filterwarnings("ignore", category=FutureWarning)

_YF_PERIOD = {"15m": "60d", "1h": "730d", "1d": "max", "1wk": "max"}


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return CACHE_DIR / f"{safe}.csv"


def _fresh(path: Path, max_age_s: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_s


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    if "adj close" in df.columns:
        df = df.drop(columns=["adj close"])
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"missing column {c}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx.as_unit("ns")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    # repair occasional bad prints: high/low must bracket open/close
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    df = df[(df["high"] - df["low"]) >= 0]
    return df


def fetch_yahoo(ticker: str, interval: str = "1h", period: str | None = None,
                max_age_s: float = 1800, use_cache: bool = True) -> pd.DataFrame:
    period = period or _YF_PERIOD.get(interval, "730d")
    path = _cache_path(f"yf_{ticker}_{interval}_{period}")
    if use_cache and _fresh(path, max_age_s):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return normalize_ohlc(df)
    import yfinance as yf

    last_err = None
    for attempt in range(4):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=False, threads=False)
            if df is not None and len(df):
                df = normalize_ohlc(df)
                df.to_csv(path)
                return df
        except Exception as e:  # network hiccup / rate limit
            last_err = e
        time.sleep(2 ** attempt)
    if path.exists():  # stale cache beats nothing
        return normalize_ohlc(pd.read_csv(path, index_col=0, parse_dates=True))
    raise RuntimeError(f"could not download {ticker} {interval}: {last_err}")


def load_csv(path: str) -> pd.DataFrame:
    """Load a broker export (MT4/MT5/TradingView). Detects common layouts."""
    raw = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower().strip("<> "): c for c in raw.columns}
    if "date" in cols and "time" in cols:
        ts = pd.to_datetime(raw[cols["date"]].astype(str) + " " + raw[cols["time"]].astype(str))
    else:
        key = next(k for k in ("datetime", "time", "date", "timestamp") if k in cols)
        col = raw[cols[key]]
        ts = pd.to_datetime(col, unit="s") if np.issubdtype(col.dtype, np.number) else pd.to_datetime(col)
    out = pd.DataFrame(index=pd.DatetimeIndex(ts))
    for name in ("open", "high", "low", "close"):
        out[name] = raw[cols[name]].values
    vol_key = next((k for k in ("tickvol", "volume", "vol") if k in cols), None)
    out["volume"] = raw[cols[vol_key]].values if vol_key else 0.0
    return normalize_ohlc(out)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open"])


def load_instrument(inst: Instrument, interval: str = "1h", csv: str | None = None,
                    use_cache: bool = True) -> pd.DataFrame:
    if csv:
        df = load_csv(csv)
        if interval == "1h" and len(df) > 2 and (df.index[1] - df.index[0]) < pd.Timedelta("1h"):
            df = resample_ohlc(df, "1h")
        return df
    return fetch_yahoo(inst.yahoo, interval, use_cache=use_cache)


# --------------------------------------------------------------------- macro

def fetch_fred(series: str, max_age_s: float = 6 * 3600) -> pd.Series:
    path = _cache_path(f"fred_{series}")
    if _fresh(path, max_age_s):
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True).iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.to_frame(series).to_csv(path)
    idx = pd.DatetimeIndex(s.index)
    s.index = (idx.tz_localize("UTC") if idx.tz is None else idx).as_unit("ns")
    s.name = series
    return s


def load_macro_daily() -> pd.DataFrame:
    """Daily intermarket closes + FRED rates. Missing sources are skipped."""
    frames = {}
    for name, ticker in INTERMARKET_YAHOO.items():
        try:
            frames[name] = fetch_yahoo(ticker, "1d", "max", max_age_s=6 * 3600)["close"]
        except Exception:
            pass
    for name, sid in FRED_SERIES.items():
        try:
            frames[name] = fetch_fred(sid)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    out = pd.DataFrame({k: v for k, v in frames.items()})
    out.index = out.index.normalize()
    out = out.groupby(level=0).last().sort_index().ffill()
    return out


# --------------------------------------------------------------- positioning

def fetch_cot(code: str, max_age_s: float = 24 * 3600) -> pd.DataFrame:
    """CFTC legacy futures-only report for one contract.

    Returned index is the *release* date (report Tuesday + 3 days) so the data
    is only used once it was actually public.
    """
    path = _cache_path(f"cot_{code}")
    if _fresh(path, max_age_s):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        params = {
            "$where": f"cftc_contract_market_code='{code}'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": "600",
            "$select": "report_date_as_yyyy_mm_dd,open_interest_all,"
                       "noncomm_positions_long_all,noncomm_positions_short_all,"
                       "comm_positions_long_all,comm_positions_short_all",
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df.pop("report_date_as_yyyy_mm_dd"))
        df = df.set_index("date").astype(float).sort_index()
        df.to_csv(path)
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = (df.index + pd.Timedelta(days=3, hours=21)).as_unit("ns")  # Friday 15:30 ET release
    spec_net = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    comm_net = df["comm_positions_long_all"] - df["comm_positions_short_all"]
    out = pd.DataFrame({"spec_net": spec_net, "comm_net": comm_net,
                        "oi": df["open_interest_all"]})
    win = 156  # ~3 years of weeks
    lo = out["spec_net"].rolling(win, min_periods=52).min()
    hi = out["spec_net"].rolling(win, min_periods=52).max()
    out["cot_index"] = (out["spec_net"] - lo) / (hi - lo).replace(0, np.nan)
    out["spec_net_chg4"] = out["spec_net"].diff(4) / out["oi"]
    return out


# ------------------------------------------------------------------ calendar

def fetch_calendar(max_age_s: float = 3600) -> pd.DataFrame:
    """This week's economic calendar (ForexFactory public feed)."""
    path = _cache_path("ff_calendar_thisweek")
    if _fresh(path, max_age_s):
        df = pd.read_csv(path)
    else:
        try:
            r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=20)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            df.to_csv(path, index=False)
        except Exception:
            return pd.DataFrame(columns=["title", "country", "date", "impact", "forecast", "previous"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def high_impact_events(currencies: tuple, start: pd.Timestamp, hours_ahead: int = 48,
                       cal: pd.DataFrame | None = None) -> pd.DataFrame:
    cal = fetch_calendar() if cal is None else cal
    if cal.empty:
        return cal
    end = start + pd.Timedelta(hours=hours_ahead)
    m = cal["country"].isin(currencies) & cal["impact"].eq("High")
    m &= (cal["date"] >= start - pd.Timedelta(hours=2)) & (cal["date"] <= end)
    return cal.loc[m].sort_values("date")
