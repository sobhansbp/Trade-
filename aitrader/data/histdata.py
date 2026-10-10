"""Free long-history 1-minute data from HistData.com (EURUSD since 2000, XAUUSD since 2009).

HistData documents its timestamps as EST without daylight saving, but checking
US data releases (NFP/CPI at 08:30 New York) shows the clock is really
London time minus 5 hours: UTC-5 while Europe is on standard time and UTC-4
during European summer time. They are converted to UTC accordingly (DST
switches happen on weekends while FX is closed, so there is no ambiguity). Files are cached under data_cache/histdata/. Downloads
are throttled to be polite to the free service.

    python -m aitrader.data.histdata EURUSD 2010 2026
"""
from __future__ import annotations

import io
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from ..config import CACHE_DIR

BASE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{period}"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) aitrader-research"}
DIR = CACHE_DIR / "histdata"


def _download(pair: str, year: int, month: int | None = None) -> bytes | None:
    period = f"{year}" if month is None else f"{year}/{month}"
    url = BASE.format(pair=pair.lower(), period=period)
    s = requests.Session()
    r = s.get(url, headers=UA, timeout=60)
    if r.status_code != 200:
        return None
    m = re.search(r'id="tk" value="([0-9a-f]+)"', r.text)
    if not m:
        return None
    data = {"tk": m.group(1), "date": str(year), "datemonth": f"{year}{month:02d}" if month else str(year),
            "platform": "ASCII", "timeframe": "M1", "fxpair": pair.upper()}
    z = s.post("https://www.histdata.com/get.php", data=data, headers={**UA, "Referer": url}, timeout=180)
    if z.status_code != 200 or z.content[:2] != b"PK":
        return None
    return z.content


def _parse(zbytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(name)
    df = pd.read_csv(io.BytesIO(raw), sep=";", header=None,
                     names=["ts", "open", "high", "low", "close", "volume"])
    london = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)
    out = df[["open", "high", "low", "close"]].astype(float)
    out.index = london_to_utc(pd.DatetimeIndex(london))
    out = out[out.index.notna()]
    out["volume"] = 0.0
    return out[~out.index.duplicated()].sort_index()


def london_to_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive London wall-clock times -> UTC."""
    loc = idx.tz_localize("Europe/London", ambiguous="NaT", nonexistent="shift_forward")
    return loc.tz_convert("UTC").as_unit("ns")


CACHE_VERSION = 2   # v1 pickles used a fixed +5h offset


def fetch(pair: str, start_year: int = 2010, end_year: int | None = None, pause: float = 2.0,
          verbose: bool = True) -> pd.DataFrame:
    """Return all cached/downloaded M1 bars for ``pair`` (UTC)."""
    DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    end_year = end_year or now.year
    frames = []
    for y in range(start_year, end_year + 1):
        periods = [None] if y < now.year else list(range(1, now.month))   # current year: monthly files
        for mth in periods:
            tag = f"{pair.upper()}_{y}" + (f"{mth:02d}" if mth else "")
            path = DIR / f"{tag}.v{CACHE_VERSION}.pkl"
            legacy = DIR / f"{tag}.pkl"
            if not path.exists() and legacy.exists():
                # v1 index = HistData time + 5h labelled UTC = London wall clock
                old = pd.read_pickle(legacy)
                old.index = london_to_utc(old.index.tz_localize(None))
                old = old[old.index.notna()]
                old.to_pickle(path)
                legacy.unlink()
            if path.exists():
                frames.append(pd.read_pickle(path))
                continue
            zb = _download(pair, y, mth)
            if zb is None:
                if verbose:
                    print(f"  {tag}: not available")
                continue
            df = _parse(zb)
            df.to_pickle(path)
            frames.append(df)
            if verbose:
                print(f"  {tag}: {len(df):,} bars {df.index[0]} -> {df.index[-1]}")
            time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated()]


def resample(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return m1.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])


if __name__ == "__main__":  # pragma: no cover
    pair = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    a = int(sys.argv[2]) if len(sys.argv) > 2 else 2010
    b = int(sys.argv[3]) if len(sys.argv) > 3 else None
    df = fetch(pair, a, b)
    print(pair, len(df), df.index.min(), df.index.max())
