"""Build M5/H1 datasets from HistData M1 and verify the UTC alignment against Yahoo."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from aitrader.data.histdata import fetch, resample
from aitrader.data.loader import fetch_yahoo
from aitrader.config import CACHE_DIR

out = {}
for sym, yahoo in (("EURUSD", "EURUSD=X"), ("XAUUSD", "GC=F")):
    m1 = fetch(sym, 2010, None, verbose=False)
    m5, h1 = resample(m1, "5min"), resample(m1, "1h")
    m5.to_pickle(CACHE_DIR / f"hd_{sym}_M5.pkl"); h1.to_pickle(CACHE_DIR / f"hd_{sym}_H1.pkl")
    y = fetch_yahoo(yahoo, "1h")
    common = h1.index.intersection(y.index)
    r1, r2 = np.log(h1["close"]).diff(), np.log(y["close"]).diff()
    best = {lag: round(r1.reindex(common).corr(r2.shift(lag).reindex(common)), 3) for lag in (-2, -1, 0, 1, 2)}
    print(f"{sym}: M1 {len(m1):,} bars {m1.index[0]} -> {m1.index[-1]} | H1 {len(h1):,} | corr with Yahoo by lag {best}")
