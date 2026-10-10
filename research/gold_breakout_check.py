"""Are gold breakout profits just the 2019-26 bull trend? Split by side and add a daily-bias filter."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "."); sys.path.insert(0, "research")
from round3_tests import Px, trading_days, local_ts, COST, IS_END
from aitrader.config import CACHE_DIR
from aitrader.strategic import daily_bias, load_daily
from aitrader.data.loader import load_macro_daily
from scipy import stats

sym = "XAUUSD"; px = Px(pd.read_pickle(CACHE_DIR / f"hd_{sym}_M5.pkl")); days = trading_days()
bias = daily_bias("XAUUSD", load_daily("XAUUSD", "GC=F"), load_macro_daily())["bias"]
LON, ET = "Europe/London", "America/New_York"

def london_breakout(days):
    r0, r1 = local_ts(days, "00:00", LON), local_ts(days, "07:00", LON)
    tu, ex = local_ts(days, "11:00", LON), local_ts(days, "16:00", LON)
    out = []
    for d, a, b, u, e in zip(days, r0, r1, tu, ex):
        s = px.span(a, b)
        if s.stop - s.start < 6: continue
        hi, lo = px.h[s].max(), px.l[s].min()
        w = px.span(b, e)
        if w.stop - w.start < 3: continue
        side = 0
        for i in range(w.start, w.stop):
            if side == 0:
                if px.t[i] >= u.value: break
                up, dn = px.h[i] > hi, px.l[i] < lo
                if up and dn: break
                if up: side, entry, stop = 1, hi, lo
                elif dn: side, entry, stop = -1, lo, hi
                if side: t_e = px.t[i]; res = None
                continue
            if (side > 0 and px.l[i] <= stop) or (side < 0 and px.h[i] >= stop):
                res = stop; break
        if side == 0: continue
        res = px.c[w.stop - 1] if res is None else res
        out.append((d, pd.Timestamp(t_e, tz="UTC"), side, side * (res - entry) / entry - COST[sym][0] / entry, side * (res - entry) / entry))
    return pd.DataFrame(out, columns=["day", "t", "side", "net", "gross"])

tr = london_breakout(days)
b = bias.reindex(pd.DatetimeIndex(tr["day"]).normalize()).values
tr["bias"] = b
def rep(df, name):
    for per, m in (("IS 2010-18", df.t < IS_END), ("OOS 2019-26", df.t >= IS_END)):
        x = df[m]
        if len(x) < 20: continue
        t = stats.ttest_1samp(x.net, 0)[0]
        print(f"  {name:38s} {per}: n={len(x):5d} gross {x.gross.mean()*1e4:+.2f}bp net {x.net.mean()*1e4:+.2f}bp t={t:+.2f}")
print("Gold London breakout of Asian range")
rep(tr[tr.side > 0], "longs only")
rep(tr[tr.side < 0], "shorts only")
rep(tr[np.sign(tr.bias) == tr.side], "with daily bias (pre-specified filter)")
rep(tr[(np.sign(tr.bias) != tr.side)], "against daily bias")
