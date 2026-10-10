"""Zero-shot test of time-series foundation models (Amazon Chronos-Bolt, Chronos-2) on
EURUSD and gold, only on data released AFTER each model (no training-set leakage).

Signal: sign(median one-step forecast - last close). Compared with the base rate
(always predicting the majority direction) and a naive momentum rule.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, ".")
from aitrader.config import CACHE_DIR  # noqa: E402

torch.set_num_threads(4)
MODELS = {"amazon/chronos-bolt-small": "2025-01-01", "amazon/chronos-2": "2026-01-01"}
COST_BP = {"EURUSD": 1.1, "XAUUSD": 1.5}     # retail round trip, bp, charged when the position flips


def daily_closes(sym):
    m5 = pd.read_pickle(CACHE_DIR / f"hd_{sym}_M5.pkl")
    et = m5.index.tz_convert("America/New_York")
    c = m5["close"][(et.hour == 16) & (et.minute == 45)]          # 16:45-16:50 ET bar close ~ NY close
    c.index = c.index.tz_convert("America/New_York").normalize().tz_localize(None)
    return c[~c.index.duplicated(keep="last")]


def hourly_closes(sym):
    h1 = pd.read_pickle(CACHE_DIR / f"hd_{sym}_H1.pkl")["close"]
    return h1


def evaluate(pipe, series: pd.Series, start: str, ctx: int = 512, max_n: int | None = None, batch: int = 64):
    vals = series.values.astype(np.float32)
    idx = np.where(series.index >= pd.Timestamp(start, tz=series.index.tz))[0]
    idx = idx[(idx >= ctx) & (idx < len(vals) - 1)]
    if max_n:
        idx = idx[-max_n:]
    preds = np.empty(len(idx))
    for k in range(0, len(idx), batch):
        chunk = idx[k:k + batch]
        contexts = [torch.tensor(vals[i - ctx + 1:i + 1]) for i in chunk]
        q, _ = pipe.predict_quantiles(contexts, prediction_length=1, quantile_levels=[0.1, 0.5, 0.9])
        if isinstance(q, list):     # Chronos-2: one (variates, horizon, quantiles) tensor per series
            preds[k:k + len(chunk)] = [float(qi[0, 0, 1]) for qi in q]
        else:                       # Chronos-Bolt: (series, horizon, quantiles)
            preds[k:k + len(chunk)] = q[:, 0, 1].numpy()
    last = vals[idx]
    nxt = vals[idx + 1]
    ret = np.log(nxt / last)
    sig = np.sign(preds - last)
    prev_ret = np.log(vals[idx] / vals[idx - 1])
    return pd.DataFrame({"ret": ret, "sig": sig, "mom": np.sign(prev_ret)}, index=series.index[idx])


def score(df: pd.DataFrame, cost_bp: float, periods_per_year: float) -> dict:
    hit = (np.sign(df.ret) == df.sig)[df.sig != 0].mean()
    base = max((df.ret > 0).mean(), (df.ret < 0).mean())
    mom_hit = (np.sign(df.ret) == df.mom)[df.mom != 0].mean()
    flips = (df.sig.diff().abs() > 0).astype(float)
    pnl = df.sig * df.ret - flips * cost_bp / 1e4
    sh = pnl.mean() / pnl.std() * np.sqrt(periods_per_year) if pnl.std() > 0 else np.nan
    gross = df.sig * df.ret
    t = stats.ttest_1samp(pnl, 0)[0]
    p_hit = stats.binomtest(int((np.sign(df.ret) == df.sig)[df.sig != 0].sum()), int((df.sig != 0).sum()), 0.5).pvalue
    return {"n": len(df), "hit_rate": round(float(hit), 4), "base_rate": round(float(base), 4),
            "naive_momentum_hit": round(float(mom_hit), 4), "p_hit_vs_50pct": round(float(p_hit), 4),
            "gross_ann": round(float(gross.mean() * periods_per_year), 4),
            "net_ann": round(float(pnl.mean() * periods_per_year), 4), "net_sharpe": round(float(sh), 2),
            "t_net": round(float(t), 2), "flip_rate": round(float(flips.mean()), 3)}


if __name__ == "__main__":
    from pathlib import Path
    from chronos import BaseChronosPipeline
    res_path = Path("research/results/foundation_models.json")
    out = json.loads(res_path.read_text()) if res_path.exists() else {}
    todo = {m: MODELS[m] for m in sys.argv[1:]} if len(sys.argv) > 1 else MODELS
    for model, start in todo.items():
        t0 = time.time()
        pipe = BaseChronosPipeline.from_pretrained(model, device_map="cpu", torch_dtype=torch.float32)
        print(f"loaded {model} in {time.time() - t0:.0f}s", flush=True)
        for sym in ("EURUSD", "XAUUSD"):
            d = evaluate(pipe, daily_closes(sym), start)
            s = score(d, COST_BP[sym], 252)
            out[f"{model}|{sym}|daily"] = s
            print(f"  {sym} daily  from {start}: {s}", flush=True)
            h = evaluate(pipe, hourly_closes(sym), start, max_n=3000)
            sh = score(h, COST_BP[sym], 24 * 260)
            out[f"{model}|{sym}|hourly"] = sh
            print(f"  {sym} hourly (last {len(h)} h): {sh}", flush=True)
            res_path.write_text(json.dumps(out, indent=1))
