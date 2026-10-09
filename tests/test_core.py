import numpy as np
import pandas as pd
import pytest

from aitrader.ai.claude_desk import enforce_guardrails, offline_brief
from aitrader.analysts.ensemble import run_ensemble
from aitrader.backtest.engine import performance, run_backtest, signal_mask
from aitrader.config import INSTRUMENTS, StrategyConfig
from aitrader.features import build_features, ml_columns
from aitrader.indicators import core as ta
from aitrader.indicators.structure import swing_points
from aitrader.risk.manager import simulate_trade, stop_distances
from aitrader.strategic import attach_bias, daily_bias


def synthetic(n=1500, seed=1, start="2025-01-06", vol=0.0012, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n * 2, freq="1h", tz="UTC")
    idx = idx[idx.dayofweek < 5][:n]
    r = rng.normal(drift, vol, n)
    close = 1.10 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(100, 1000, n).astype(float)}, index=idx)


@pytest.fixture(scope="module")
def df():
    return synthetic()


def test_no_lookahead_in_features(df):
    inst = INSTRUMENTS["EURUSD"]
    full = build_features(df, inst)
    cut = 1100
    part = build_features(df.iloc[:cut], inst)
    cols = ml_columns(full)
    a = full[cols].iloc[:cut - 30]
    b = part[cols].iloc[:cut - 30]
    diff = (a - b).abs()
    bad = diff.columns[(diff > 1e-9).any()]
    assert list(bad) == [], f"features changed when future bars were added: {list(bad)}"


def test_swings_are_confirmed_after_right_bars(df):
    sw = swing_points(df, 5, 5)
    t = sw["ph"].dropna().index[0]
    pos = int(sw.loc[t, "ph_pos"])
    assert df.index.get_loc(t) - pos == 5
    assert df["high"].iloc[pos] == sw.loc[t, "ph"]


def test_indicators_ranges(df):
    r = ta.rsi(df["close"])
    assert r.between(0, 100).all()
    h = ta.hurst(df["close"]).dropna()
    assert h.between(-0.5, 1.5).all()


def test_simulate_trade_target_and_stop_priority():
    inst = INSTRUMENTS["EURUSD"]
    cfg = StrategyConfig(partial_fraction=0.0)
    o = np.array([1.0, 1.0, 1.0, 1.0])
    h = np.array([1.0, 1.002, 1.030, 1.0])
    l = np.array([1.0, 0.999, 0.999, 1.0])
    c = np.array([1.0, 1.001, 1.02, 1.0])
    tr = simulate_trade(o, h, l, c, 0, 1, 0.004, inst, cfg)
    assert tr.reason == "target" and tr.r_multiple > 2.3
    # stop and target inside the same bar -> stop wins (conservative)
    h2 = np.array([1.0, 1.03, 1.0, 1.0])
    l2 = np.array([1.0, 0.99, 1.0, 1.0])
    tr2 = simulate_trade(o, h2, l2, c, 0, 1, 0.004, inst, cfg)
    assert tr2.reason == "stop" and tr2.r_multiple < -1.0


def test_partial_then_breakeven():
    inst = INSTRUMENTS["EURUSD"]
    cfg = StrategyConfig()
    o = np.array([1.0, 1.0, 1.003, 1.0])
    h = np.array([1.0, 1.0045, 1.003, 1.0])
    l = np.array([1.0, 0.9995, 0.999, 1.0])
    c = np.array([1.0, 1.004, 1.0, 1.0])
    tr = simulate_trade(o, h, l, c, 0, 1, 0.004, inst, cfg)
    assert tr.partial and tr.reason == "breakeven"
    assert 0.3 < tr.r_multiple < 0.6


def test_ensemble_and_signals(df):
    inst = INSTRUMENTS["EURUSD"]
    f = build_features(df, inst)
    bias = daily_bias("EURUSD", df["close"].resample("1D").last().dropna())
    f = f.join(attach_bias(f.index, bias.drop(columns=["close"])))
    ens = run_ensemble(f, "EURUSD")
    a_cols = [c for c in ens.columns if c.startswith("a_")]
    assert ens[a_cols].abs().max().max() <= 1.0 + 1e-9
    assert ens["composite"].abs().max() <= 1.0
    cfg = StrategyConfig()
    sig = signal_mask(f, ens, cfg)
    assert set(np.unique(sig)) <= {-1.0, 0.0, 1.0}
    # never trade against the daily bias
    side = np.sign(f["sb_bias"].fillna(0))
    assert ((sig != 0) & (sig != side)).sum() == 0
    stops = stop_distances(f, cfg)
    res = run_backtest(f, sig, stops, inst, cfg, start=300)
    assert "stats" in res


def test_daily_bias_uses_only_past():
    idx = pd.date_range("2015-01-01", periods=900, freq="B")
    rng = np.random.default_rng(3)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    b1 = daily_bias("XAUUSD", c)
    c2 = c.copy()
    c2.iloc[-1] *= 1.5  # shock on the last day must not change that day's bias
    b2 = daily_bias("XAUUSD", c2)
    assert b1["bias"].iloc[-1] == b2["bias"].iloc[-1]


def test_guardrails():
    live = {"decision": "WAIT", "plan": {"entry_ref": 1.10, "stop": 1.102, "stop_distance": 0.002}}
    out = enforce_guardrails(live, {"decision": "SHORT", "stop": 1.102})
    assert out["final_decision"] == "WAIT"
    live["decision"] = "LONG"
    out = enforce_guardrails(live, {"decision": "SHORT", "stop": 1.098})
    assert out["final_decision"] == "WAIT"
    out = enforce_guardrails(live, {"decision": "LONG", "stop": 1.05})
    assert out["final_decision"] == "LONG" and out["stop"] == 1.102


def test_offline_brief_mentions_decision():
    live = {"symbol": "EURUSD", "price": 1.1, "time_utc": "2026-01-01 00:00", "composite": 0.1,
            "regime": "trending", "decision": "WAIT", "bias": "BEARISH", "why_not": ["x"],
            "analysts": [{"name": "trend", "title_fa": "روند", "score": -0.5, "reasons": []}],
            "plan": {"side": "SHORT", "entry_ref": 1.1, "stop": 1.102, "tp1_partial": 1.098,
                     "tp2_final": 1.095, "risk_fraction": 0.005}, "strategic": {"bias": -1.0}, "news": []}
    b = offline_brief(live)
    assert "صبر" in b["summary_fa"] and "WAIT" in b["summary_en"]


def test_performance_handles_empty():
    s = performance(pd.DataFrame(), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-01"), 10_000)
    assert s["trades"] == 0


def test_claude_desk_with_mock(monkeypatch):
    import json as _json
    from types import SimpleNamespace

    from aitrader.ai import claude_desk

    calls = []

    class FakeMessages:
        def create(self, **kw):
            calls.append(kw)
            fmt = kw.get("output_config", {}).get("format")
            if fmt:
                text = _json.dumps({"decision": "SHORT", "conviction": 60, "agrees_with_quant": True,
                                    "entry_zone_low": 1.1, "entry_zone_high": 1.101, "stop": 1.102,
                                    "targets": [1.098], "invalidation": "x", "primary_scenario": "p",
                                    "alternative_scenario": "a", "levels_to_watch": [], "key_risks": [],
                                    "summary_fa": "خلاصه", "summary_en": "summary"})
            else:
                text = "case text"
            return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                                   content=[SimpleNamespace(type="text", text=text)])

    fake = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()), messages=FakeMessages())
    monkeypatch.setattr(claude_desk, "have_credentials", lambda: True)
    monkeypatch.setattr(claude_desk, "_client", lambda: fake)
    live = {"symbol": "EURUSD", "decision": "SHORT", "price": 1.1, "time_utc": "2026-01-01",
            "plan": {"entry_ref": 1.1, "stop": 1.102, "stop_distance": 0.002}, "analysts": [],
            "levels": {}, "backtest": {}}
    out = claude_desk.run_desk(live)
    assert out["mode"] == "claude" and out["decision"]["final_decision"] == "SHORT"
    assert len(calls) == 3
    assert all(c["model"] == claude_desk.MODEL and c["fallbacks"] == "default" for c in calls)
    assert calls[-1]["output_config"]["format"]["type"] == "json_schema"
