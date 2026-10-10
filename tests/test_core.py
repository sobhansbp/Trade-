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
    from aitrader.ai import llm
    monkeypatch.setattr(llm, "provider", lambda: "claude")
    monkeypatch.setattr(claude_desk, "_client", lambda: fake)
    live = {"symbol": "EURUSD", "decision": "SHORT", "price": 1.1, "time_utc": "2026-01-01",
            "plan": {"entry_ref": 1.1, "stop": 1.102, "stop_distance": 0.002}, "analysts": [],
            "levels": {}, "backtest": {}}
    out = claude_desk.run_desk(live)
    assert out["mode"] == "llm" and out["decision"]["final_decision"] == "SHORT"
    assert len(calls) == 3
    assert all(c["model"] == claude_desk.MODEL and c["fallbacks"] == "default" for c in calls)
    assert calls[-1]["output_config"]["format"]["type"] == "json_schema"


def test_news_filter_only_blocks_or_shrinks():
    from aitrader.news.analyst import apply_news_filter
    base = {"decision": "LONG", "why_not": [], "analysts": [],
            "plan": {"risk_fraction": 0.005, "lots_for_equity": 1.0}}
    v = {"score": -0.6, "headline_score": -0.5, "model_score": -0.7, "event_risk": False, "n": 5}
    out = apply_news_filter({**base, "why_not": [], "analysts": []}, v)
    assert out["decision"] == "WAIT"
    v2 = {**v, "score": 0.3, "event_risk": True}
    out2 = apply_news_filter({**base, "why_not": [], "analysts": [], "plan": dict(base["plan"])}, v2)
    assert out2["decision"] == "LONG" and out2["plan"]["risk_fraction"] == 0.0025
    w = {"decision": "WAIT", "why_not": [], "analysts": [], "plan": {"risk_fraction": 0.005, "lots_for_equity": 1}}
    assert apply_news_filter(w, {**v, "score": 0.9})["decision"] == "WAIT"   # news never opens a trade


def test_paper_account_lifecycle(tmp_path, monkeypatch):
    from aitrader.live import paper
    monkeypatch.setattr(paper, "TRADES", tmp_path / "t.csv")
    monkeypatch.setattr(paper, "EQUITY", tmp_path / "e.csv")
    cfg = StrategyConfig()
    monkeypatch.setattr(paper, "STATE", tmp_path / "s.json")
    st = paper.load_state(10_000)
    idx = pd.date_range("2026-01-05 08:00", periods=4, freq="1h", tz="UTC")
    bars = pd.DataFrame({"open": [1.1000] * 4, "high": [1.1005] * 4, "low": [1.0995] * 4,
                         "close": [1.1000] * 4, "volume": 0.0}, index=idx)
    live = {"symbol": "EURUSD", "time_utc": str(idx[-1]),
            "plan": {"stop_distance": 0.0020, "risk_fraction": 0.005}}
    msgs = paper.tactical_step(st, live, bars, "LONG", cfg)
    assert "EURUSD" in st["tactical"]["positions"] and any("opened" in m for m in msgs)
    p = st["tactical"]["positions"]["EURUSD"]
    assert abs(p["risk_usd"] - 50.0) < 1e-6
    # same bar again -> no duplicate entry
    paper.tactical_step(st, live, bars, "LONG", cfg)
    assert len(st["tactical"]["positions"]) == 1
    # next bars: TP1 hit, then back to breakeven
    idx2 = pd.date_range(idx[-1] + pd.Timedelta("1h"), periods=2, freq="1h", tz="UTC")
    b2 = pd.DataFrame({"open": [1.1001, 1.1015], "high": [1.1025, 1.1016], "low": [1.0999, 1.0990],
                       "close": [1.1020, 1.0995], "volume": 0.0}, index=idx2)
    live2 = {**live, "time_utc": str(idx2[-1])}
    msgs = paper.tactical_step(st, live2, pd.concat([bars, b2]), "WAIT", cfg)
    assert "EURUSD" not in st["tactical"]["positions"]
    t = pd.read_csv(tmp_path / "t.csv")
    assert t["reason"].iloc[0] == "breakeven" and 0.3 < t["r_multiple"].iloc[0] < 0.6
    assert st["tactical"]["equity"] > 10_000
    # swing book marks to market and rebalances
    paper.swing_step(st, "XAUUSD", 4000.0, 1.0, 1.0, 2)
    paper.swing_step(st, "XAUUSD", 4040.0, 1.0, 1.0, 2)
    assert st["swing"]["equity"] > 10_000 + 40   # +1% on 50% notional = ~+50 minus costs
    md = paper.status_markdown(st, {}, None)
    assert "حساب دمو" in md


def test_summary_text():
    from aitrader.live.runner import summary_text
    st = {"initial": 10_000, "tactical": {"equity": 10_100, "positions": {}}, "swing": {"equity": 9_950, "positions": {}}}
    live = {"price": 1.1, "decision": "WAIT", "composite": 0.1, "strategic": {"bias": -1.0}}
    txt = summary_text(st, {"EURUSD": live}, None)
    assert "گزارش حساب دمو" in txt and "+1.00%" in txt and "EURUSD" in txt


def test_groq_json_step_down(monkeypatch):
    from types import SimpleNamespace

    from aitrader.ai import llm
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((json.get("response_format") or {}).get("type"))
        if len(calls) < 3:
            return SimpleNamespace(status_code=400, text='{"error":{"code":"json_validate_failed"}}', headers={})
        return SimpleNamespace(status_code=200, headers={}, text="",
                               json=lambda: {"choices": [{"finish_reason": "stop",
                                                          "message": {"content": 'ok {"a": 1} done'}}]})
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("AITRADER_LLM", "groq")
    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = llm.chat_json("sys", "user", {"type": "object", "properties": {"a": {"type": "integer"}},
                                        "required": ["a"], "additionalProperties": False})
    assert out == {"a": 1}
    assert calls == ["json_schema", "json_object", None]


def test_forward_rules_and_logger(tmp_path, monkeypatch):
    from aitrader.live import forward as fw

    # one synthetic London day (BST): flat 2000 range overnight, break lower at 08:00 London, drift down
    idx = pd.date_range("2026-10-13 22:00", "2026-10-14 20:00", freq="5min", tz="UTC")
    px = np.full(len(idx), 2000.0)
    brk = idx >= pd.Timestamp("2026-10-14 07:00", tz="UTC")          # 08:00 London
    px[brk] = 2000.0 - 2.0 - 0.05 * np.arange(brk.sum())
    df = pd.DataFrame({"open": px, "close": px, "high": px + 0.5, "low": px - 0.5}, index=idx)
    df.loc[~brk, ["high", "low"]] = [2001.0, 1999.0]
    r = fw.gold_london_breakout_short(df, pd.Timestamp("2026-10-14"))
    assert r["side"] == -1 and r["entry"] == 1999.0 and r["gross_bp"] > 0 and r["net_bp"] < r["gross_bp"]
    # an upward first break is not part of the (short-only) hypothesis
    up = df.copy()
    up.loc[brk, ["open", "close"]] = 2005.0
    up.loc[brk, "high"], up.loc[brk, "low"] = 2006.0, 2004.0
    assert fw.gold_london_breakout_short(up, pd.Timestamp("2026-10-14"))["side"] == 0

    # the logger only evaluates completed days after START and never logs a day twice
    monkeypatch.setattr(fw, "PATH", tmp_path / "fwd.csv")
    monkeypatch.setattr(fw, "START", pd.Timestamp("2026-10-14"))
    data = {"XAUUSD": df}
    first = fw.update(now=pd.Timestamp("2026-10-14 13:00", tz="UTC"), data=data)   # 08:20 ET + 30 min passed
    assert len(first) == 1 and "gold_overnight_long" in first[0] and "no data" in first[0]
    later = fw.update(now=pd.Timestamp("2026-10-14 15:45", tz="UTC"), data=data)   # 16:00 London + 30 min
    assert len(later) == 1 and "breakout" in later[0]
    last = fw.update(now=pd.Timestamp("2026-10-14 18:30", tz="UTC"), data=data)    # 13:30 ET + 30 min
    assert len(last) == 1 and "gold_day_long" in last[0]
    assert fw.update(now=pd.Timestamp("2026-10-14 19:00", tz="UTC"), data=data) == []
    s = {h["hypothesis"]: h for h in fw.summary()}
    assert s["gold_london_breakout_short"]["n"] == 1 and s["gold_london_breakout_short"]["mean_net_bp"] > 0
