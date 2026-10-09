"""Claude-powered research desk.

A TradingAgents-style debate on top of the quantitative engine:

  1. Bull researcher  - strongest evidence-based case for longs
  2. Bear researcher  - strongest evidence-based case for shorts
  3. Head trader + risk manager - weighs both against the quant dossier and the
     chart image, returns a structured decision (JSON schema enforced)

Guardrails are enforced in code, not by prompt: the AI may *veto* a quant
signal (turn it into WAIT) or tighten nothing / widen nothing outside the risk
envelope, but it can never open a trade the quant engine did not propose,
flip its direction, or change position risk.

Works without credentials: ``offline_brief`` writes a deterministic summary.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np

MODEL = os.environ.get("AITRADER_MODEL", "claude-opus-5-5")

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["LONG", "SHORT", "WAIT"]},
        "conviction": {"type": "integer"},
        "agrees_with_quant": {"type": "boolean"},
        "entry_zone_low": {"type": "number"},
        "entry_zone_high": {"type": "number"},
        "stop": {"type": "number"},
        "targets": {"type": "array", "items": {"type": "number"}},
        "invalidation": {"type": "string"},
        "primary_scenario": {"type": "string"},
        "alternative_scenario": {"type": "string"},
        "levels_to_watch": {"type": "array", "items": {"type": "string"}},
        "key_risks": {"type": "array", "items": {"type": "string"}},
        "summary_fa": {"type": "string"},
        "summary_en": {"type": "string"},
    },
    "required": ["decision", "conviction", "agrees_with_quant", "entry_zone_low", "entry_zone_high",
                 "stop", "targets", "invalidation", "primary_scenario", "alternative_scenario",
                 "levels_to_watch", "key_risks", "summary_fa", "summary_en"],
    "additionalProperties": False,
}

SYSTEM = """You are part of a professional FX & precious-metals research desk.
You receive a machine-generated dossier for one instrument: a strategic daily bias \
validated on 20 years of data, twelve specialist analyst scores (trend, momentum, mean \
reversion, smart-money structure, liquidity sweeps, patterns, breakouts, levels, macro, \
seasonality, COT positioning, machine learning), key levels, upcoming high-impact news, \
and honest out-of-sample backtest statistics, plus an H1 chart image.

Ground every claim in the dossier or the chart. Quote prices with the instrument's precision. \
Out-of-sample evidence matters more than any single pattern; the backtests show hourly \
technical signals are weak on their own, so treat them as timing tools. Be explicit about \
uncertainty. Never invent news, prices or levels that are not in the inputs."""


def _client():
    try:
        import anthropic
    except ImportError:
        return None
    try:
        return anthropic.Anthropic()
    except Exception:
        return None


def have_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE")
                or Path.home().joinpath(".config", "anthropic").exists())


def build_dossier(live: dict) -> dict:
    keep = ["symbol", "time_utc", "price", "decision", "bias", "composite", "agreement", "confidence",
            "ml_p_long", "ml_p_short", "regime", "trend_strength", "vol_state", "atr", "why_not",
            "plan", "strategic", "htf", "news", "news_blackout", "patterns"]
    d = {k: live.get(k) for k in keep}
    d["analysts"] = [{"name": a["name"], "score": round(a["score"], 3), "reasons": a["reasons"]}
                     for a in live.get("analysts", [])]
    lv = live.get("levels", {})
    d["levels"] = {"support_resistance": lv.get("support_resistance"),
                   "zones": [{k: z[k] for k in ("kind", "top", "bottom", "touched")} for z in lv.get("zones", [])],
                   "session": lv.get("session"), "fibonacci_last_leg": lv.get("fibonacci_last_leg"),
                   "harmonic_prz": lv.get("harmonic_prz")}
    bt = live.get("backtest", {})
    keys = ("trades", "win_rate", "avg_r", "profit_factor", "sharpe", "max_drawdown", "total_return")
    d["evidence"] = {
        k: {kk: (bt.get(k) or {}).get(kk) for kk in keys}
        for k in ("production_holdout", "production_dev", "random_entries")}
    d["evidence"]["swing_mode_20y"] = (live.get("swing_backtest") or {}).get("2005_2023")
    d["evidence"]["note"] = "hourly technical signals alone had ~zero edge; daily bias + desk agreement is the tested rule"
    return json.loads(json.dumps(d, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))


def _content(dossier: dict, chart_png: str | None, instruction: str) -> list:
    blocks = []
    if chart_png and Path(chart_png).exists():
        data = base64.standard_b64encode(Path(chart_png).read_bytes()).decode()
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
    blocks.append({"type": "text", "text": "DOSSIER (JSON):\n" + json.dumps(dossier, ensure_ascii=False)})
    blocks.append({"type": "text", "text": instruction})
    return blocks


def _call(client, content, effort="high", schema=None, max_tokens=16000, system=SYSTEM):
    import anthropic

    kwargs = dict(model=MODEL, max_tokens=max_tokens, system=system,
                  thinking={"type": "adaptive"},
                  messages=[{"role": "user", "content": content}])
    output_config = {"effort": effort}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    kwargs["output_config"] = output_config
    try:
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    except anthropic.BadRequestError:
        resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model declined: {getattr(resp.stop_details, 'category', None)}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text


def run_desk(live: dict, chart_png: str | None = None) -> dict:
    """Bull/bear debate + head-trader decision on the configured LLM provider
    (Groq or Claude). Falls back to the offline brief."""
    from . import llm

    prov = llm.provider()
    if prov is None:
        return {"mode": "offline", **offline_brief(live)}
    dossier = build_dossier(live)
    if live.get("news_view"):
        dossier["news"] = {"view": live["news_view"], "headlines": live.get("news_headlines", [])[:15]}
    img = None
    # vision models on Groq misread chart numbers in testing; send the image only to Claude
    # unless explicitly enabled
    if chart_png and Path(chart_png).exists() and (prov == "claude" or os.environ.get("AITRADER_GROQ_VISION") == "1"):
        img = base64.standard_b64encode(Path(chart_png).read_bytes()).decode()
    base = "DOSSIER (JSON):\n" + json.dumps(dossier, ensure_ascii=False) + "\n\n"
    try:
        bull = llm.chat(SYSTEM, base + "Role: BULL researcher. In <= 220 words, make the strongest evidence-based "
                        "case for a LONG over the next 1-3 days. Cite specific analyst readings, levels and news. "
                        "End with the single condition that would prove you wrong.",
                        effort="low", max_tokens=1800, image_png_b64=img)
        bear = llm.chat(SYSTEM, base + "Role: BEAR researcher. In <= 220 words, make the strongest evidence-based "
                        "case for a SHORT over the next 1-3 days. Cite specific analyst readings, levels and news. "
                        "End with the single condition that would prove you wrong.",
                        effort="low", max_tokens=1800, image_png_b64=img)
        decision = llm.chat_json(
            SYSTEM, base + "Role: HEAD TRADER + RISK MANAGER. Bull case:\n" + bull + "\n\nBear case:\n" + bear +
            "\n\nWeigh both against the dossier. Rules: you may only choose the quant decision "
            f"('{live['decision']}') or WAIT; prefer WAIT when evidence conflicts, news is imminent "
            "or the setup is late. Fill every field. conviction is 0-100. If WAIT, describe the exact "
            "conditions/levels that would trigger a trade in the scenario fields and still give the "
            "conditional entry zone/stop/targets for the strategic bias direction. summary_fa must be "
            "fluent Persian (Farsi) for a trader, 4-7 sentences.",
            DECISION_SCHEMA, effort="medium", max_tokens=5000 if prov == "groq" else 16000, image_png_b64=img)
    except (llm.LLMError, json.JSONDecodeError, KeyError) as e:
        return {"mode": "offline", "error": str(e), **offline_brief(live)}
    return {"mode": "llm", "provider": prov, "model": llm.model_name(), "bull_case": bull, "bear_case": bear,
            "decision": enforce_guardrails(live, decision)}


def enforce_guardrails(live: dict, ai: dict) -> dict:
    quant = live["decision"]
    final = ai.get("decision", "WAIT")
    notes = []
    if quant == "WAIT" and final != "WAIT":
        notes.append("AI proposed a trade the quant engine did not - overridden to WAIT")
        final = "WAIT"
    if quant in ("LONG", "SHORT") and final not in (quant, "WAIT"):
        notes.append("AI tried to flip direction - overridden to WAIT")
        final = "WAIT"
    plan = live.get("plan", {})
    stop = ai.get("stop")
    if plan and stop:
        entry = plan["entry_ref"]
        dist = abs(entry - stop)
        q = plan["stop_distance"]
        if not (0.6 * q <= dist <= 1.5 * q):
            notes.append(f"AI stop {stop} outside risk envelope - quant stop {plan['stop']} kept")
            ai["stop"] = plan["stop"]
    ai["final_decision"] = final
    ai["guardrail_notes"] = notes
    return ai


# ------------------------------------------------------------ offline brief

def _fa_side(x: str) -> str:
    return {"LONG": "خرید", "SHORT": "فروش", "WAIT": "صبر / بدون معامله",
            "BULLISH": "صعودی", "BEARISH": "نزولی"}.get(x, x)


def offline_brief(live: dict) -> dict:
    """Deterministic narrative built from the analyst outputs (no LLM)."""
    a = sorted(live.get("analysts", []), key=lambda x: -abs(x["score"]))
    bulls = [x for x in a if x["score"] > 0.15][:4]
    bears = [x for x in a if x["score"] < -0.15][:4]
    p = live.get("plan", {})
    st = live.get("strategic", {})
    sym = live["symbol"]
    fa = [f"تحلیل {sym} در قیمت {live['price']} (UTC {live['time_utc'][:16]}):",
          f"سوگیری راهبردی روزانه {st.get('bias', 0):+.2f} ({_fa_side(live['bias'])}) و برآیند میز تحلیل ساعتی "
          f"{live['composite']:+.3f}؛ رژیم بازار: {live['regime']}.",
          f"تصمیم سیستم: {_fa_side(live['decision'])}."]
    if live.get("why_not"):
        fa.append("دلیل عدم ورود: " + "؛ ".join(live["why_not"]))
    if bulls:
        fa.append("شواهد صعودی: " + "، ".join(f"{x['title_fa']} ({x['score']:+.2f})" for x in bulls))
    if bears:
        fa.append("شواهد نزولی: " + "، ".join(f"{x['title_fa']} ({x['score']:+.2f})" for x in bears))
    if p:
        fa.append(f"پلن مشروط ({_fa_side(p['side'])}): ورود حدود {p['entry_ref']}، حد ضرر {p['stop']}، "
                  f"تارگت اول {p['tp1_partial']} و تارگت نهایی {p['tp2_final']} (ریسک {p['risk_fraction']*100:.2f}٪ سرمایه).")
    if live.get("news"):
        n = live["news"][0]
        fa.append(f"خبر مهم پیش‌رو: {n['title']} ({n['currency']}) {n['hours_from_now']:+.1f} ساعت دیگر.")
    en = (f"{sym} @ {live['price']}: strategic bias {st.get('bias', 0):+.2f}, desk composite "
          f"{live['composite']:+.3f}, regime {live['regime']}. Decision: {live['decision']}. "
          + ("Blocked because: " + "; ".join(live['why_not']) + ". " if live.get("why_not") else "")
          + ("Bull evidence: " + ", ".join(f"{x['name']} {x['score']:+.2f}" for x in bulls) + ". " if bulls else "")
          + ("Bear evidence: " + ", ".join(f"{x['name']} {x['score']:+.2f}" for x in bears) + "." if bears else ""))
    return {"summary_fa": "\n".join(fa), "summary_en": en, "final_decision": live["decision"]}
