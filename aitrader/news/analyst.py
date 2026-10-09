"""LLM news analyst ("analyst #13").

Scores recent headlines per instrument with a structured schema, aggregates
them with confidence and time decay, flags event risk, and journals every
scored headline with the price at scoring time so its real predictive value
can be measured *forward* (an LLM cannot be fairly backtested on news it may
have seen during training).
"""
from __future__ import annotations

import csv
import math
from datetime import datetime, timezone

from ..config import JOURNAL_DIR
from ..ai import llm

CATEGORIES = ["monetary_policy", "inflation", "employment", "growth", "geopolitics", "risk_sentiment",
              "central_bank_gold", "trade_tariffs", "fiscal", "technical_commentary", "other"]

SYSTEM = """You are a macro news analyst on an FX and precious-metals desk.
For each headline decide whether it is relevant for the listed instruments and estimate the
likely price impact over the stated horizon: -2 strongly bearish, -1 bearish, 0 none, +1 bullish,
+2 strongly bullish (for the instrument as quoted, e.g. EURUSD up = euro stronger, XAUUSD up = gold up).
Reason from macro mechanics: hawkish Fed / strong US data / rising US real yields -> USD up, gold down;
risk-off and geopolitical stress -> gold up; ECB hawkish -> EURUSD up. Opinion pieces, price recaps
and technical commentary carry little new information: give them impact 0 or low confidence.
Never invent facts beyond the headline. Persian summaries must be short and fluent.
Use one of these categories: """ + ", ".join(CATEGORIES) + "."


def _schema(symbols: list[str]) -> dict:
    sym_int = {s: {"type": "integer"} for s in symbols}
    return {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "relevant": {"type": "boolean"},
                    "impact": {"type": "object", "properties": sym_int, "required": symbols,
                               "additionalProperties": False},
                    "horizon_hours": {"type": "integer"},
                    "confidence": {"type": "integer"},
                    "category": {"type": "string"},
                },
                "required": ["id", "relevant", "impact", "horizon_hours", "confidence", "category"],
                "additionalProperties": False}},
            "overall": {"type": "object", "properties": {s: {
                "type": "object",
                "properties": {"score": {"type": "number"}, "event_risk": {"type": "boolean"},
                               "rationale_fa": {"type": "string"}},
                "required": ["score", "event_risk", "rationale_fa"], "additionalProperties": False}
                for s in symbols}, "required": symbols, "additionalProperties": False},
            "theme_fa": {"type": "string"},
        },
        "required": ["items", "overall", "theme_fa"],
        "additionalProperties": False,
    }


def analyze(headlines: list[dict], symbols: list[str], prices: dict | None = None,
            calendar: list[dict] | None = None, facts: str | None = None) -> dict:
    if not headlines:
        return {"symbols": {s: {"score": 0.0, "event_risk": False, "rationale_fa": "خبری یافت نشد", "n": 0}
                            for s in symbols}, "theme_fa": "", "items": [], "provider": llm.provider()}
    now = datetime.now(timezone.utc)
    lines = []
    for i, h in enumerate(headlines):
        age = (now - datetime.fromisoformat(h["time"])).total_seconds() / 3600
        lines.append(f"[{i}] {age:.0f}h {h['source'][:18]}: {h['title'][:140]}")
    ctx = f"Instruments: {', '.join(symbols)}. UTC now {now:%Y-%m-%d %H:%M}."
    if prices:
        ctx += " Prices: " + ", ".join(f"{k} {v}" for k, v in prices.items()) + "."
    if facts:
        ctx += "\nVerified market data (trust these over headlines): " + facts
    if calendar:
        ctx += "\nUpcoming/just released high-impact events: " + "; ".join(
            f"{e['time_utc'][:16]} {e['currency']} {e['title']} (fcst {e.get('forecast', '')}, prev {e.get('previous', '')})"
            for e in calendar[:10])
    user = (ctx + "\n\nHeadlines:\n" + "\n".join(lines)
            + "\n\nList only headlines that are relevant with a non-zero impact on at least one instrument "
              "(omit the rest). confidence is an integer from 0 to 100. Then give an overall score in [-1,1] per instrument for the next "
              "24 hours, event_risk=true only if a scheduled or breaking event makes the next few hours "
              "unusually dangerous, and a 3-4 sentence Persian rationale per instrument plus a Persian theme.")
    res = llm.chat_json(SYSTEM, user, _schema(symbols), max_tokens=4000, effort="low")
    by_id = {it["id"]: it for it in res.get("items", []) if isinstance(it, dict)}
    out = {}
    for s in symbols:
        num = den = 0.0
        n = 0
        for i, h in enumerate(headlines):
            it = by_id.get(i)
            if not it or not it.get("relevant"):
                continue
            imp = max(-2, min(2, int(it["impact"].get(s, 0))))
            if imp == 0:
                continue
            age = (now - datetime.fromisoformat(h["time"])).total_seconds() / 3600
            hz = max(2, min(72, int(it.get("horizon_hours", 24))))
            conf = float(it.get("confidence", 50))
            conf = conf * 20 if conf <= 5 else conf     # tolerate a 0-5 scale
            w = max(0.0, min(1.0, conf / 100)) * math.exp(-age / hz)
            num += w * imp / 2
            den += w
            n += 1
        agg = math.tanh(1.5 * num / (den + 1.0)) if den else 0.0
        ov = res.get("overall", {}).get(s, {})
        model_score = max(-1.0, min(1.0, float(ov.get("score", 0.0))))
        out[s] = {"score": round(0.5 * agg + 0.5 * model_score, 3), "headline_score": round(agg, 3),
                  "model_score": round(model_score, 3), "event_risk": bool(ov.get("event_risk", False)),
                  "rationale_fa": ov.get("rationale_fa", ""), "n": n}
    scored = []
    for i, h in enumerate(headlines):
        it = by_id.get(i)
        if it and it.get("relevant") and any(v for v in it["impact"].values()):
            scored.append({**h, **{k: it[k] for k in ("impact", "confidence", "category", "horizon_hours")}})
    _journal(scored, prices or {})
    return {"symbols": out, "theme_fa": res.get("theme_fa", ""), "items": scored[:30],
            "provider": llm.provider(), "model": llm.model_name(), "n_headlines": len(headlines)}


def _journal(scored: list[dict], prices: dict) -> None:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = JOURNAL_DIR / "news_scores.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["scored_at", "headline_time", "source", "title", "category", "confidence",
                        "horizon_h", "impact_json", "prices_json"])
        now = datetime.now(timezone.utc).isoformat()
        import json
        for s in scored:
            w.writerow([now, s["time"], s["source"], s["title"], s["category"], s["confidence"],
                        s["horizon_hours"], json.dumps(s["impact"]), json.dumps(prices)])


def apply_news_filter(live: dict, view: dict | None, veto_level: float = 0.4) -> dict:
    """News can only *block* a trade: strong opposing news or flagged event risk -> WAIT."""
    if not view:
        return live
    live["news_view"] = view
    live["analysts"].append({"name": "news", "title_fa": "اخبار (هوش مصنوعی)", "score": float(view["score"]),
                             "reasons": [f"{view['n']} relevant headlines, headline score {view['headline_score']:+.2f}, "
                                         f"model view {view['model_score']:+.2f}"
                                         + (", EVENT RISK" if view["event_risk"] else "")]})
    if view["event_risk"] and live.get("plan"):
        # LLMs flag event risk generously: halve size rather than block the trade
        p = live["plan"]
        p["risk_fraction"] = round(p["risk_fraction"] / 2, 4)
        p["lots_for_equity"] = round(p["lots_for_equity"] / 2, 2)
        p["news_note"] = "risk halved: news analyst flagged event risk"
    if live["decision"] in ("LONG", "SHORT"):
        side = 1 if live["decision"] == "LONG" else -1
        if view["score"] * side <= -veto_level:
            live["why_not"].append(f"news flow {view['score']:+.2f} strongly opposes the trade")
            live["decision"] = "WAIT"
    return live


def news_context(results: dict) -> dict | None:
    """Collect headlines once and score them for every analysed symbol."""
    from .collector import collect

    if llm.provider() is None:
        return None
    syms = list(results)
    lives = {s: r["live"] for s, r in results.items()}
    facts = []
    for s, L in lives.items():
        for a in L["analysts"]:
            if a["name"] == "macro":
                facts.append(f"{s}: " + "; ".join(a["reasons"]))
        facts.append(f"{s} daily strategic bias {L['strategic']['bias']:+.2f}")
    cal = [e for L in lives.values() for e in L.get("news", [])]
    try:
        heads = collect(syms, max_items=45 if llm.provider() == "groq" else 80)
        view = analyze(heads, syms, {s: L["price"] for s, L in lives.items()}, cal, facts=" | ".join(facts))
        print(f"news: {view.get('n_headlines', 0)} headlines scored by {view.get('provider')}:{view.get('model')}")
        return view
    except Exception as e:  # never let news break the analysis
        print("news analysis skipped:", e)
        return None
