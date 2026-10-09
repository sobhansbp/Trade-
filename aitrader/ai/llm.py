"""Provider-agnostic LLM access: Groq (OpenAI-compatible API) or Claude.

Provider choice (first match):
  AITRADER_LLM=groq|claude   explicit
  GROQ_API_KEY set           -> groq
  Anthropic credentials      -> claude
Keys are read from the environment only; never hard-code them.
"""
from __future__ import annotations

import json
import os
import time

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("AITRADER_GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.environ.get("AITRADER_GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
# Groq counts input + max_completion_tokens against the tokens-per-minute quota
# (8k TPM on the free tier), so keep requests lean there.
GROQ_TPM = int(os.environ.get("AITRADER_GROQ_TPM", "8000"))


class LLMError(RuntimeError):
    pass


def provider() -> str | None:
    p = os.environ.get("AITRADER_LLM", "").lower()
    if p in ("groq", "claude"):
        return p
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    from .claude_desk import have_credentials
    return "claude" if have_credentials() else None


def model_name() -> str:
    p = provider()
    if p == "groq":
        return GROQ_MODEL
    if p == "claude":
        from .claude_desk import MODEL
        return MODEL
    return "offline"


def _groq(messages: list, schema: dict | None, model: str, max_tokens: int, effort: str) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set")
    est_in = sum(len(m["content"]) if isinstance(m["content"], str) else 2000 for m in messages) // 3
    max_tokens = int(max(800, min(max_tokens, GROQ_TPM * 0.92 - est_in)))
    body = {"model": model, "messages": messages, "max_completion_tokens": max_tokens, "temperature": 0.2}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = {"low": "low", "medium": "medium"}.get(effort, "high")
    if schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "result", "schema": schema, "strict": True}}
    last = None
    for attempt in range(7):
        try:
            r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = LLMError(f"groq {r.status_code}: {r.text[:200]}")
            wait = r.headers.get("retry-after")
            time.sleep(min(65.0, float(wait) + 1 if wait else 5 * (attempt + 1)))
            continue
        if r.status_code >= 400:
            # strict schema unsupported by a model -> retry once in plain JSON mode
            if schema and "json_schema" in r.text and body.get("response_format", {}).get("type") == "json_schema":
                body["response_format"] = {"type": "json_object"}
                messages[-1]["content"] = (messages[-1]["content"] if isinstance(messages[-1]["content"], str)
                                           else messages[-1]["content"])
                continue
            raise LLMError(f"groq {r.status_code}: {r.text[:300]}")
        out = r.json()["choices"][0]
        if out.get("finish_reason") == "length" and not (out["message"].get("content") or "").strip():
            raise LLMError("groq ran out of output tokens (reasoning used the whole budget)")
        return out["message"]["content"] or ""
    raise LLMError(f"groq request failed: {last}")


def _claude(system: str, user: str | list, schema: dict | None, max_tokens: int, effort: str) -> str:
    from .claude_desk import _call, _client
    client = _client()
    if client is None:
        raise LLMError("no Anthropic client")
    content = user if isinstance(user, list) else [{"type": "text", "text": user}]
    import anthropic
    try:
        return _call(client, content, effort=effort, schema=schema, max_tokens=max_tokens, system=system)
    except (anthropic.APIError, RuntimeError) as e:
        raise LLMError(str(e)) from e


def chat(system: str, user: str, schema: dict | None = None, max_tokens: int = 8000,
         effort: str = "high", image_png_b64: str | None = None) -> str:
    p = provider()
    if p is None:
        raise LLMError("no LLM provider configured (set GROQ_API_KEY or ANTHROPIC_API_KEY)")
    if p == "groq":
        model = GROQ_MODEL
        if image_png_b64:
            model = GROQ_VISION_MODEL
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_png_b64}}]
        else:
            content = user
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        return _groq(msgs, schema, model, max_tokens, effort)
    blocks = []
    if image_png_b64:
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_png_b64}})
    blocks.append({"type": "text", "text": user})
    return _claude(system, blocks, schema, max_tokens, effort)


def chat_json(system: str, user: str, schema: dict, **kw) -> dict:
    txt = chat(system, user + "\n\nReturn only JSON matching the schema.", schema=schema, **kw)
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`").split("\n", 1)[1] if "\n" in txt else txt
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        s, e = txt.find("{"), txt.rfind("}")
        if s >= 0 and e > s:
            return json.loads(txt[s:e + 1])
        raise LLMError(f"model did not return JSON: {txt[:200]}")
