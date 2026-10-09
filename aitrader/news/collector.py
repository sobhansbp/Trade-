"""Headline collector: central banks, financial media and Google News queries."""
from __future__ import annotations

import email.utils
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests

from ..config import CACHE_DIR

FEEDS = {
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ECB": "https://www.ecb.europa.eu/rss/press.html",
    "Investing.com forex": "https://www.investing.com/rss/news_1.rss",
    "Investing.com commodities": "https://www.investing.com/rss/news_11.rss",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
}
GOOGLE_QUERIES = {
    "EURUSD": ["EUR/USD", "euro dollar ECB", "Lagarde"],
    "XAUUSD": ["gold price", "gold futures", "central bank gold buying"],
    "USD": ["Federal Reserve Powell", "US dollar index", "US CPI inflation", "Treasury yields"],
    "GBPUSD": ["GBP/USD pound sterling", "Bank of England"],
    "USDJPY": ["USD/JPY yen", "Bank of Japan"],
    "AUDUSD": ["AUD/USD Australian dollar RBA"],
    "USDCAD": ["USD/CAD Canadian dollar Bank of Canada"],
    "XAGUSD": ["silver price"],
}
UA = {"User-Agent": "Mozilla/5.0 (aitrader news collector)"}


def _gnews(q: str) -> str:
    return ("https://news.google.com/rss/search?q=" + requests.utils.quote(q + " when:1d")
            + "&hl=en-US&gl=US&ceid=US:en")


def _parse(xml_text: str, source: str) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        pub = it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date") or ""
        try:
            dt = email.utils.parsedate_to_datetime(pub)
            dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            try:
                dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.now(timezone.utc)
        src_el = it.find("source")
        src = src_el.text.strip() if src_el is not None and src_el.text else source
        if source.startswith("Google") and " - " in title:
            title, _, src = title.rpartition(" - ")
        out.append({"title": re.sub(r"\s+", " ", title), "source": src, "feed": source,
                    "time": dt.astimezone(timezone.utc).isoformat(), "link": it.findtext("link") or ""})
    return out


def collect(symbols: list[str], hours: int = 24, max_items: int = 80, cache_s: int = 900) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(("|".join(sorted(symbols)) + str(hours)).encode()).hexdigest()[:10]
    path = CACHE_DIR / f"news_{key}.json"
    if path.exists() and time.time() - path.stat().st_mtime < cache_s:
        return json.loads(path.read_text())
    urls = dict(FEEDS)
    for grp in ["USD", *symbols]:
        for q in GOOGLE_QUERIES.get(grp, []):
            urls[f"Google News: {q}"] = _gnews(q)
    from concurrent.futures import ThreadPoolExecutor

    def fetch(kv):
        name, url = kv
        try:
            r = requests.get(url, headers=UA, timeout=20)
            return _parse(r.text, name) if r.ok else []
        except requests.RequestException:
            return []

    items = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(fetch, urls.items()):
            items += res
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["time"], reverse=True):
        norm = re.sub(r"[^a-z0-9 ]", "", it["title"].lower())[:90]
        if norm in seen or datetime.fromisoformat(it["time"]) < cutoff:
            continue
        seen.add(norm)
        out.append(it)
    # keep central-bank items first, then most recent
    out.sort(key=lambda x: (x["feed"] not in ("Federal Reserve", "ECB"), x["time"]), reverse=False)
    cb = [x for x in out if x["feed"] in ("Federal Reserve", "ECB")]
    rest = sorted([x for x in out if x["feed"] not in ("Federal Reserve", "ECB")], key=lambda x: x["time"], reverse=True)
    out = (cb + rest)[:max_items]
    path.write_text(json.dumps(out))
    return out
