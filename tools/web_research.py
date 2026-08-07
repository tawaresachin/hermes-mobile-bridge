#!/usr/bin/env python3
"""Web research tool — multi-source parallel research briefs.

Searches DuckDuckGo, Google News RSS and major news feeds IN PARALLEL,
extracts the top pages, and returns a structured brief: per-source key
points, a synthesized summary, and a 'latest updates' section with dated
headlines.

Stdlib only beyond httpx (which the project already depends on).
"""
from __future__ import annotations

import asyncio
import html as html_mod
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse
from xml.etree import ElementTree as ET

import httpx

from tools import BaseTool
from tools.web import _decode_ddg_url

# ─── Constants ───────────────────────────────────────────────────────────

NEWS_FEEDS = [
    ("BBC", "https://feeds.bbci.co.uk/news/rss.xml"),
    ("Guardian", "https://www.theguardian.com/world/rss"),
    ("NYT", "https://feeds.nytimes.com/nyt/rss/HomePage.xml"),
]
GNEWS_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
DDG_URL = "https://html.duckduckgo.com/html/?q={q}"

PER_URL_TEXT_CAP = 8000
PER_URL_BYTES_CAP = 1_500_000
BRIEF_CHAR_BUDGET = 4800
SOURCE_TIMEOUT = 8.0
STAGE1_TIMEOUT = 12.0
STAGE3_TIMEOUT = 15.0
HARD_TOTAL_TIMEOUT = 30.0
FRESH_WINDOW_HOURS = 48

_UA_HEADERS = {"User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36"}

_http = httpx.AsyncClient(
    timeout=SOURCE_TIMEOUT, follow_redirects=True, headers=_UA_HEADERS
)

_STOP_WORDS = frozenset(
    "a an the of on in at for to and or but with from by is are was were be been "
    "it its this that these those as latest news today new says said".split()
)

_LOG = logging.getLogger("web_research")


# ─── Dataclasses ─────────────────────────────────────────────────────────


@dataclass
class SourceHit:
    url: str
    title: str
    snippet: str = ""
    source: str = ""
    published: Optional[datetime] = None
    rank: int = 0
    extractable: bool = True


@dataclass
class ExtractResult:
    url: str
    title: str = ""
    text: str = ""
    error: str = ""


# ─── Small helpers ───────────────────────────────────────────────────────


def _elem_text(item, tag: str) -> str:
    el = item.find(tag)
    if el is None or not el.text:
        return ""
    return el.text.strip()


def _parse_rss_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse RSS pubDate (RFC 822) or ISO 8601 into an aware UTC datetime."""
    if not value:
        return None
    value = value.strip()
    dt = None
    if "T" in value:  # ISO 8601 first (email parser mangles ISO strings)
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            dt = None
    if dt is None:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            dt = None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _relative_age(dt: datetime, now: datetime) -> str:
    delta = now - dt
    if delta < timedelta(hours=1):
        minutes = max(1, int(delta.total_seconds() // 60))
        return f"{minutes}m ago"
    if delta < timedelta(hours=48):
        return f"{int(delta.total_seconds() // 3600)}h ago"
    if delta < timedelta(days=21):
        return f"{delta.days}d ago"
    return dt.strftime("%b %d")


def _short_err(e) -> str:
    s = str(e)
    return s if len(s) <= 120 else s[:117] + "..."


def _topic_relevance(title: str, query: str) -> bool:
    """Token-overlap gate for feed items: >=2 shared tokens or any shared
    token of length >= 4 counts as relevant."""
    q = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in _STOP_WORDS}
    t = {t for t in re.findall(r"[a-z0-9]+", title.lower()) if t not in _STOP_WORDS}
    if not q:
        return True
    overlap = q & t
    if len(overlap) >= 2:
        return True
    if any(len(tok) >= 4 for tok in overlap):
        return True
    return len(q) == 1 and bool(overlap)


# ─── HTML → text (stdlib HTMLParser, article/main preferred) ─────────────


class _TextExtractor(HTMLParser):
    """Pull readable text out of HTML, preferring <article>/<main> subtrees."""

    _DROP_TAGS = frozenset({
        "script", "style", "nav", "header", "footer", "aside", "noscript",
        "iframe", "svg", "form", "button", "select", "option", "template",
    })
    _BLOCK_TAGS = frozenset({
        "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section",
        "article", "tr", "blockquote", "pre", "ul", "ol", "table",
    })

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._drop_depth = 0
        self._container = None      # 'article' or 'main' once one is seen
        self._in_container = 0
        self._pre: list[str] = []   # body text before any article/main
        self._chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._DROP_TAGS:
            self._drop_depth += 1
        elif tag in ("article", "main") and self._container is None:
            self._container = tag
            self._in_container = 1
            self._pre = []
        elif tag == self._container:
            self._in_container += 1
        elif not self._drop_depth and tag in self._BLOCK_TAGS:
            target = (self._chunks if self._container is not None
                      else self._pre)
            target.append("\n")

    def handle_endtag(self, tag):
        if tag in self._DROP_TAGS and self._drop_depth > 0:
            self._drop_depth -= 1
        elif tag == self._container:
            self._in_container -= 1
            if self._in_container <= 0:
                self._in_container = 0
                self._container = None

    def handle_data(self, data):
        if self._drop_depth or not data.strip():
            return
        if self._container is not None:
            if self._in_container > 0:
                self._chunks.append(data)
        else:
            self._pre.append(data)

    def close(self):
        super().close()
        if self._container is None and not self._chunks:
            self._chunks = self._pre


def _html_to_text(html_text: str) -> str:
    """Extract readable text, preferring <article>/<main> subtrees."""
    if not html_text:
        return ""
    p = _TextExtractor()
    try:
        p.feed(html_text)
        p.close()
    except Exception:
        return ""
    text = "".join(p._chunks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _extract_title(html_text: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.DOTALL | re.IGNORECASE)
    if m:
        return html_mod.unescape(re.sub(r"<.*?>", "", m.group(1))).strip()
    return ""


def _strip_boilerplate(text: str) -> str:
    """Drop short lines, consecutive repeats; collapse whitespace; cap size."""
    lines, prev = [], None
    for raw in text.splitlines():
        line = " ".join(raw.split()).strip()
        if len(line) < 40:
            continue
        if line == prev:
            continue
        lines.append(line)
        prev = line
    out = "\n".join(lines).strip()
    if len(out) > PER_URL_TEXT_CAP:
        suffix = "\n...[text truncated]"
        out = out[: PER_URL_TEXT_CAP - len(suffix)] + suffix
    return out


# ─── Stage 1: discovery ──────────────────────────────────────────────────


class WebResearch(BaseTool):
    name = "web_research"
    description = (
        "Multi-source web research on any topic. Searches DuckDuckGo, Google "
        "News RSS, and major news feeds IN PARALLEL, extracts the top pages, "
        "and returns a structured brief: per-source key points, a synthesized "
        "summary, and a 'latest updates' section with dated headlines. Prefer "
        "this over web_search when the user wants depth, multiple sources, or "
        "up-to-date information."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The research topic or question",
            },
            "max_sources": {
                "type": "integer",
                "description": "Maximum number of sources (default 5, max 8)",
                "default": 5,
            },
            "freshness": {
                "type": "string",
                "enum": ["auto", "latest", "anytime"],
                "description": (
                    "auto: prefer recent but keep best overall. latest: "
                    "prioritize items published in the last 48 hours. "
                    "anytime: relevance only."
                ),
                "default": "auto",
            },
        },
        "required": ["topic"],
    }

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._client = client if client is not None else _http

    async def run(self, topic: str, max_sources: int = 5, freshness: str = "auto") -> str:
        try:
            max_sources = min(max(int(max_sources), 1), 8)
        except (TypeError, ValueError):
            max_sources = 5
        if freshness not in ("auto", "latest", "anytime"):
            freshness = "auto"
        try:
            return await asyncio.wait_for(
                self._pipeline(topic, max_sources, freshness), HARD_TOTAL_TIMEOUT
            )
        except asyncio.TimeoutError:
            return (
                f"⚠ Research timed out after {HARD_TOTAL_TIMEOUT:.0f}s. "
                "Try a narrower topic or fewer sources."
            )
        except Exception as e:  # never raise
            return f"⚠ Research failed: {e}"

    # ── pipeline ──────────────────────────────────────────────────────────

    async def _pipeline(self, topic: str, max_sources: int, freshness: str) -> str:
        now = datetime.now(timezone.utc)
        stage1 = await self._stage1_discover(topic)
        failures: dict[str, str] = {}
        all_hits: list[SourceHit] = []
        for key, res in stage1.items():
            if isinstance(res, Exception):
                failures[key] = str(res) or type(res).__name__
                _LOG.warning("stage1 source %s failed: %s", key, res)
            else:
                all_hits.extend(res)
                _LOG.info("stage1 source %s: %d hits", key, len(res))
        if not all_hits:
            if failures:
                errs = ", ".join(f"{k}: {_short_err(v)}" for k, v in failures.items())
                return (
                    f"⚠ Research failed: all {len(failures)} sources errored "
                    f"({errs}). Try again or rephrase the topic."
                )
            return "⚠ Research failed: no sources returned any results. Try again or rephrase the topic."

        merged = _merge_and_dedupe(all_hits)
        ranked = _rank_hits(merged, freshness, now)
        pool = [h for h in ranked if h.extractable][: max_sources + 1]
        _LOG.info("extraction pool (%d): %s", len(pool), [h.url for h in pool])

        extracts = await self._stage3_extract(pool)
        extracts_by_url = {r.url: r for r in extracts if isinstance(r, ExtractResult)}
        texts = [r.text for r in extracts_by_url.values() if r.text]
        summary = _summarize(texts, topic, max_chars=800) if texts else (
            "No readable content could be extracted from any source."
        )

        dated = sorted(
            (h for h in merged if h.published is not None),
            key=_pub_dt, reverse=True,
        )
        latest = [(h.published, h.title, h.source) for h in dated[:8]]

        return _format_brief(
            topic, freshness, pool, extracts_by_url, summary, latest, failures, now
        )

    async def _stage1_discover(self, query: str) -> dict:
        results: dict = {}

        async def run_one(key: str, coro):
            try:
                results[key] = await asyncio.wait_for(coro, SOURCE_TIMEOUT)
                _LOG.info("stage1 %s ok (%d hits)", key, len(results[key]))
            except Exception as e:
                results[key] = e
                _LOG.warning("stage1 %s failed: %s", key, e)

        tasks = [
            asyncio.create_task(run_one("ddg", self._ddg_search(query))),
            asyncio.create_task(run_one("gnews", self._gnews_search(query))),
        ]
        for name, url in NEWS_FEEDS:
            tasks.append(asyncio.create_task(run_one(name, self._fetch_feed(name, url, query))))
        await asyncio.wait_for(asyncio.gather(*tasks), STAGE1_TIMEOUT)
        return results

    async def _ddg_search(self, query: str, limit: int = 8) -> list[SourceHit]:
        url = DDG_URL.format(q=quote_plus(query))
        resp = await self._client.get(url, headers=_UA_HEADERS)
        if resp.status_code != 200:
            # DDG answers bot challenges with 202 + captcha page; treat as failure
            resp.raise_for_status()
            raise httpx.HTTPStatusError(
                f"DDG returned HTTP {resp.status_code} (likely bot challenge)",
                request=resp.request, response=resp,
            )
        html_text = resp.text
        hits: list[SourceHit] = []
        for m in re.finditer(
            r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>',
            html_text, re.DOTALL,
        ):
            href = _decode_ddg_url(m.group(1))
            title = re.sub(r"<.*?>", "", m.group(2)).strip()
            snippet = ""
            sm = re.search(
                r'<a class="result__snippet".*?>(.*?)</a>',
                html_text[m.end():], re.DOTALL,
            )
            if sm:
                snippet = re.sub(r"<.*?>", "", sm.group(1)).strip()
            hits.append(SourceHit(
                url=href, title=title, snippet=snippet[:300],
                source="DuckDuckGo", rank=len(hits) + 1,
            ))
            if len(hits) >= limit:
                break
        return hits

    async def _gnews_search(self, query: str, limit: int = 8) -> list[SourceHit]:
        url = GNEWS_URL.format(q=quote_plus(query))
        resp = await self._client.get(url, headers=_UA_HEADERS)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        hits: list[SourceHit] = []
        for item in root.findall(".//item"):
            title = _elem_text(item, "title")
            link = _elem_text(item, "link")
            if not title or not link:
                continue
            clean = re.sub(r"\s+-\s+[^-]*$", "", title).strip()
            headline_only = link.startswith("https://news.google.com/rss/articles/")
            hits.append(SourceHit(
                url=link,
                title=clean,
                snippet=re.sub(r"<.*?>", "", _elem_text(item, "description"))[:300],
                source="Google News",
                published=_parse_rss_datetime(_elem_text(item, "pubDate")),
                rank=len(hits) + 1,
                extractable=not headline_only,
            ))
            if len(hits) >= limit:
                break
        return hits

    async def _fetch_feed(self, name: str, url: str, query: str, cap: int = 5) -> list[SourceHit]:
        resp = await self._client.get(url, headers=_UA_HEADERS)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        hits: list[SourceHit] = []
        for item in root.findall(".//item"):
            title = _elem_text(item, "title")
            link = _elem_text(item, "link")
            if not title or not link or not _topic_relevance(title, query):
                continue
            hits.append(SourceHit(
                url=link,
                title=title,
                snippet=re.sub(r"<.*?>", "", _elem_text(item, "description"))[:300],
                source=name,
                published=_parse_rss_datetime(_elem_text(item, "pubDate")),
                rank=len(hits) + 1,
            ))
            if len(hits) >= cap:
                break
        return hits

    # ── stage 3: extraction ───────────────────────────────────────────────

    async def _stage3_extract(self, pool: list[SourceHit]) -> list[ExtractResult]:
        if not pool:
            return []
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *[self._extract_article(h.url) for h in pool],
                    return_exceptions=True,
                ),
                STAGE3_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return [ExtractResult(url=h.url, error="timeout") for h in pool]
        out = []
        for h, r in zip(pool, results):
            out.append(r if isinstance(r, ExtractResult) else ExtractResult(url=h.url, error=str(r)))
        return out

    async def _extract_article(self, url: str) -> ExtractResult:
        # SSRF guard: only public http(s) targets — never localhost, the
        # OmniRoute gateway, Tailscale/CGNAT, or RFC1918 ranges. Redirects
        # are walked and re-checked hop-by-hop (a Location header pointing
        # at a private host is refused).
        from tools.web import fetch_safe, is_safe_url
        if not is_safe_url(url):
            return ExtractResult(url=url, title="", text="", error="blocked non-public URL")
        try:
            resp, err = await fetch_safe(url, headers=_UA_HEADERS, client=self._client)
            if err:
                return ExtractResult(url=url, error=err.removeprefix("⚠ "))
            assert resp is not None  # err == None implies a response
            try:
                if resp.status_code != 200:
                    return ExtractResult(url=url, error=f"HTTP {resp.status_code}")
                chunks = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > PER_URL_BYTES_CAP:
                        return ExtractResult(
                            url=url, error=f"page too large (>{PER_URL_BYTES_CAP} bytes)"
                        )
                    chunks.append(chunk)
            finally:
                await resp.aclose()
            html_text = b"".join(chunks).decode("utf-8", errors="replace")
            text = _strip_boilerplate(_html_to_text(html_text))
            if not text:
                return ExtractResult(url=url, error="no readable text extracted")
            return ExtractResult(url=url, title=_extract_title(html_text), text=text)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return ExtractResult(url=url, error=str(e) or type(e).__name__)


# ─── Stage 2: merge, dedupe, rank ────────────────────────────────────────


def _normalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        path = (p.path or "/").rstrip("/") or "/"
        query = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            kl = k.lower()
            if kl.startswith("utm_") or kl in ("fbclid", "gclid", "gclsrc"):
                continue
            query.append((k, v))
        return urlunparse((p.scheme.lower() or "http", host, path, "", urlencode(query), ""))
    except Exception:
        return url


def _normalize_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"\s+-\s+[^-]*$", "", t)  # strip trailing ' - site' suffix
    return " ".join(t.split())


def _title_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", _normalize_title(a)))
    tb = set(re.findall(r"[a-z0-9]+", _normalize_title(b)))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _merge_and_dedupe(hits: list[SourceHit]) -> list[SourceHit]:
    """Drop exact URL dups (keep the dated one), then near-duplicate titles
    (Jaccard >= 0.6 keeps the dated hit, else the first)."""
    seen_urls: dict[str, SourceHit] = {}
    merged: list[SourceHit] = []
    for h in hits:
        nu = _normalize_url(h.url)
        if nu in seen_urls:
            existing = seen_urls[nu]
            if existing.published is None and h.published is not None:
                idx = merged.index(existing)
                merged[idx] = h
                seen_urls[nu] = h
            continue
        seen_urls[nu] = h
        merged.append(h)

    final: list[SourceHit] = []
    for h in merged:
        dup = False
        for f in final:
            if _title_similarity(f.title, h.title) >= 0.6:
                if f.published is None and h.published is not None:
                    final[final.index(f)] = h
                dup = True
                break
        if not dup:
            final.append(h)
    return final


def _pub_dt(h: SourceHit) -> datetime:
    """Sort key helper: published datetime or epoch (undated sorts last in desc)."""
    return h.published if h.published is not None else datetime.min.replace(tzinfo=timezone.utc)


def _rank_hits(hits: list[SourceHit], freshness: str, now: datetime) -> list[SourceHit]:
    dated = [h for h in hits if h.published is not None]
    undated = sorted(
        (h for h in hits if h.published is None), key=lambda h: h.rank or 999
    )
    if freshness == "latest":
        window = now - timedelta(hours=FRESH_WINDOW_HOURS)
        recent = sorted(
            (h for h in dated if _pub_dt(h) >= window), key=_pub_dt, reverse=True
        )
        rest = sorted(
            (h for h in dated if _pub_dt(h) < window), key=_pub_dt, reverse=True
        )
        return recent + undated + rest
    if freshness == "anytime":
        return sorted(hits, key=lambda h: h.rank or 999)
    # auto: dated (any age) before undated, recency within tier
    dated_sorted = sorted(dated, key=_pub_dt, reverse=True)
    return dated_sorted + undated


# ─── Stage 4: key points, summary, brief ─────────────────────────────────


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text)
    return [
        re.sub(r"\s+", " ", p).strip()
        for p in parts if len(p.strip()) >= 30
    ]


def _query_tokens(topic: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", topic.lower()) if t not in _STOP_WORDS}


def _extract_key_points(text: str, topic: str, max_chars: int = 480) -> list[str]:
    """Top 3 topic-weighted sentences, ~120-160 chars each."""
    q_tokens = _query_tokens(topic)
    scored = []
    for s in _split_sentences(text):
        toks = set(re.findall(r"[a-z0-9]+", s.lower()))
        score = len(toks & q_tokens) if q_tokens else 1
        scored.append((score, len(s), s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out, used = [], 0
    for _score, ln, s in scored:
        if used + ln + 2 > max_chars:
            continue
        out.append(s)
        used += ln + 2
        if len(out) >= 3:
            break
    return out


def _summarize(texts: list[str], topic: str, max_chars: int = 800) -> str:
    """Extractive summary: top query-weighted sentences, <= max_chars."""
    q_tokens = _query_tokens(topic)
    candidates = []
    for t in texts:
        candidates.extend(_split_sentences(t))
    scored = []
    for s in candidates:
        toks = set(re.findall(r"[a-z0-9]+", s.lower()))
        score = len(toks & q_tokens) if q_tokens else 0
        scored.append((score, len(s), s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    parts, used, seen = [], 0, set()
    for _score, ln, s in scored:
        key = re.sub(r"[^a-z0-9]+", "", s.lower())
        if key in seen:  # drop exact duplicate sentences across sources
            continue
        seen.add(key)
        if used + ln + 2 > max_chars:
            continue
        parts.append(s)
        used += ln + 2
        if used >= int(max_chars * 0.75):
            break
    if not parts:
        return "No summary available."
    return " ".join(parts)


def _format_brief(topic: str, freshness: str, pool: list[SourceHit],
                  extracts_by_url: dict, summary: str, latest: list,
                  failures: dict, now: datetime) -> str:
    lines = [f"🔎 RESEARCH: {topic}  (freshness={freshness})", ""]
    lines.append("━ SOURCES ━")
    if not pool:
        lines.append("(no sources could be gathered)")
    for i, hit in enumerate(pool, 1):
        res = extracts_by_url.get(hit.url)
        lines.append(f"{i}. {hit.title or hit.url}")
        lines.append(f"   {hit.url}")
        if hit.published:
            lines.append(f"   Date: {_relative_age(hit.published, now)}")
        else:
            lines.append("   Date: unknown")
        if res is None or res.error:
            err = res.error if res else "not extracted"
            lines.append(f"   ⚠ extraction failed: {err}")
        else:
            for kp in _extract_key_points(res.text, topic):
                lines.append(f"   • {kp}")
        lines.append("")
    lines.append("━ SUMMARY ━")
    lines.append(summary)
    lines.append("")
    lines.append("━ LATEST UPDATES ━")
    if latest:
        for dt, title, src in latest:
            if now - dt < timedelta(hours=24):
                label = _relative_age(dt, now)
            else:
                label = dt.strftime("%b %d")
            lines.append(f"• {label} — {title} [{src}]")
    else:
        lines.append("(no dated items found)")
    lines.append("")
    lines.append("━ NOTES ━")
    if failures:
        for key, err in failures.items():
            lines.append(f"- source {key} failed: {_short_err(err)}")
    else:
        lines.append("- all discovery sources responded OK")
    lines.append("- Google News headline links (news.google.com/rss/articles/...) "
                 "are redirect stubs; only their titles and dates are used.")
    lines.append("- Some sites render content with JavaScript; if extraction failed, "
                 "the URL itself may still open in a browser.")
    out = "\n".join(lines).strip()
    if len(out) > BRIEF_CHAR_BUDGET:
        suffix = "\n...[brief truncated]"
        out = out[: BRIEF_CHAR_BUDGET - len(suffix)] + suffix
    return out
