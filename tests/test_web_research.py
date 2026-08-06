#!/usr/bin/env python3
"""Unit tests for tools/web_research.py — plain python, no pytest.

Run:  python3 tests/test_web_research.py
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

import tools.web_research as wr
from tools.web import _decode_ddg_url
from tools.web_research import (
    BRIEF_CHAR_BUDGET,
    FRESH_WINDOW_HOURS,
    PER_URL_TEXT_CAP,
    ExtractResult,
    SourceHit,
    _format_brief,
    _html_to_text,
    _merge_and_dedupe,
    _normalize_title,
    _normalize_url,
    _parse_rss_datetime,
    _rank_hits,
    _strip_boilerplate,
    _summarize,
)

UTC = timezone.utc


def _rfc(dt: datetime) -> str:
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


# ─── date parsing ────────────────────────────────────────────────────────


def test_parse_rss_datetime():
    assert _parse_rss_datetime("Mon, 03 Aug 2026 19:55:29 GMT") == datetime(2026, 8, 3, 19, 55, 29, tzinfo=UTC)
    assert _parse_rss_datetime("Mon, 03 Aug 2026 19:55:29 +0000") == datetime(2026, 8, 3, 19, 55, 29, tzinfo=UTC)
    assert _parse_rss_datetime("Mon, 03 Aug 2026 14:55:29 -0500") == datetime(2026, 8, 3, 19, 55, 29, tzinfo=UTC)
    assert _parse_rss_datetime("2026-08-03T19:55:29Z") == datetime(2026, 8, 3, 19, 55, 29, tzinfo=UTC)
    assert _parse_rss_datetime("2026-08-03T19:55:29+01:00") == datetime(2026, 8, 3, 18, 55, 29, tzinfo=UTC)
    assert _parse_rss_datetime("not a date") is None
    assert _parse_rss_datetime("") is None
    assert _parse_rss_datetime(None) is None
    print("PASS  _parse_rss_datetime (RFC822 GMT/+0000/-0500, ISO, garbage->None)")


def test_decode_ddg_url():
    assert _decode_ddg_url(
        "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle%3Fa%3D1%26b%3D2&rut=abc"
    ) == "https://example.com/article?a=1&b=2"
    assert _decode_ddg_url("https://example.com/plain") == "https://example.com/plain"
    assert _decode_ddg_url("//duckduckgo.com/l/?rut=abc") == "//duckduckgo.com/l/?rut=abc"
    assert _decode_ddg_url("//duckduckgo.com/l/?uddg=notaurl") == "//duckduckgo.com/l/?uddg=notaurl"
    assert _decode_ddg_url(
        "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F%3Futm_source%3Dx&amp;rut=1"
    ) == "https://example.com/?utm_source=x"
    print("PASS  _decode_ddg_url (encoded uddg, missing, malformed, HTML-escaped)")


# ─── normalize / dedupe ──────────────────────────────────────────────────


def test_normalize_url_and_title():
    assert _normalize_url(
        "https://Example.com/Path/?utm_source=news&a=1&fbclid=z#top"
    ) == "https://example.com/Path?a=1"
    assert _normalize_url("http://x.com//") == "http://x.com/"
    assert _normalize_url("https://x.com/a?utm_campaign=c&gclid=g") == "https://x.com/a"
    assert _normalize_title("  EU Passes   Landmark AI Act - BBC News ") == "eu passes landmark ai act"
    assert _normalize_title("Plain Title") == "plain title"
    print("PASS  _normalize_url / _normalize_title")


def test_dedupe():
    now = datetime.now(UTC)
    exact1 = SourceHit(url="https://Example.com/Story?utm_source=x&fbclid=y#frag",
                       title="Same story - Site A", published=now - timedelta(hours=1), rank=1)
    exact2 = SourceHit(url="https://example.com/story/", title="Same story",
                       published=None, rank=2)          # URL dup, undated
    near = SourceHit(url="https://example.com/other", title="Same story details",
                     published=None, rank=3)             # title near-dup (Jaccard 2/3)
    distinct = SourceHit(url="https://example.com/different",
                         title="Completely unrelated headline", published=None, rank=4)
    merged = _merge_and_dedupe([exact1, exact2, near, distinct])
    assert len(merged) == 2, [h.title for h in merged]
    assert [h.title for h in merged] == ["Same story - Site A", "Completely unrelated headline"]
    print("PASS  _merge_and_dedupe (exact URL dup, near-dup title, distinct kept)")


# ─── html → text / boilerplate / caps ────────────────────────────────────


def test_html_to_text_prefers_article():
    html = """<html><body><nav>Home World Tech</nav><header>Site Masthead</header>
    <script>var junk = 1;</script>
    <article><h1>Headline</h1><p>The European Union approved landmark artificial intelligence regulation on Tuesday for high-risk systems.</p></article>
    <footer>Copyright 2026</footer></body></html>"""
    text = _html_to_text(html)
    assert "European Union approved" in text
    assert "Home World Tech" not in text
    assert "Site Masthead" not in text
    assert "Copyright 2026" not in text
    assert "var junk" not in text
    print("PASS  _html_to_text prefers <article>, drops nav/header/footer/script")


def test_html_to_text_no_article():
    html = ("<html><body><p>Just a paragraph of body text that is long enough "
            "to survive the boilerplate filter later.</p><footer>Footer junk</footer></body></html>")
    text = _html_to_text(html)
    assert "Just a paragraph" in text
    assert "Footer junk" not in text
    print("PASS  _html_to_text without <article> keeps body, drops footer")


def test_strip_boilerplate():
    text = ("Short.\n\n"
            "Long meaningful sentence about the artificial intelligence regulation act passed in the European Union on Tuesday.\n" * 3)
    out = _strip_boilerplate(text)
    assert "Short." not in out
    nonempty = [l for l in out.splitlines() if l.strip()]
    assert len(nonempty) == 1, nonempty  # consecutive repeats dropped
    print("PASS  _strip_boilerplate (short lines + consecutive repeats dropped)")


def test_text_cap():
    big = "The quick brown fox jumps over the lazy dog. " * 500
    out = _strip_boilerplate(big)
    assert len(out) <= PER_URL_TEXT_CAP, len(out)
    assert "truncated" in out
    print("PASS  text cap enforced (<= %d chars)" % PER_URL_TEXT_CAP)


def test_byte_cap_aborts_stream():
    consumed = {"bytes": 0}
    from httpx._content import AsyncIteratorByteStream  # lazy chunked stream

    def handler(request):
        async def gen():
            chunk = b"x" * 100_000
            for _ in range(30):  # 3 MB total
                consumed["bytes"] += len(chunk)
                yield chunk
        return httpx.Response(200, stream=AsyncIteratorByteStream(gen()), request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    res = asyncio.run(wr.WebResearch(client=client)._extract_article("https://example.com/big"))
    assert "too large" in res.error, res.error
    assert consumed["bytes"] < 30 * 100_000, f"stream not aborted: {consumed['bytes']}"
    print(f"PASS  byte cap aborts stream after {consumed['bytes']} bytes (< 3MB)")


# ─── ranking ─────────────────────────────────────────────────────────────


def test_freshness_ranking():
    now = datetime.now(UTC)

    def hit(title, dt, rank):
        return SourceHit(url=f"https://x.test/{rank}", title=title, published=dt, rank=rank)

    recent = hit("Recent AI regulation story", now - timedelta(hours=2), 1)
    older = hit("Older AI regulation story", now - timedelta(days=20), 2)
    undated1 = SourceHit(url="https://x.test/u1", title="Undated one", rank=3)
    undated2 = SourceHit(url="https://x.test/u2", title="Undated two", rank=4)
    gnews_old = hit("Two-day old headline", now - timedelta(hours=FRESH_WINDOW_HOURS + 1), 5)
    hits = [older, undated2, recent, gnews_old, undated1]

    latest = _rank_hits(hits, "latest", now)
    assert [h.title for h in latest] == [
        "Recent AI regulation story", "Undated one", "Undated two",
        "Two-day old headline", "Older AI regulation story",
    ], [h.title for h in latest]

    auto = _rank_hits(hits, "auto", now)
    assert [h.title for h in auto] == [
        "Recent AI regulation story", "Two-day old headline",
        "Older AI regulation story", "Undated one", "Undated two",
    ], [h.title for h in auto]

    anytime = _rank_hits(hits, "anytime", now)
    assert [h.title for h in anytime] == [
        "Recent AI regulation story", "Older AI regulation story",
        "Undated one", "Undated two", "Two-day old headline",
    ], [h.title for h in anytime]
    print("PASS  freshness ranking (latest/auto/anytime + undated fill)")


# ─── brief budget ────────────────────────────────────────────────────────


def test_format_brief_budget():
    now = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
    pool, extracts = [], {}
    long_text = ("The European Union approved landmark artificial intelligence "
                 "regulation on Tuesday setting binding rules for high-risk systems. ") * 40
    for i in range(9):  # max_sources=8 -> 9 pool entries (worst case)
        h = SourceHit(
            url=f"https://example.com/{i}",
            title=f"Worst case research headline number {i} about artificial intelligence regulation",
            source="S", published=now - timedelta(hours=i), rank=i,
        )
        pool.append(h)
        extracts[h.url] = ExtractResult(url=h.url, title=h.title, text=long_text)
    latest = [(h.published, h.title, h.source) for h in pool[:8]]
    failures = {"BBC": "timeout", "gnews": "HTTP 429"}
    summary = _summarize([e.text for e in extracts.values()], "AI regulation")
    assert len(summary) <= 800
    out = _format_brief("AI regulation", "auto", pool, extracts, summary, latest, failures, now)
    assert len(out) <= BRIEF_CHAR_BUDGET, len(out)
    assert "🔎 RESEARCH: AI regulation" in out
    assert "━ SOURCES ━" in out  # head sections survive; tail is truncated by design
    assert "BBC failed: timeout" in out or "brief truncated" in out

    # small case: everything fits, all sections present
    small_pool = pool[:2]
    small_extracts = {h.url: extracts[h.url] for h in small_pool}
    small = _format_brief("AI regulation", "auto", small_pool, small_extracts,
                          summary, latest[:2], failures, now)
    assert len(small) <= BRIEF_CHAR_BUDGET
    for sec in ("━ SOURCES ━", "━ SUMMARY ━", "━ LATEST UPDATES ━", "━ NOTES ━"):
        assert sec in small, f"missing {sec}"
    assert "source BBC failed: timeout" in small
    print(f"PASS  _format_brief <= {BRIEF_CHAR_BUDGET} chars worst case (got {len(out)}), sections complete when it fits")


# ─── all-fail friendly string ────────────────────────────────────────────


def test_all_fail_friendly():
    def handler(request):
        return httpx.Response(500, text="boom", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)
    out = asyncio.run(wr.WebResearch(client=client).run("any topic at all"))
    assert out.startswith("⚠ Research failed: all 5 sources errored"), out
    assert "500" in out
    assert "ddg" in out and "gnews" in out and "BBC" in out
    print("PASS  all-sources-fail friendly string, never raises")


# ─── mocked end-to-end pipeline ──────────────────────────────────────────


def test_mocked_pipeline():
    now = datetime.now(UTC)
    gnews_dt1 = now - timedelta(hours=5)
    gnews_dt2 = now - timedelta(hours=26)
    guard_dt = now - timedelta(hours=3, minutes=5)
    nyt_dt = now - timedelta(hours=6, minutes=10)

    DDG_HTML = """
    <html><body>
    <div class="result">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle1%3Futm_source%3Dddg&amp;rut=abc1">EU passes landmark AI regulation act</a>
      <a class="result__snippet" href="x">The European Union approved new rules for high-risk artificial intelligence systems on Tuesday.</a>
    </div>
    <div class="result">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle2&amp;rut=abc2">AI regulation bill advances in US Senate</a>
      <a class="result__snippet" href="x">Lawmakers moved the artificial intelligence regulation bill one step closer to a vote.</a>
    </div>
    <div class="result">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle3&amp;rut=abc3">Startups brace for AI regulation compliance costs</a>
      <a class="result__snippet" href="x">Small firms say the new AI regulation rules will raise costs.</a>
    </div>
    </body></html>
    """

    GNEWS_RSS = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Google News</title>
    <item><title>AI regulation bill advances in US Senate - The Verge</title>
      <link>https://news.google.com/rss/articles/CBMiXYZ</link>
      <pubDate>{_rfc(gnews_dt1)}</pubDate>
      <description>&lt;p&gt;The US Senate committee advanced the artificial intelligence regulation bill.&lt;/p&gt;</description></item>
    <item><title>Global AI safety summit set for October - Reuters</title>
      <link>https://news.google.com/rss/articles/CBMiABC</link>
      <pubDate>{_rfc(gnews_dt2)}</pubDate>
      <description>Organizers announced the October summit on artificial intelligence safety.</description></item>
    </channel></rss>"""

    GUARDIAN_RSS = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
    <item><title>AI regulation: what the new EU law means for business</title>
      <link>https://www.theguardian.com/world/2026/aug/05/ai-regulation</link>
      <pubDate>{_rfc(guard_dt)}</pubDate>
      <description>The EU's artificial intelligence regulation enters its enforcement phase.</description></item>
    </channel></rss>"""

    NYT_RSS = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
    <item><title>AI regulation enforcement begins for tech giants</title>
      <link>https://www.nytimes.com/2026/08/05/technology/ai-regulation.html</link>
      <pubDate>{_rfc(nyt_dt)}</pubDate>
      <description>Big technology companies begin complying with the new artificial intelligence regulation.</description></item>
    </channel></rss>"""

    ARTICLE_HTML = """<html><head><title>EU passes landmark AI regulation act</title></head>
    <body>
    <nav>Home World Tech More</nav>
    <header>Site Masthead</header>
    <script>var x = 1;</script>
    <article>
    <h1>EU passes landmark AI regulation act</h1>
    <p>The European Union approved landmark artificial intelligence regulation on Tuesday, setting binding rules for high-risk systems across the bloc.</p>
    <p>Regulators said the new law balances innovation with safety and gives companies two years to comply with the requirements.</p>
    <p>Industry groups welcomed the legal clarity but warned that compliance costs could hurt smaller startups in the sector.</p>
    </article>
    <footer>Copyright 2026 Example News</footer>
    </body></html>"""

    class Router:
        def __init__(self):
            self.paths = []

        def handler(self, request):
            url = str(request.url)
            self.paths.append(url)
            if url.startswith("https://html.duckduckgo.com/"):
                return httpx.Response(200, text=DDG_HTML, request=request)
            if url.startswith("https://news.google.com/rss/"):
                return httpx.Response(200, text=GNEWS_RSS, request=request)
            if "feeds.bbci.co.uk" in url:
                return httpx.Response(500, text="boom", request=request)  # one feed down
            if "theguardian.com" in url:
                if "/rss" in url:
                    return httpx.Response(200, text=GUARDIAN_RSS, request=request)
                return httpx.Response(200, text=ARTICLE_HTML, request=request)
            if "nytimes.com" in url:
                if "/rss" in url:
                    return httpx.Response(200, text=NYT_RSS, request=request)
                return httpx.Response(500, text="nope", request=request)  # article extraction fails
            if "example.com" in url:
                return httpx.Response(200, text=ARTICLE_HTML, request=request)
            return httpx.Response(404, text="nothing", request=request)

    router = Router()
    client = httpx.AsyncClient(transport=httpx.MockTransport(router.handler), timeout=10.0)
    out = asyncio.run(wr.WebResearch(client=client).run(
        "latest AI regulation news", max_sources=3, freshness="latest"
    ))

    assert "🔎 RESEARCH: latest AI regulation news  (freshness=latest)" in out
    for sec in ("━ SOURCES ━", "━ SUMMARY ━", "━ LATEST UPDATES ━", "━ NOTES ━"):
        assert sec in out, f"missing section {sec}"
    # uddg redirect decoded in every shown URL
    assert "//duckduckgo.com/l/" not in out, out
    assert "https://example.com/article1" in out
    # NYT article extraction failure is isolated and reported
    assert "⚠ extraction failed" in out
    assert "500" in out
    # 500 feed (BBC) isolated; other sources survived
    assert "Guardian" in out
    assert "source BBC failed" in out
    assert "all discovery sources responded OK" not in out
    # key points extracted (bullets) and dated/undated source dates
    assert "   • " in out
    assert "Date: unknown" in out
    assert "ago" in out
    # GNews rss/articles stubs never fetched (headline-only)
    assert not any("/rss/articles/" in p for p in router.paths), router.paths
    # latest section populated with dated GNews headline
    assert "AI regulation bill advances in US Senate" in out
    assert "Google News" in out
    print("PASS  mocked end-to-end pipeline (5 sources, uddg decoded, failures isolated, gnews headline-only)")


if __name__ == "__main__":
    test_parse_rss_datetime()
    test_decode_ddg_url()
    test_normalize_url_and_title()
    test_dedupe()
    test_html_to_text_prefers_article()
    test_html_to_text_no_article()
    test_strip_boilerplate()
    test_text_cap()
    test_byte_cap_aborts_stream()
    test_freshness_ranking()
    test_format_brief_budget()
    test_all_fail_friendly()
    test_mocked_pipeline()
    print("\nAll web_research tests passed.")
