#!/usr/bin/env python3
"""Web tools — search and extract content from the web."""
from __future__ import annotations

import html as html_mod
import ipaddress
import re
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

from tools import BaseTool

_http = httpx.AsyncClient(timeout=30.0, follow_redirects=False)

_REDIRECT_STATUSES = (301, 302, 303, 307, 308)


async def fetch_safe(url: str, headers: dict | None = None, max_hops: int = 5, client: httpx.AsyncClient | None = None):
    """Fetch a URL with SSRF re-check on EVERY redirect hop.

    ``follow_redirects=True`` only validates the INITIAL URL — a Location
    header can then point at 127.0.0.1 / 169.254.169.254 unchecked.
    We walk the redirect chain ourselves and validate each hop.
    Returns (httpx.Response, None) or (None, error_string).
    """
    http = client or _http
    current = url
    for _ in range(max_hops + 1):
        if not is_safe_url(current):
            return None, "⚠ Refusing to fetch non-public URL (blocked private/loopback host)"
        req = http.build_request("GET", current, headers=headers)
        resp = await http.send(req, stream=True)
        if resp.status_code in _REDIRECT_STATUSES:
            location = resp.headers.get("location")
            await resp.aclose()
            if not location:
                return None, f"⚠ Failed to fetch: HTTP {resp.status_code}"
            current = str(httpx.URL(current).join(location))
            continue
        return resp, None
    return None, "⚠ Too many redirects"


def is_safe_url(url: str) -> bool:
    """SSRF guard: only http(s) to PUBLIC hosts.

    Blocks loopback, link-local, multicast, CGNAT (Tailscale 100.64/10)
    and RFC1918 private ranges. Hostnames are allowed (DNS-rebinding is
    accepted residual risk for a personal bridge); IP literals in blocked
    ranges are rejected outright. Also rejects localhost/*.local names."""
    try:
        u = urlparse(url)
        if u.scheme not in ("http", "https"):
            return False
        host = (u.hostname or "").lower()
        if not host:
            return False
        if host == "localhost" or host.endswith(".local") or host.endswith(".localdomain"):
            return False
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return True  # hostname — allow
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            return False
        if ip.is_private or ip.is_reserved:
            return False
        # Tailscale CGNAT range (also private in modern Python — belt & braces)
        if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):
            return False
        return True
    except Exception:
        return False


def _decode_ddg_url(href: str) -> str:
    """Decode a DuckDuckGo /l/?uddg= redirect link into the real target URL.

    DDG HTML results link to //duckduckgo.com/l/?uddg=<urlencoded-url>&rut=...
    Fetching that /l/ endpoint returns HTTP 400, so callers must decode the
    uddg param. Falls back to the raw href when uddg is absent or malformed.
    """
    href = html_mod.unescape(href.strip())
    if "uddg=" not in href:
        return href
    try:
        qs = parse_qs(urlparse(href).query)
        if qs.get("uddg"):
            decoded = unquote(qs["uddg"][0])
            if decoded.startswith(("http://", "https://")):
                return decoded
    except Exception:
        pass
    return href


class WebSearch(BaseTool):
    name = "web_search"
    description = "Search the web for information. Returns up to 10 results with titles, URLs, and descriptions."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "limit": {
                "type": "integer",
                "description": "Number of results (default 5, max 10)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def run(self, query: str, limit: int = 5) -> str:
        limit = min(limit, 10)
        try:
            # Use DuckDuckGo's HTML API (no API key needed)
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            resp = await _http.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36",
            })
            if resp.status_code != 200:
                return f"⚠ Search failed: HTTP {resp.status_code}"

            # Parse the HTML response for result links
            html = resp.text
            results = []
            # Simple regex-based extraction of DDG results
            for match in re.finditer(
                r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>',
                html, re.DOTALL
            ):
                url = _decode_ddg_url(match.group(1))
                title = re.sub(r'<.*?>', '', match.group(2)).strip()
                # Find snippet after this anchor
                snippet_match = re.search(
                    r'<a class="result__snippet".*?>(.*?)</a>',
                    html[match.end():], re.DOTALL
                )
                snippet = ""
                if snippet_match:
                    snippet = re.sub(r'<.*?>', '', snippet_match.group(1)).strip()
                results.append(f"{title}\n  {url}\n  {snippet[:200]}")
                if len(results) >= limit:
                    break

            if not results:
                return "No results found."
            return "\n\n".join(results)

        except Exception as e:
            return f"⚠ Search failed: {e}"


class WebExtract(BaseTool):
    name = "web_extract"
    description = "Extract readable content from a web page URL. Returns text in markdown format."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to extract content from"
            },
        },
        "required": ["url"],
    }

    async def run(self, url: str) -> str:
        if not is_safe_url(url):
            return "⚠ Refusing to fetch non-public URL (blocked private/loopback host)"
        try:
            resp, err = await fetch_safe(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            if err:
                return err
            assert resp is not None  # err == None implies a response
            try:
                if resp.status_code != 200:
                    return f"⚠ Failed to fetch: HTTP {resp.status_code}"
                # Byte-cap the download (1.5 MB) — a huge page must not
                # flood memory.
                chunks = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > 1_500_000:
                        chunks.append(b"[content too large - truncated]")
                        break
                    chunks.append(chunk)
                html = b"".join(chunks).decode("utf-8", errors="replace")
            finally:
                await resp.aclose()

            # Strip HTML tags for basic readability
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            # Truncate to reasonable size
            if len(text) > 15000:
                text = text[:15000] + "\n... [content truncated]"

            return text

        except Exception as e:
            return f"⚠ Extract failed: {e}"
