#!/usr/bin/env python3
"""Web tools — search and extract content from the web."""
from __future__ import annotations

import json
import re
from urllib.parse import quote_plus

import httpx

from tools import BaseTool

_http = httpx.AsyncClient(timeout=30.0, follow_redirects=True)


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
                url = match.group(1)
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
        try:
            resp = await _http.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml",
            })
            if resp.status_code != 200:
                return f"⚠ Failed to fetch: HTTP {resp.status_code}"

            html = resp.text

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
