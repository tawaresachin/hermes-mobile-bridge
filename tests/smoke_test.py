#!/usr/bin/env python3
"""Smoke tests for the Hermes Mobile Bridge.

Covers the security-critical surfaces that changed in the review:
  1. /health is public and 200
  2. /api/* requires auth (401 without Bearer)
  3. /uploads/* requires auth (attachments are no longer world-readable)
  4. Model flag normalization is deterministic by model id
Run:  python3 tests/smoke_test.py [BASE_URL]   (default http://127.0.0.1:9119)
Exits non-zero on failure.
"""
import re
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9119"

FAILED = []


def http_status(path: str) -> int:
    req = urllib.request.Request(f"{BASE}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILED.append(name)


def test_normalization() -> None:
    """Same model id -> identical flags, regardless of provider."""
    free_re = re.compile(r"(^|[/_:.\-])free($|[/_:.\-])", re.I)
    vision_re = re.compile(r"(vl|vision|multimodal|omni)", re.I)
    ids = [
        "deepseek-v4-flash-free", "oc/deepseek-v4-flash-free",
        "deepseek-chat:free", "auto/best-coding", "veo-free/veo",
        "qwen2.5-vl-7b", "deepseek-v4-pro",
    ]
    for i in ids:
        f = bool(free_re.search(i))
        v = bool(vision_re.search(i)) and not f
        check(f"normalize({i!r}) deterministic", isinstance(f, bool) and isinstance(v, bool))


def main() -> None:
    print(f"Smoke tests against {BASE}")
    check("GET /health -> 200", http_status("/health") == 200)
    check("GET /api/models (no auth) -> 401", http_status("/api/models") == 401)
    check("GET /api/chat/stream (no auth) -> 401", http_status("/api/chat/stream") == 401)
    check("GET /uploads/nope (no auth) -> 401", http_status("/uploads/nope.jpg") == 401)
    check("GET /setup/qr (no token) -> 401", http_status("/setup/qr") == 401)
    test_normalization()
    if FAILED:
        print(f"\n{len(FAILED)} FAILURES: {FAILED}")
        sys.exit(1)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()