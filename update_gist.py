#!/usr/bin/env python3
"""
Update the GitHub Gist registry with the current tunnel URL.
Call this from a cron job or after tunnel restart: python3 update_gist.py <tunnel_url>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

GIST_TOKEN = os.getenv("GIST_TOKEN", "")
STORE_PATH = Path(os.getenv("STORE_PATH", os.path.expanduser("~/.hermes-bridge")))

GIST_FILENAME = "hermes-bridge-url.json"


def get_gist_id() -> str:
    """Read gist_id from identity file."""
    identity_file = STORE_PATH / "identity.json"
    if identity_file.exists():
        data = json.loads(identity_file.read_text())
        return data.get("gist_id", "")
    return ""


def get_tunnel_url() -> str:
    """Read tunnel URL from cloudflared log or env."""
    url = os.getenv("HERMES_TUNNEL_URL", "")
    if url:
        return url
    # Try to detect from cloudflared log (store lives under STORE_PATH on
    # every OS; TUNNEL_LOG env overrides for custom setups)
    url_file = Path(os.getenv("TUNNEL_LOG", str(Path.home() / ".hermes-mobile-server" / ".current_tunnel_url")))
    if url_file.exists():
        import re
        content = url_file.read_text()
        match = re.search(r'https://[a-z0-9.-]+\.trycloudflare\.com', content)
        if match:
            return match.group(0)
    return ""


async def update_gist(tunnel_url: str) -> bool:
    """Update the gist with the current URL."""
    gist_id = get_gist_id()
    if not gist_id or not GIST_TOKEN:
        print("Missing gist_id or GIST_TOKEN")
        return False

    # Read identity for email/device_name
    identity_file = STORE_PATH / "identity.json"
    email = "unknown"
    device_name = "Hermes Bridge"
    if identity_file.exists():
        data = json.loads(identity_file.read_text())
        email = data.get("email", email)
        device_name = data.get("device_name", device_name)

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={
                "Authorization": f"Bearer {GIST_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "files": {
                    GIST_FILENAME: {
                        "content": json.dumps({
                            "url": tunnel_url,
                            "email": email,
                            "updated_at": int(time.time()),
                            "device_name": device_name,
                            "platform": sys.platform,
                        })
                    }
                }
            },
        )
        if resp.status_code in (200, 204):
            print(f"✓ Gist updated: {tunnel_url}")
            return True
        else:
            print(f"✗ Gist update failed: {resp.status_code} {resp.text[:200]}")
            return False


if __name__ == "__main__":
    import asyncio

    url = sys.argv[1] if len(sys.argv) > 1 else get_tunnel_url()
    if not url:
        print("Usage: python3 update_gist.py <tunnel_url>")
        print("   Or set HERMES_TUNNEL_URL env var")
        sys.exit(1)

    asyncio.run(update_gist(url))
