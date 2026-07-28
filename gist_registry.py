#!/usr/bin/env python3
"""
Gist-based discovery registry.
Bridge server writes its tunnel URL to a GitHub Gist,
mobile app reads it. No server deployment needed.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import httpx

# ─── Config ─────────────────────────────────────────────────────────────

STORE_PATH = Path(os.getenv("STORE_PATH", os.path.expanduser("~/.hermes-bridge")))
STORE_PATH.mkdir(parents=True, exist_ok=True)

IDENTITY_FILE = STORE_PATH / "identity.json"

# The gist ID is generated once and stored in identity.json
# GIST_TOKEN comes from env (GitHub Personal Access Token with gist scope)
GIST_TOKEN = os.getenv("GIST_TOKEN", "")

# ─── Identity (same as discovery.py) ────────────────────────────────────


class BridgeIdentity:
    def __init__(self):
        self.device_id: str = ""
        self.email: str = ""
        self.device_name: str = ""
        self.gist_id: str = ""  # GitHub Gist ID for URL sharing
        self._loaded = False

    def load_or_create(self) -> "BridgeIdentity":
        if IDENTITY_FILE.exists():
            data = json.loads(IDENTITY_FILE.read_text())
            self.device_id = data.get("device_id", "")
            self.email = data.get("email", "")
            self.device_name = data.get("device_name", "")
            self.gist_id = data.get("gist_id", "")
            self._loaded = True
        return self

    def create_new(self, email: str, gist_id: str = "", device_name: str = "") -> "BridgeIdentity":
        import secrets
        import socket
        self.device_id = secrets.token_hex(16)
        self.email = email.strip().lower()
        self.device_name = device_name or socket.gethostname() or "Hermes Bridge"
        self.gist_id = gist_id
        self._save()
        self._loaded = True
        return self

    def is_loaded(self) -> bool:
        return self._loaded

    def _save(self):
        IDENTITY_FILE.write_text(
            json.dumps({
                "device_id": self.device_id,
                "email": self.email,
                "device_name": self.device_name,
                "gist_id": self.gist_id,
            }, indent=2)
        )
        IDENTITY_FILE.chmod(0o600)


# ─── Gist Registry Client ──────────────────────────────────────────────


class GistRegistry:
    """
    Uses a GitHub Gist to share the bridge URL.
    The gist contains a single file `hermes-bridge-url.json` with:
      { "url": "https://...", "updated_at": 1234567890 }
    The mobile app reads this gist via the public raw URL (no auth needed).
    """

    GIST_FILENAME = "hermes-bridge-url.json"
    GIST_DESCRIPTION = "Hermes Bridge URL — auto-updated"

    def __init__(self, identity: BridgeIdentity):
        self.identity = identity
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(15.0))

    async def ensure_gist(self) -> bool:
        """
        Create or verify the GitHub Gist exists.
        Returns True if gist is ready.
        """
        if not GIST_TOKEN:
            print("⚠ GIST_TOKEN not set. Set it to a GitHub PAT with 'gist' scope.")
            return False

        if self.identity.gist_id:
            # Verify old gist still exists
            resp = await self._http.get(
                f"https://api.github.com/gists/{self.identity.gist_id}",
                headers={"Authorization": f"Bearer {GIST_TOKEN}"},
            )
            if resp.status_code == 200:
                return True
            print(f"⚠ Gist {self.identity.gist_id} not found, creating new one...")

        # Create the gist
        resp = await self._http.post(
            "https://api.github.com/gists",
            headers={
                "Authorization": f"Bearer {GIST_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "description": self.GIST_DESCRIPTION,
                "public": False,
                "files": {
                    self.GIST_FILENAME: {
                        "content": json.dumps({
                            "url": None,
                            "email": self.identity.email,
                            "updated_at": int(time.time()),
                            "device_name": self.identity.device_name,
                        })
                    }
                },
            },
        )
        if resp.status_code not in (201, 200):
            print(f"✗ Failed to create gist: {resp.status_code} {resp.text[:200]}")
            return False

        data = resp.json()
        self.identity.gist_id = data["id"]
        self.identity._save()
        print(f"   Created gist: {data['html_url']}")
        return True

    async def update_url(self, tunnel_url: str) -> bool:
        """Update the gist with the current tunnel URL."""
        if not self.identity.gist_id or not GIST_TOKEN:
            return False

        resp = await self._http.patch(
            f"https://api.github.com/gists/{self.identity.gist_id}",
            headers={
                "Authorization": f"Bearer {GIST_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "files": {
                    self.GIST_FILENAME: {
                        "content": json.dumps({
                            "url": tunnel_url,
                            "email": self.identity.email,
                            "updated_at": int(time.time()),
                            "device_name": self.identity.device_name,
                        })
                    }
                }
            },
        )
        return resp.status_code in (200, 204)

    def get_discover_url(self) -> Optional[str]:
        """Return the public URL the mobile app can read to discover the bridge."""
        if not self.identity.gist_id:
            return None
        return f"https://gist.githubusercontent.com/raw/{self.identity.gist_id}/{self.GIST_FILENAME}"

    @staticmethod
    async def discover(gist_id_or_url: str) -> Optional[dict]:
        """Read a bridge URL from a gist. Can accept a gist ID or full raw URL."""
        if gist_id_or_url.startswith("http"):
            url = gist_id_or_url
        else:
            url = f"https://gist.githubusercontent.com/raw/{gist_id_or_url}/{GistRegistry.GIST_FILENAME}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
        return None

    async def close(self):
        await self._http.aclose()
