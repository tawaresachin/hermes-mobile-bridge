#!/usr/bin/env python3
"""
Hermes Bridge Registry — Cloudflare Worker
Maps user emails to their current bridge tunnel URL.

Endpoints:
  POST /api/v1/register     — one-time bridge registration
  POST /api/v1/heartbeat    — periodic URL update (every 3 min)
  GET  /api/v1/discover/:email — app discovers bridge URL
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

import httpx

# ─── Config ─────────────────────────────────────────────────────────────

REGISTRY_URL = os.getenv("HERMES_REGISTRY_URL", "https://hermes-bridge-registry.nousresearch.workers.dev")
REGISTRY_API = f"{REGISTRY_URL}/api/v1"

STORE_PATH = Path(os.getenv("STORE_PATH", os.path.expanduser("~/.hermes-bridge")))
STORE_PATH.mkdir(parents=True, exist_ok=True)

IDENTITY_FILE = STORE_PATH / "identity.json"

# ─── Identity Management ────────────────────────────────────────────────


class BridgeIdentity:
    """Persistent device identity for this bridge server."""

    def __init__(self):
        self.device_id: str = ""
        self.device_secret: str = ""
        self.email: str = ""
        self.device_name: str = ""
        self._loaded = False

    def load_or_create(self) -> "BridgeIdentity":
        """Load existing identity or create a new one."""
        if IDENTITY_FILE.exists():
            data = json.loads(IDENTITY_FILE.read_text())
            self.device_id = data.get("device_id", "")
            self.device_secret = data.get("device_secret", "")
            self.email = data.get("email", "")
            self.device_name = data.get("device_name", "")
            self._loaded = True
        return self

    def create_new(self, email: str, device_name: str = "") -> "BridgeIdentity":
        """Generate a new identity for first-time setup."""
        self.device_id = self._generate_device_id()
        self.device_secret = self._generate_device_secret()
        self.email = email.strip().lower()
        self.device_name = device_name or self._default_device_name()
        self._save()
        self._loaded = True
        return self

    def is_loaded(self) -> bool:
        return self._loaded

    def _save(self) -> None:
        IDENTITY_FILE.write_text(
            json.dumps(
                {
                    "device_id": self.device_id,
                    "device_secret": self.device_secret,
                    "email": self.email,
                    "device_name": self.device_name,
                },
                indent=2,
            )
        )
        IDENTITY_FILE.chmod(0o600)

    @staticmethod
    def _generate_device_id() -> str:
        return secrets.token_hex(16)  # 32 hex chars

    @staticmethod
    def _generate_device_secret() -> str:
        return secrets.token_urlsafe(32)  # 43 chars, 256 bits

    @staticmethod
    def _default_device_name() -> str:
        import socket
        return socket.gethostname() or "Hermes Bridge"


# ─── Registry Client ────────────────────────────────────────────────────


class RegistryClient:
    """HTTP client for communicating with the central registry."""

    def __init__(self, identity: BridgeIdentity):
        self.identity = identity
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(15.0))

    async def register(self) -> bool:
        """
        Register this bridge with the central registry.
        Called once during 'hermes-bridge init'.
        Returns True on success, False if already registered.
        """
        if not self.identity.is_loaded():
            raise RuntimeError("Identity not loaded. Create one first.")

        try:
            resp = await self._http.post(
                f"{REGISTRY_API}/register",
                json={
                    "email": self.identity.email,
                    "device_id": self.identity.device_id,
                    "device_secret": self.identity.device_secret,
                    "device_name": self.identity.device_name,
                },
            )
            if resp.status_code == 200:
                return True
            elif resp.status_code == 409:
                print("⚠ Already registered. Running heartbeat...")
                return True
            else:
                print(f"✗ Registration failed: {resp.status_code} {resp.text[:200]}")
                return False
        except Exception as e:
            print(f"✗ Registry unreachable: {e}")
            return False

    async def heartbeat(self, tunnel_url: str, platform: str = "", version: str = "") -> bool:
        """
        Update the registry with current tunnel URL.
        Called every 3 minutes while the bridge is running.
        """
        if not self.identity.is_loaded():
            return False

        try:
            resp = await self._http.post(
                f"{REGISTRY_API}/heartbeat",
                json={
                    "device_id": self.identity.device_id,
                    "device_secret": self.identity.device_secret,
                    "tunnel_url": tunnel_url,
                    "platform": platform or __import__("sys").platform,
                    "version": version or "2.0.0",
                    "device_name": self.identity.device_name,
                },
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def discover(self, email: str) -> Optional[dict]:
        """Discover the bridge URL for a given email."""
        try:
            email_encoded = email.strip().lower()
            resp = await self._http.get(f"{REGISTRY_API}/discover/{email_encoded}")
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception:
            return None

    async def close(self):
        await self._http.aclose()


# ─── Self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import tempfile

    print("Testing Bridge Identity...")
    with tempfile.TemporaryDirectory() as tmp:
        # Patch STORE_PATH for testing
        import discovery
        discovery.STORE_PATH = Path(tmp)
        discovery.IDENTITY_FILE = Path(tmp) / "identity.json"

        identity = discovery.BridgeIdentity()
        identity.create_new("test@example.com", "Test Device")
        assert identity.is_loaded()
        assert identity.device_id
        assert len(identity.device_secret) > 20
        print(f"  Device ID: {identity.device_id[:16]}...")
        print(f"  Email: {identity.email}")
        print(f"  Name: {identity.device_name}")

        # Test persistence
        identity2 = discovery.BridgeIdentity()
        identity2.load_or_create()
        assert identity2.device_id == identity.device_id
        assert identity2.device_secret == identity.device_secret
        print("  Persistence: OK")

    print("✅ All tests passed!")
