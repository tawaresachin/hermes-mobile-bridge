#!/usr/bin/env python3
"""
Update the GitHub registry with the current bridge tunnel URL.
Usage:  python3 register_bridge.py https://your-tunnel-url.trycloudflare.com
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO = "tawaresachin/hermes-bridge-registry"
REGISTRY_FILE = "bridges.json"
# Email is read from env or identity file
HERMES_EMAIL = os.getenv("HERMES_EMAIL", "")


def get_email() -> str:
    if HERMES_EMAIL:
        return HERMES_EMAIL
    # Try identity file
    ident = Path(os.getenv("HOME", "~"), ".hermes-bridge", "identity.json")
    if ident.exists():
        try:
            data = json.loads(ident.read_text())
            return data.get("email", "")
        except Exception:
            pass
    # Fallback: ask
    print("Set HERMES_EMAIL env var or run with email as second argument")
    return ""


def main():
    tunnel_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("HERMES_TUNNEL_URL", "")
    email = sys.argv[2] if len(sys.argv) > 2 else get_email()

    if not tunnel_url or not email:
        print("Usage: python3 register_bridge.py <tunnel_url> [email]")
        print("   Or set HERMES_TUNNEL_URL and HERMES_EMAIL env vars")
        sys.exit(1)

    # Clone repo to temp dir
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        result = subprocess.run(
            ["gh", "repo", "clone", REPO, "."],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Failed to clone: {result.stderr}")
            # Try to clone via HTTPS
            result = subprocess.run(
                ["git", "clone", f"https://github.com/{REPO}.git", "."],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                print(f"Git clone failed: {result.stderr}")
                sys.exit(1)

        # Read current registry
        entries = []
        reg_path = Path(tmp) / REGISTRY_FILE
        if reg_path.exists():
            try:
                data = json.loads(reg_path.read_text())
                entries = data.get("entries", [])
            except Exception:
                pass

        # Update or add entry
        email_lower = email.strip().lower()
        found = False
        for entry in entries:
            if entry.get("email", "").strip().lower() == email_lower:
                entry["url"] = tunnel_url
                entry["updated_at"] = int(time.time())
                entry["device_name"] = os.uname().nodename if hasattr(os, 'uname') else "Hermes Bridge"
                found = True
                break

        if not found:
            entries.append({
                "email": email_lower,
                "url": tunnel_url,
                "updated_at": int(time.time()),
                "device_name": os.uname().nodename if hasattr(os, 'uname') else "Hermes Bridge",
            })

        reg_path.write_text(json.dumps({"entries": entries}, indent=2))

        # Commit and push
        subprocess.run(["git", "add", "."], capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", f"Update bridge URL for {email_lower}"],
            capture_output=True, text=True
        )
        result2 = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True, text=True, timeout=30,
        )
        if result2.returncode == 0:
            print(f"✓ Registered: {email_lower} → {tunnel_url}")
        else:
            print(f"✗ Push failed: {result2.stderr}")

    sys.exit(0)


if __name__ == "__main__":
    main()
