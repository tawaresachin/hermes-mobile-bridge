"""
Hermes Mobile Bridge Plugin
============================
REST/SSE server for the Hermes Mobile Android app.
Integrates with Hermes config, OmniRoute, and runs the same agent loop.

Install:  hermes plugins install tawaresachin/hermes-mobile-bridge
Start:    hermes mobile-serve
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import secrets
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── CLI command handler ──────────────────────────────────────────────


def _setup_parser(subparser) -> None:
    """Add mobile-serve arguments to the subparser."""
    subparser.add_argument(
        "--port", type=int, default=int(os.getenv("PORT", "9119")),
        help="Port to bind (default: 9119, env: PORT)",
    )
    subparser.add_argument(
        "--host", type=str, default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind (default: 0.0.0.0, env: HOST)",
    )
    subparser.add_argument(
        "--no-omniroute", action="store_true",
        help="Skip OmniRoute auto-lifecycle management",
    )
    subparser.add_argument(
        "--tunnel", action="store_true",
        help="Auto-start a Cloudflare tunnel",
    )


def _start_server(port: int, host: str, omniroute: bool, tunnel: bool) -> None:
    """Start the Hermes Mobile Bridge server."""
    bridge_dir = Path(__file__).resolve().parent.parent.parent.parent / "hermes-mobile-server"
    if not bridge_dir.exists():
        bridge_dir = Path.home() / "hermes-mobile-server"
    if not bridge_dir.exists():
        print("⚠ Bridge server not found at ~/hermes-mobile-server/")
        print("   Clone it: git clone https://github.com/tawaresachin/hermes-mobile-bridge.git")
        return

    env = os.environ.copy()
    env["AI_BASE_URL"] = env.get("AI_BASE_URL", "http://localhost:20128/v1")
    env["AI_MODEL"] = env.get("AI_MODEL", "auto/best-coding")
    env["HOST"] = host
    env["PORT"] = str(port)

    print(f"🤖 Hermes Mobile Bridge starting on http://{host}:{port}")

    if omniroute:
        _ensure_omnirouter()
    ts_ip = _ensure_tailscale()

    if ts_ip:
        print(f"   🖧 Tailscale: {ts_ip}")
        env["HERMES_TAILSCALE_IP"] = ts_ip

    print(f"   AI: {env['AI_MODEL']} @ {env['AI_BASE_URL']}")
    print(f"   Plugin: ~/.hermes/plugins/mobile-bridge/")

    subprocess.run(
        ["/data/data/com.termux/files/usr/bin/python3", "-B", "server.py"],
        cwd=str(bridge_dir),
        env=env,
    )


def _ensure_omnirouter() -> None:
    """Auto-lifecycle for OmniRoute: install -> start -> configure."""
    or_url = "http://localhost:20128/v1"
    or_key = os.environ.get("HERMES_CUSTOM_LOCALHOST_20128_API_KEY", "") or \
             os.environ.get("OMNIROUTE_API_KEY", "")

    or_bin = shutil.which("omniroute")
    if not or_bin:
        logger.info("OmniRoute not found — installing via npm...")
        try:
            subprocess.run(["npm", "install", "-g", "omniroute"],
                           capture_output=True, timeout=120)
            or_bin = shutil.which("omniroute")
        except Exception as e:
            logger.warning("OmniRoute install failed: %s", e)

    if not or_bin:
        logger.warning("OmniRoute not available — continuing without it")
        return

    running = False
    try:
        req = urllib.request.Request(f"{or_url}/models",
                                     headers={"User-Agent": "HermesBridge/2.0"})
        with urllib.request.urlopen(req, timeout=3):
            running = True
            logger.info("OmniRoute server already running")
    except Exception:
        pass

    if not running:
        logger.info("Starting OmniRoute server...")
        try:
            proc = subprocess.Popen([or_bin, "serve"],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    start_new_session=True)
            for attempt in range(20):
                time.sleep(1)
                try:
                    req = urllib.request.Request(f"{or_url}/models",
                                                 headers={"User-Agent": "HermesBridge/2.0"})
                    with urllib.request.urlopen(req, timeout=2):
                        running = True
                        logger.info("OmniRoute ready after %ds", attempt + 1)
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.warning("Failed to start OmniRoute: %s", e)

    if not or_key and running:
        or_key = f"sk-{secrets.token_hex(16)}"
        os.environ["HERMES_CUSTOM_LOCALHOST_20128_API_KEY"] = or_key

    if not os.environ.get("AI_BASE_URL"):
        os.environ["AI_BASE_URL"] = or_url
    if not os.environ.get("AI_MODEL"):
        os.environ["AI_MODEL"] = "auto/best-coding"
    if or_key:
        os.environ["AI_API_KEY"] = or_key

    logger.info("OmniRoute: %s @ %s (key: %s...%s)",
                os.environ.get("AI_MODEL", "?"),
                os.environ.get("AI_BASE_URL", "?"),
                or_key[:8] if or_key else "none", or_key[-4:] if or_key else "")


def _ensure_tailscale() -> str:
    """Ensure Tailscale on the bridge device: detect → auto-install APK → guide.
    Returns the Tailscale IP or empty string."""
    ts_bin = shutil.which("tailscale")
    ts_ip = ""

    # 1. Detect Tailscale IP from network interfaces (Android app mode — no CLI needed)
    ts_ip = _detect_tailscale_ip()
    if ts_ip:
        return ts_ip

    # 2. Try CLI
    if ts_bin:
        try:
            result = subprocess.run(
                [ts_bin, "ip", "-4"], capture_output=True, text=True, timeout=5,
            )
            ip = result.stdout.strip()
            if ip and ip.startswith("100."):
                logger.info("Tailscale: %s", ip)
                return ip
        except Exception:
            pass

    # 3. Auto-install the Tailscale Android app (APK) if missing
    if not _is_tailscale_app_installed():
        _auto_install_tailscale_app()

    # 4. Re-check after install attempt
    ts_ip = _detect_tailscale_ip()
    if ts_ip:
        logger.info("Tailscale active after setup: %s", ts_ip)
        return ts_ip

    # 5. Not available — guide user
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  Tailscale not active                                     ║")
    print("  ║                                                           ║")
    print("  ║  1. Install the Tailscale app (auto-attempted) or from    ║")
    print("  ║     Play Store / F-Droid.                                 ║")
    print("  ║  2. Open the app and sign in (same account as phone).     ║")
    print("  ║  3. Restart 'hermes mobile-serve'.                        ║")
    print("  ║                                                           ║")
    print("  ║  Until then: Cloudflare tunnel is used as fallback.       ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    logger.info("Tailscale not active — using Cloudflare tunnel fallback")

    return ts_ip


def _detect_tailscale_ip() -> str:
    """Look for a 100.x Tailscale IP on any network interface."""
    try:
        result = subprocess.run(
            ["ip", "addr"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "100." in line and "inet " in line:
                match = re.search(r"inet (100\.\d+\.\d+\.\d+)", line)
                if match:
                    return match.group(1)
    except Exception:
        pass
    return ""


def _is_tailscale_app_installed() -> bool:
    """Check if the Tailscale Android app is installed."""
    try:
        result = subprocess.run(
            ["pm", "list", "packages", "com.tailscale.ipn"],
            capture_output=True, text=True, timeout=10,
        )
        return "com.tailscale.ipn" in result.stdout
    except Exception:
        return False


def _auto_install_tailscale_app() -> None:
    """Guide installation of the Tailscale Android app (Play Store / F-Droid)."""
    logger.info("Tailscale app not installed — opening store for one-tap install...")
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  Tailscale app not installed.                            ║")
    print("  ║  Opening Play Store / F-Droid page...                    ║")
    print("  ║  Tap 'Install', then open the app and sign in.           ║")
    print("  ║  Use the SAME account as your app phone.                 ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    # Open Play Store page for Tailscale
    try:
        subprocess.run(
            ["am", "start", "-a", "android.intent.action.VIEW",
             "-d", "market://details?id=com.tailscale.ipn"],
            capture_output=True, text=True, timeout=10,
        )
        logger.info("Opened Play Store for com.tailscale.ipn")
    except Exception as e:
        logger.warning("Could not open Play Store: %s", e)


# ─── Plugin entry point ──────────────────────────────────────────────


def register(ctx) -> None:
    """Register the mobile-bridge plugin with Hermes."""
    ctx.register_cli_command(
        name="mobile-serve",
        help="Start the Hermes Mobile REST/SSE server for Android app",
        description=(
            "Starts the backend server that the Hermes Mobile Android app "
            "connects to. Provides chat, model switching, file upload, and "
            "tool execution via REST/SSE. Auto-integrates with OmniRoute "
            "for free token-free auto-routing."
        ),
        setup_fn=_setup_parser,
        handler_fn=_handle_mobile_serve,
    )
    logger.info("mobile-bridge plugin registered: hermes mobile-serve")


def _handle_mobile_serve(args: argparse.Namespace) -> None:
    """Handle `hermes mobile-serve` CLI command."""
    _start_server(
        port=args.port,
        host=args.host,
        omniroute=not args.no_omniroute,
        tunnel=args.tunnel,
    )
