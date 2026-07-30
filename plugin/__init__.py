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
import sys
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
    # Import bridge server — it's designed to run as __main__
    # We re-use its config, app, and agent loop
    bridge_dir = Path(__file__).resolve().parent.parent.parent.parent / "hermes-mobile-server"
    if not bridge_dir.exists():
        # Fall back to bundled copy or current directory
        bridge_dir = Path.cwd()

    # Ensure bridge server is on path
    sys.path.insert(0, str(bridge_dir))

    # Apply OmniRoute config if enabled
    if omniroute:
        _ensure_omnirouter()

    # Launch the bridge server as a subprocess
    # (it has its own venv/deps, so we don't import directly)
    import subprocess
    bridge_dir = Path(__file__).resolve().parent.parent.parent.parent / "hermes-mobile-server"

    if not bridge_dir.exists():
        print("⚠ Bridge server not found. Clone it:")
        print(f"   git clone https://github.com/tawaresachin/hermes-mobile-bridge.git {bridge_dir}")
        return

    env = os.environ.copy()
    env["AI_BASE_URL"] = env.get("AI_BASE_URL", "http://localhost:20128/v1")
    env["AI_MODEL"] = env.get("AI_MODEL", "auto/best-coding")
    env["HOST"] = host
    env["PORT"] = str(port)

    print(f"🤖 Hermes Mobile Bridge (plugin mode) starting on http://{host}:{port}")
    print(f"   AI: {env['AI_MODEL']} @ {env['AI_BASE_URL']}")
    print(f"   Plugin: ~/.hermes/plugins/mobile-bridge/")
    print(f"   Bridge: {bridge_dir}/server.py")

    subprocess.run(
        [sys.executable, "-B", "server.py"],
        cwd=str(bridge_dir),
        env=env,
    )


def _ensure_omnirouter() -> None:
    """Auto-lifecycle for OmniRoute: install → start → configure."""
    import secrets
    import shutil
    import subprocess
    import time
    import urllib.request

    or_url = "http://localhost:20128/v1"
    or_key = os.environ.get("HERMES_CUSTOM_LOCALHOST_20128_API_KEY", "")
    if not or_key:
        or_key = os.environ.get("OMNIROUTE_API_KEY", "")

    # 1. Install if missing
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

    # 2. Start if not running
    running = False
    try:
        req = urllib.request.Request(f"{or_url}/models",
                                     headers={"User-Agent": "HermesBridge/2.0"})
        with urllib.request.urlopen(req, timeout=3):
            running = True
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

    # 3. API key
    if not or_key:
        or_key = f"sk-{secrets.token_hex(16)}"
        os.environ["HERMES_CUSTOM_LOCALHOST_20128_API_KEY"] = or_key

    # 4. Set env vars for the bridge server
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
