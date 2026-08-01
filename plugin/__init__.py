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
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ─── Bootstrap: make platform_utils importable ──────────────────────────
# The plugin ships INSIDE the bridge repo (plugin/__init__.py), so the repo
# root is our parent — whether installed by symlink, by copy inside the repo,
# or as the legacy standalone ~/hermes-mobile-server layout.
def _ensure_platform_utils() -> None:
    for cand in (Path(__file__).resolve().parent.parent, Path.home() / "hermes-mobile-server"):
        if (cand / "platform_utils.py").exists():
            sp = str(cand)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            return

_ensure_platform_utils()
import platform_utils  # noqa: E402

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


def _ensure_forced_defaults(bridge_dir: Path) -> None:
    """Force token-saving defaults at setup: caveman style + context headroom.

    Written into the bridge's .env so every conversation, on every restart,
    keeps replies short and context compact (rolling summary) — the user
    should never have to opt in per-session."""
    env_path = bridge_dir / ".env"
    defaults = {
        "CAVEMAN_STYLE": "1",          # short caveman replies — fewer tokens
        "CONTEXT_RECENT_K": "24",      # verbatim recent-message window
        "CONTEXT_SUMMARY_BATCH": "12", # older messages roll into the summary
    }
    try:
        existing: dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip()
        changed = False
        for k, v in defaults.items():
            if existing.get(k) != v:
                existing[k] = v
                changed = True
        if changed:
            lines = [f"{k}={v}" for k, v in existing.items()]
            env_path.write_text("\n".join(lines) + "\n")
            logger.info("Forced defaults written to %s: %s", env_path.name, ", ".join(defaults))
    except Exception as e:
        logger.warning("Could not write forced defaults to %s: %s", env_path, e)


def _bootstrap_server(bridge_dir: Path) -> None:
    """First-run bootstrap + auto-update:
    - server code missing → clone the repo + install deps (fresh install)
    - server code present AND it's a git checkout we manage → `git pull` so
      every `hermes mobile-serve` start picks up the latest code (updates).
    Set BRIDGE_AUTO_UPDATE=0 to disable the pull."""
    if not (bridge_dir / "server.py").exists():
        print("   ⬇ Bridge server not found — bootstrapping…")
        try:
            import urllib.request
            bridge_dir.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(
                ["git", "clone", "--depth", "1", GIT_URL, str(bridge_dir)],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0 or not (bridge_dir / "server.py").exists():
                raise RuntimeError((r.stderr or r.stdout or "clone failed").strip()[:300])
            print("   ✅ Server cloned to", bridge_dir)
        except Exception as e:
            print(f"   ⚠ Auto-bootstrap failed: {e}")
            print("     Clone manually: git clone https://github.com/tawaresachin/hermes-mobile-bridge")
            raise

        # Install Python deps (idempotent — pip skips already-satisfied).
        try:
            req = bridge_dir / "requirements.txt"
            if req.exists():
                py = sys.executable
                print("   📦 Installing dependencies (pip install -r requirements.txt)…")
                r = subprocess.run(
                    [py, "-m", "pip", "install", "-q", "-r", str(req)],
                    capture_output=True, text=True, timeout=300,
                )
                if r.returncode != 0:
                    logger.warning("pip install had warnings: %s", (r.stderr or "")[:200])
        except Exception as e:
            logger.warning("Dependency install failed: %s", e)
        return

    # Server exists — auto-update if it's a git checkout we manage.
    auto_update = os.getenv("BRIDGE_AUTO_UPDATE", "1").lower() not in ("0", "false", "no", "off")
    if auto_update and (bridge_dir / ".git").exists():
        try:
            r = subprocess.run(
                ["git", "-C", str(bridge_dir), "pull", "--ff-only", "-q"],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0 and r.stdout.strip() and "Already up to date" not in r.stdout:
                print("   🔄 Server updated:", r.stdout.strip().splitlines()[-1][:80])
        except Exception as e:
            logger.warning("Server auto-update skipped: %s", e)


GIT_URL = "https://github.com/tawaresachin/hermes-mobile-bridge.git"


def _self_update_plugin() -> None:
    """Keep the INSTALLED PLUGIN in sync with the repo (copy-install safe).

    Hermes subdir installs keep no `.git` in the installed plugin dir, so
    `hermes plugins update` can't pull. Instead we shallow-clone the repo and
    swap __init__.py + plugin.yaml. No-op when running from inside the repo
    (symlink/dev layout). Changes apply on the NEXT start. Gate:
    BRIDGE_AUTO_UPDATE=0 disables."""
    auto_update = os.getenv("BRIDGE_AUTO_UPDATE", "1").lower() not in ("0", "false", "no", "off")
    if not auto_update:
        return
    plugin_dir = Path(__file__).resolve().parent
    # Inside the repo (symlink or dev checkout) — the repo IS the source.
    if (plugin_dir.parent / "server.py").exists():
        return
    try:
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "repo"
            r = subprocess.run(
                ["git", "clone", "--depth", "1", GIT_URL, str(clone)],
                capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0:
                raise RuntimeError((r.stderr or "clone failed").strip()[:200])
            src = clone / "plugin"
            changed = False
            for f in ("__init__.py", "plugin.yaml"):
                if (src / f).exists():
                    dst = plugin_dir / f
                    if not dst.exists() or dst.read_bytes() != (src / f).read_bytes():
                        shutil.copy2(src / f, dst)
                        changed = True
            if changed:
                print("   🔄 Plugin updated (applies on next start)")
    except Exception as e:
        logger.warning("Plugin self-update skipped: %s", e)


def _start_server(port: int, host: str, omniroute: bool) -> None:
    """Start the Hermes Mobile Bridge server (cross-platform)."""
    # Keep the plugin itself current (copy-installs have no .git to pull).
    _self_update_plugin()

    # Smart bridge dir resolution: the plugin ships INSIDE the bridge repo
    # (plugin/__init__.py), so prefer the repo next to us; fall back to the
    # legacy ~/hermes-mobile-server location for copy-installs.
    repo_root = Path(__file__).resolve().parent.parent
    bridge_dir = repo_root if (repo_root / "server.py").exists() else Path.home() / "hermes-mobile-server"
    _bootstrap_server(bridge_dir)  # fresh installs auto-clone + install deps
    if not bridge_dir.exists():
        print("⚠ Bridge server not found. Clone: git clone https://github.com/tawaresachin/hermes-mobile-bridge")
        return

    os_name = platform_utils.detect_os()
    print(f"🤖 Hermes Mobile Bridge ({os_name}) starting on http://{host}:{port}")

    if omniroute:
        _ensure_omnirouter()
    ts_ip = _ensure_tailscale()

    # Forced defaults (caveman + context headroom) are persisted at setup so
    # they survive restarts and apply to every conversation automatically.
    _ensure_forced_defaults(bridge_dir)

    # Build env AFTER lifecycle setup so generated keys propagate to the subprocess
    env = os.environ.copy()
    env["AI_BASE_URL"] = env.get("AI_BASE_URL", "http://localhost:20128/v1")
    env["AI_MODEL"] = env.get("AI_MODEL", "auto/best-coding")
    env["HOST"] = host
    env["PORT"] = str(port)
    env["CAVEMAN_STYLE"] = env.get("CAVEMAN_STYLE", "1")
    env["CONTEXT_RECENT_K"] = env.get("CONTEXT_RECENT_K", "24")
    env["CONTEXT_SUMMARY_BATCH"] = env.get("CONTEXT_SUMMARY_BATCH", "12")

    if ts_ip:
        print(f"   🖧 Tailscale: {ts_ip}")
        env["HERMES_TAILSCALE_IP"] = ts_ip

    print(f"   AI: {env['AI_MODEL']} @ {env['AI_BASE_URL']}")
    print(f"   Plugin: ~/.hermes/plugins/mobile-bridge/")

    # Start the Cloudflare tunnel supervisor (auto-restart + URL tracking)
    _ensure_tunnel_supervisor(port, bridge_dir)

    # Pick a python that has the server's deps. The hermes venv may lack
    # bcrypt/qrcode even though the system python3 has everything.
    server_python = _pick_server_python()
    print(f"   Server interpreter: {server_python}")

    subprocess.run(
        [server_python, "-B", "server.py"],
        cwd=str(bridge_dir),
        env=env,
    )


def _pick_server_python() -> str:
    """Return a python3 that can import the bridge server's dependencies
    (OS-aware: Windows py launcher / macOS Homebrew / Termux)."""
    return platform_utils.pick_server_python()


def _ensure_tunnel_supervisor(port: int, bridge_dir: Path) -> None:
    """Auto-lifecycle for the Cloudflare quick tunnel (cross-platform):
    start the supervisor in the background if it isn't already running.
    It kills stale cloudflared instances, restarts on crash, and keeps
    .current_tunnel_url fresh so the app's refresh always gets a live URL."""
    # Already running? Portable check via the supervisor's pidfile + os.kill
    # (pgrep doesn't exist on Windows).
    pidfile = bridge_dir / ".tunnel_supervisor.pid"
    try:
        if pidfile.exists():
            pid = int(pidfile.read_text().strip())
            os.kill(pid, 0)  # raises if dead
            logger.info("Tunnel supervisor already running (pid %s)", pid)
            return
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        pass  # stale pidfile → respawn below

    sup_path = bridge_dir / "tunnel_supervisor.py"
    if not sup_path.exists():
        logger.warning("tunnel_supervisor.py not found in %s", bridge_dir)
        return

    try:
        popen_kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if platform_utils.detect_os() == "windows":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            [sys.executable, "-B", str(sup_path), "--port", str(port)],
            cwd=str(bridge_dir),
            **popen_kwargs,
        )
        # Write a pidfile so the "already running" check works everywhere.
        try:
            pidfile.write_text(str(proc.pid))
        except Exception:
            pass
        logger.info("Tunnel supervisor started (pid %s)", proc.pid)
    except Exception as e:
        logger.warning("Failed to start tunnel supervisor: %s", e)


def _ensure_omnirouter() -> None:
    """Auto-lifecycle for OmniRoute: install -> start -> configure
    (OS-aware binary download; falls back to npm when present)."""
    or_url = "http://localhost:20128/v1"
    or_key = os.environ.get("HERMES_CUSTOM_LOCALHOST_20128_API_KEY", "") or \
             os.environ.get("OMNIROUTE_API_KEY", "")

    or_bin = shutil.which("omniroute")
    if not or_bin:
        try:
            or_bin = platform_utils.omniroute_bin()
        except Exception as e:
            logger.warning("OmniRoute binary unavailable: %s", e)
    if not or_bin:
        logger.info("OmniRoute not found — trying npm install...")
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
    """Ensure Tailscale on the bridge device (OS-aware):
    Android → app-mode detection + auto-install APK;
    Linux/macOS/Windows → CLI detection (`tailscale ip -4`), else install hint.
    Returns the Tailscale IP or empty string (Cloudflare tunnel fallback)."""
    os_name = platform_utils.detect_os()
    ts_bin = platform_utils.tailscale_bin()
    ts_ip = ""

    # 1. Android app mode — no CLI needed (netlink needs root on Android 14)
    if os_name == "android":
        ts_ip = platform_utils.tailscale_ip(None)
        if ts_ip:
            return ts_ip
        if not _is_tailscale_app_installed():
            _auto_install_tailscale_app()
        ts_ip = platform_utils.tailscale_ip(None)
        if ts_ip:
            logger.info("Tailscale active after setup: %s", ts_ip)
            return ts_ip
    else:
        # 2. Desktop CLI: `tailscale ip -4` (Linux/macOS/Windows)
        ts_ip = platform_utils.tailscale_ip(ts_bin)
        if ts_ip:
            logger.info("Tailscale: %s", ts_ip)
            return ts_ip
        # 3. Not installed — give the per-OS install hint, keep tunnel fallback
        _print_tailscale_hint(os_name)

    return ""


def _print_tailscale_hint(os_name: str) -> None:
    hints = {
        "linux": "sudo apt install tailscale   # or: curl -fsSL https://tailscale.com/install.sh | sh",
        "macos": "brew install tailscale       # then: open -a Tailscale and sign in",
        "windows": "winget install Tailscale.Tailscale   # or Microsoft Store",
        "android": "Install the Tailscale app from Play Store / F-Droid and sign in",
    }
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  Tailscale not active                                     ║")
    print(f"  ║  {hints.get(os_name, 'Install Tailscale')!s:<45}║")
    print("  ║  Sign in with the SAME account as the phone.             ║")
    print("  ║  Then restart 'hermes mobile-serve'.                     ║")
    print("  ║  Until then: Cloudflare tunnel is used as fallback.      ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    logger.info("Tailscale not active — using Cloudflare tunnel fallback")


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
    )
