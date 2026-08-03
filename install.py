#!/usr/bin/env python3
"""
Hermes Mobile Bridge — single-command installer / updater / launcher.

ONE command installs, updates, and starts everything (plugin + server),
on Linux, macOS, Windows and Android/Termux:

    curl -fsSL https://raw.githubusercontent.com/tawaresachin/hermes-mobile-bridge/main/install.py | python3 -

Idempotent and safe to re-run at any time:

  Fresh machine  -> installs the Hermes plugin, bootstraps the server, starts it
  Existing setup -> updates the plugin (--force reinstall), git-pulls the server, restarts it

Options:
    --no-serve   Install/update only — do not start the server.
    --no-stt     Skip whisper STT (binary + ~141MB model). Default: ON —
                 one command installs a COMPLETE server with voice STT.
    --force      Force-reinstall the Hermes plugin (update path).
    --dir DIR    Server checkout directory (default ~/hermes-mobile-server).
    --help       Show this help.

Exit codes: 0 success, 1 failure (any step fails loudly).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

GIT_URL = "https://github.com/tawaresachin/hermes-mobile-bridge.git"
REPO_RAW = "https://raw.githubusercontent.com/tawaresachin/hermes-mobile-bridge/main"


def say(msg: str) -> None:
    print(f"  {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"\n✖ {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ─── Steps ───────────────────────────────────────────────────────────────


def step_detect() -> str:
    if os.environ.get("TERMUX_VERSION") or Path("/data/data/com.termux").exists():
        return "android"
    name = os.uname().sysname.lower() if hasattr(os, "uname") else os.name
    if name == "darwin":
        return "macos"
    if os.name == "nt":
        return "windows"
    return "linux"


def step_hermes() -> str | None:
    """Return the hermes CLI path, or None (standalone mode). Requires git."""
    hermes = shutil.which("hermes")
    git = shutil.which("git")
    if hermes and not git:
        fail("git is required (plugin install/update). Install git and re-run.")
    return hermes


def step_install_plugin(hermes: str, force: bool) -> None:
    """Install or update the Hermes plugin (idempotent, subdir-aware).

    Always force-reinstalls: Hermes subdir installs keep no .git, so
    reinstall IS the update path — and it makes the one-liner idempotent
    (fresh machine → install, existing machine → update)."""
    say("→ Hermes plugin: install/update mobile-bridge")
    cmd = [hermes, "plugins", "install", "tawaresachin/hermes-mobile-bridge/plugin", "--enable", "--force"]
    r = run(cmd, timeout=300)
    if r.returncode != 0:
        fail(f"Plugin install failed:\n{(r.stderr or r.stdout).strip()[:500]}")
    say("✓ plugin installed/enabled")


def step_server(bridge_dir: Path, force_plugin: bool) -> None:
    """Clone or git-pull the server checkout (install vs update)."""
    if (bridge_dir / "server.py").exists():
        say(f"→ Server: updating {bridge_dir}")
        r = run(["git", "-C", str(bridge_dir), "pull", "--ff-only", "-q"], timeout=120)
        if r.returncode != 0:
            fail(f"Server update failed:\n{(r.stderr or r.stdout).strip()[:300]}")
        say("✓ server up to date")
    else:
        say(f"→ Server: cloning into {bridge_dir}")
        bridge_dir.parent.mkdir(parents=True, exist_ok=True)
        r = run(["git", "clone", "--depth", "1", GIT_URL, str(bridge_dir)], timeout=300)
        if r.returncode != 0:
            fail(f"Server clone failed:\n{(r.stderr or r.stdout).strip()[:300]}")
        say("✓ server cloned")
        _install_deps(bridge_dir)


def _install_deps(bridge_dir: Path) -> None:
    req = bridge_dir / "requirements.txt"
    if not req.exists():
        return
    say("→ Installing Python dependencies (pip install -r requirements.txt)")
    r = run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], timeout=600)
    if r.returncode != 0:
        fail(f"Dependency install failed:\n{(r.stderr or r.stdout).strip()[:400]}")
    say("✓ dependencies installed")


def step_stt(bridge_dir: Path) -> None:
    """Ensure whisper-cli + multilingual model (STT for the voice screen).
    Uses the freshly-cloned repo's platform_utils for per-OS handling;
    degrades gracefully (voice falls back to the phone recognizer)."""
    sys.path.insert(0, str(bridge_dir))
    try:
        import platform_utils
    except Exception as e:
        say(f"⚠ STT skipped — platform_utils unavailable: {e}")
        return
    say("→ STT: whisper-cli + multilingual model (~147MB total)")
    cli = platform_utils.whisper_cli()
    if cli is None:
        say("  ⚠ whisper-cli not available on this OS — STT will fall back")
        say("    to the phone's recognizer (with its beep). Fix per README.")
    else:
        model = platform_utils.whisper_model()
        say(f"✓ STT ready: {cli}")
        say(f"  model: {model}")


def step_serve(hermes: str | None, bridge_dir: Path, port: str) -> None:
    """Start the server: via Hermes plugin when available, else standalone."""
    env = dict(os.environ)
    if port:
        env["PORT"] = port
    if hermes:
        say(f"→ Starting: hermes mobile-serve (port {port or '9119'})")
        try:
            subprocess.run([hermes, "mobile-serve"], env=env)
        except KeyboardInterrupt:
            say("stopped")
        except Exception as e:
            fail(f"Failed to start server: {e}")
    else:
        say(f"→ Starting: python3 server.py (standalone, port {port or '9119'})")
        try:
            subprocess.run([sys.executable, "-B", str(bridge_dir / "server.py")], cwd=str(bridge_dir), env=env)
        except KeyboardInterrupt:
            say("stopped")
        except Exception as e:
            fail(f"Failed to start server: {e}")


# ─── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Hermes Mobile Bridge installer/updater")
    ap.add_argument("--no-serve", action="store_true", help="install/update only")
    ap.add_argument("--no-stt", action="store_true", help="skip whisper STT (binary + ~141MB model)")
    ap.add_argument("--force", action="store_true", help="force-reinstall the plugin (update)")
    ap.add_argument("--dir", default=str(Path.home() / "hermes-mobile-server"), help="server checkout dir")
    ap.add_argument("--port", default="", help="server port override")
    args = ap.parse_args()

    bridge_dir = Path(args.dir).expanduser().resolve()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Hermes Mobile Bridge — install / update / serve        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    os_name = step_detect()
    say(f"OS detected: {os_name}")

    hermes = step_hermes()
    if hermes:
        step_install_plugin(hermes, args.force)
    else:
        say("→ Hermes CLI not found — standalone mode (server only)")
        say("  Tip: install Hermes to get `hermes mobile-serve` + auto-updates.")

    step_server(bridge_dir, args.force)

    if not args.no_stt:
        step_stt(bridge_dir)

    if args.no_serve:
        say("\n✓ Install/update complete (--no-serve). Start later with:\n")
        if hermes:
            say("    hermes mobile-serve")
        else:
            say(f"    python3 {bridge_dir / 'server.py'}")
        return

    print()
    step_serve(hermes, bridge_dir, args.port)


if __name__ == "__main__":
    main()
