#!/usr/bin/env python3
"""
Platform layer — auto-detects the host OS (Linux / Windows / macOS / Android)
and provides per-OS tool paths, download URLs, and install hints so the
Hermes Mobile Bridge runs everywhere with ZERO per-OS configuration.

Usage:
    from platform_utils import detect_os, pick_server_python, cloudflared_bin, tailscale_bin
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ─── OS detection ───────────────────────────────────────────────────────


def detect_os() -> str:
    """Return one of: 'android', 'linux', 'windows', 'macos'."""
    if os.environ.get("TERMUX_VERSION") or os.path.exists("/data/data/com.termux"):
        return "android"
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        return "macos"
    if sys_name == "windows" or os.name == "nt":
        return "windows"
    return "linux"


def is_windows() -> bool:
    return detect_os() == "windows"


def exe_name(name: str) -> str:
    """Append .exe on Windows (cloudflared, omniroute, tailscale)."""
    return f"{name}.exe" if is_windows() else name


# ─── Python interpreter selection ───────────────────────────────────────


def pick_server_python() -> str:
    """Return a python3 that can import the bridge server's dependencies.

    Preference: current interpreter → system python3 → 'python3' / 'py'.
    """
    candidates: list[str] = []
    if sys.executable and sys.executable not in candidates:
        candidates.append(sys.executable)
    # Android/Termux system python
    candidates.append("/data/data/com.termux/files/usr/bin/python3")
    # macOS Homebrew / common
    candidates.append("/opt/homebrew/bin/python3")
    candidates.append("/usr/local/bin/python3")
    # Windows launcher + fallbacks
    if is_windows():
        candidates.append("py -3")
        candidates.append("python")
    candidates.append("python3")

    for cand in candidates:
        try:
            r = subprocess.run(
                [cand, "-c", "import fastapi, uvicorn, bcrypt, qrcode"],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode == 0:
                return cand
        except Exception:
            continue
    return candidates[0]


# ─── Binary locations & install hints (per OS) ──────────────────────────


def _bin_dir() -> Path:
    """Per-OS binary dir: ~/.hermes-mobile-bridge/bin (Android/Termux-safe)."""
    d = Path.home() / ".hermes-mobile-bridge" / "bin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cloudflared_bin() -> str:
    """Full path to a cloudflared binary, downloading it if missing."""
    target = _bin_dir() / exe_name("cloudflared")
    if target.exists():
        return str(target)
    if shutil.which("cloudflared"):
        return "cloudflared"  # system install wins
    url = {
        "linux": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        "android": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
        "macos": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz",
        "windows": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
    }.get(detect_os())
    if not url:
        raise RuntimeError("cloudflared: unsupported OS")
    import urllib.request
    print(f"   ⬇ Downloading cloudflared for {detect_os()}…")
    tmp = target.with_suffix(".download")
    urllib.request.urlretrieve(url, tmp)
    if detect_os() == "macos":
        import tarfile
        with tarfile.open(tmp) as tf:
            member = next(m for m in tf.getmembers() if "cloudflared" in m.name and not m.isdir())
            tf.extract(member, _bin_dir())
        (Path(_bin_dir()) / member.name).rename(target)
        tmp.unlink(missing_ok=True)
    else:
        tmp.rename(target)
    target.chmod(0o755)
    return str(target)


def tailscale_bin() -> str | None:
    """Path to a tailscale CLI if available (returns None when missing —
    the bridge still works via the Cloudflare tunnel fallback)."""
    found = shutil.which(exe_name("tailscale"))
    if found:
        return found
    cand = _bin_dir() / exe_name("tailscale")
    return str(cand) if cand.exists() else None


def tailscale_ip(tailscale: str | None) -> str | None:
    """Best-effort Tailscale IPv4 detection, cross-platform:
    Android reads the stored tailscale-ip file (netlink needs root);
    everywhere else asks the CLI directly."""
    store = Path(os.getenv("STORE_PATH", Path.home() / ".hermes-mobile-server"))
    ip_file = store / "tailscale-ip"
    if ip_file.exists():
        ip = ip_file.read_text().strip()
        if ip:
            return ip
    if tailscale:
        try:
            r = subprocess.run([tailscale, "ip", "-4"], capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                ip = r.stdout.strip().split()[0]
                try:
                    ip_file.parent.mkdir(parents=True, exist_ok=True)
                    ip_file.write_text(ip)
                except Exception:
                    pass
                return ip
        except Exception:
            pass
    return None


# ─── OmniRoute ──────────────────────────────────────────────────────────


def omniroute_bin() -> str:
    """Path to the OmniRoute CLI, downloading it if missing (per-OS)."""
    target = _bin_dir() / exe_name("omniroute")
    if target.exists():
        return str(target)
    if shutil.which(exe_name("omniroute")):
        return exe_name("omniroute")
    os_name = detect_os()
    arch_map = {"amd64": "amd64", "arm64": "arm64"}
    arch = arch_map.get(platform.machine().lower() in ("arm64", "aarch64") and "arm64" or "amd64", "amd64")
    urls = {
        "linux": f"https://github.com/tawaresachin/omniroute/releases/latest/download/omniroute-linux-{arch}",
        "android": f"https://github.com/tawaresachin/omniroute/releases/latest/download/omniroute-linux-{arch}",
        "macos": f"https://github.com/tawaresachin/omniroute/releases/latest/download/omniroute-darwin-{arch}",
        "windows": f"https://github.com/tawaresachin/omniroute/releases/latest/download/omniroute-windows-{arch}.exe",
    }.get(os_name)
    if not urls:
        raise RuntimeError(f"omniroute: unsupported OS {os_name}")
    import urllib.request
    print(f"   ⬇ Downloading OmniRoute for {os_name}…")
    urllib.request.urlretrieve(urls, target)
    target.chmod(0o755)
    return str(target)
