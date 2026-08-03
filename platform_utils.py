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


# ─── Whisper STT (whisper.cpp) ─────────────────────────────────────────

WHISPER_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
_WHISPER_RELEASE = "https://github.com/ggml-org/whisper.cpp/releases/latest/download"


def _flatten_tree(src: Path, dst: Path) -> None:
    """Move the CONTENTS of src's single top-level dir up into dst, so the
    whisper-cli binary and its shared libs ($ORIGIN resolution) end up side
    by side in dst. Handles both tar/zip layouts (one root dir each)."""
    if not src.exists():
        return
    roots = [p for p in src.iterdir() if p.is_dir()]
    root = roots[0] if len(roots) == 1 else src
    for item in root.iterdir():
        shutil.move(str(item), str(dst / item.name))


def whisper_cli() -> str | None:
    """Path to the whisper.cpp CLI, installing the official prebuilt if
    missing (Linux x64/arm64 + Windows x64 have releases; macOS → brew
    whisper-cpp; Android/Termux → compile instructions). Returns None when
    unavailable so callers can degrade (STT falls back to the phone's
    recognizer)."""
    target = _bin_dir() / exe_name("whisper-cli")
    if target.exists():
        return str(target)
    if shutil.which("whisper-cli"):
        return "whisper-cli"
    os_name = detect_os()
    if os_name == "windows":
        url = f"{_WHISPER_RELEASE}/whisper-bin-x64.zip"
        archive = _bin_dir() / "whisper-bin-x64.zip"
        import urllib.request, zipfile
        print("   ⬇ Downloading whisper-cli (Windows x64)…")
        urllib.request.urlretrieve(url, archive)
        tmp = _bin_dir() / "_whisper_tmp"
        tmp.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        # Flatten the single top-level dir so libs + exe sit beside each
        # other in _bin_dir() (whisper-cli resolves $ORIGIN libs itself).
        _flatten_tree(tmp, _bin_dir())
        shutil.rmtree(tmp, ignore_errors=True)
        archive.unlink(missing_ok=True)
        return str(target)
    if os_name == "linux":
        arch = "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "x64"
        url = f"{_WHISPER_RELEASE}/whisper-bin-ubuntu-{arch}.tar.gz"
        archive = _bin_dir() / f"whisper-bin-ubuntu-{arch}.tar.gz"
        import tarfile, urllib.request
        print(f"   ⬇ Downloading whisper-cli (Linux {arch})…")
        urllib.request.urlretrieve(url, archive)
        tmp = _bin_dir() / "_whisper_tmp"
        tmp.mkdir(exist_ok=True)
        # filter="data" avoids the Python 3.14 tar sanitization warning.
        with tarfile.open(archive) as tf:
            tf.extractall(tmp, filter="data")
        _flatten_tree(tmp, _bin_dir())
        shutil.rmtree(tmp, ignore_errors=True)
        archive.unlink(missing_ok=True)
        target.chmod(0o755)
        return str(target)
    if os_name == "macos":
        # No prebuilt macOS CLI in the release — brew whisper-cpp ships one.
        if shutil.which("brew"):
            print("   ⬇ Installing whisper-cpp via Homebrew…")
            subprocess.run(["brew", "install", "whisper-cpp"], capture_output=True, timeout=600)
            if shutil.which("whisper-cli"):
                return "whisper-cli"
        print("   ⚠ whisper-cli unavailable — install it:")
        print("     brew install whisper-cpp")
        return None
    # Android/Termux: no official prebuilt (bionic libc) — compile once:
    print("   ⚠ whisper-cli not found on Android — compile it (one time):")
    print("     git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp")
    print("     cmake -B build -DWHISPER_BUILD_TESTS=OFF && cmake --build build -j4")
    print(f"     cp build/bin/whisper-cli {target}")
    return None


def whisper_model() -> str:
    """Path to the multilingual whisper model (ggml-base, ~141MB, covers
    Marathi/Hindi + all major world languages), downloading if missing."""
    store = Path(os.getenv("STORE_PATH", Path.home() / ".hermes-mobile-server"))
    model = store / "models" / "ggml-base.bin"
    if model.exists() and model.stat().st_size > 100_000_000:
        return str(model)
    model.parent.mkdir(parents=True, exist_ok=True)
    print("   ⬇ Downloading whisper model ggml-base (~141MB)…")
    tmp = model.with_suffix(".download")
    import urllib.request
    urllib.request.urlretrieve(WHISPER_MODEL_URL, tmp)
    tmp.replace(model)
    return str(model)
