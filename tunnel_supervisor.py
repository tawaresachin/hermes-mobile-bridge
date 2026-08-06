#!/usr/bin/env python3
"""
Tunnel Supervisor — keeps the Cloudflare quick tunnel alive.

Responsibilities:
  1. Kill any stale cloudflared processes (duplicate tunnels split the URL).
  2. Start exactly ONE cloudflared -> http://localhost:PORT.
  3. Tail its log output; the moment a fresh trycloudflare.com URL appears,
     write it to STORE_PATH/.current_tunnel_url (the bridge reads this file
     on every /setup/connect call, so the app's refresh always gets the
     CURRENT live URL).
  4. Watch the process; if it dies (crash / expiry), restart it and repeat.

Usage:
    python3 tunnel_supervisor.py [--port 9119] [--cloudflared /path/to/binary]
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

STORE_PATH = Path.home() / ".hermes-mobile-server"
URL_FILE = STORE_PATH / ".current_tunnel_url"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

logger = None  # replaced in main()


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[tunnel-supervisor {ts}] {msg}"
    print(line, flush=True)
    if logger:
        logger.write(line + "\n")
        logger.flush()


def find_cloudflared(candidate: str) -> str:
    """Locate the cloudflared binary."""
    if candidate and os.path.isfile(candidate):
        return candidate
    for p in ("cloudflared", os.path.expanduser("~/../usr/bin/cloudflared")):
        try:
            r = subprocess.run(["which", p], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            pass
    return "cloudflared"  # let PATH resolve it


def kill_stale(port: int) -> None:
    """Kill every cloudflared that tunnels to our port — we want exactly one."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", f"cloudflared.*--url http://localhost:{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for pid in r.stdout.split():
            pid = pid.strip()
            if not pid.isdigit():
                continue
            try:
                os.kill(int(pid), signal.SIGKILL)
                log(f"killed stale cloudflared PID {pid}")
            except ProcessLookupError:
                pass
    except Exception as e:
        log(f"kill_stale: {e}")


def start_cloudflared(binary: str, port: int) -> subprocess.Popen:
    """Start cloudflared, returning the Popen handle (stdout piped)."""
    cmd = [binary, "tunnel", "--url", f"http://localhost:{port}"]
    log("starting: " + " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=os.environ.copy(),
    )


def watch_for_url(proc: subprocess.Popen, known_url: str) -> str:
    """
    Read cloudflared's stdout line by line. The first trycloudflare.com URL
    found becomes the current URL and is persisted to .current_tunnel_url.
    Returns the URL (known_url if nothing new yet).
    """
    global current_url
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            break  # process exited / EOF
        line = line.strip()
        if not line:
            continue
        m = URL_RE.search(line)
        if m:
            url = m.group(0)
            if url != known_url:
                current_url = url
                STORE_PATH.mkdir(parents=True, exist_ok=True)
                URL_FILE.write_text(url + "\n")
                log(f"🆕 tunnel URL: {url}")
            return url
    return known_url


def health_check(url: str) -> bool:
    """Quick reachability probe of the current tunnel URL."""
    import urllib.request
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=6) as resp:
            return 200 <= resp.status < 500  # any HTTP answer means tunnel alive
    except Exception:
        return False


def _local_health(port: int) -> bool:
    """Probe the local bridge server (bypasses the tunnel entirely)."""
    import urllib.request
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health", method="GET")
        with urllib.request.urlopen(req, timeout=4) as resp:
            return resp.status == 200
    except Exception:
        return False


def main() -> None:
    global logger, current_url
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9119)
    parser.add_argument("--cloudflared", default="")
    args = parser.parse_args()

    STORE_PATH.mkdir(parents=True, exist_ok=True)
    logger = open(STORE_PATH / "tunnel_supervisor.log", "a")
    binary = find_cloudflared(args.cloudflared)
    log(f"cloudflared binary: {binary}")
    log(f"port: {args.port}")

    # Seed current_url from the existing file so a fresh tunnel re-uses it
    current_url = ""
    if URL_FILE.exists():
        content = URL_FILE.read_text()
        m = URL_RE.search(content)
        if m:
            current_url = m.group(0)
            log(f"seeded from file: {current_url}")

    restarts = 0
    while True:
        kill_stale(args.port)
        proc = start_cloudflared(binary, args.port)
        if not proc:
            log("failed to start cloudflared — retrying in 10s")
            time.sleep(10)
            continue

        # Phase 1: wait for the fresh URL (usually < 10s)
        deadline = time.time() + 45
        got_url = False
        while time.time() < deadline:
            if proc.poll() is not None:
                log(f"cloudflared exited early (rc={proc.returncode}) — restarting")
                break
            url = watch_for_url(proc, current_url)
            if url != current_url:
                current_url = url
                got_url = True
                restarts += 1
                log(f"restart #{restarts}: live at {current_url}")
                break
            time.sleep(0.3)
        else:
            log("timed out waiting for URL — will restart")
            proc.kill()
            continue

        # Phase 2: supervise — restart on crash, log URL changes
        if got_url:
            while True:
                rc = proc.poll()
                if rc is not None:
                    log(f"cloudflared DIED (rc={rc}) — restarting in 3s")
                    time.sleep(3)
                    break  # outer loop restarts
                url = watch_for_url(proc, current_url)
                if url != current_url:
                    current_url = url
                    restarts += 1
                    log(f"URL changed → {current_url} (restart #{restarts})")
                # Periodic liveness probe: if the bridge is up locally but the
                # tunnel stops answering, restart cloudflared. (If the local
                # server is down, don't churn the tunnel — that's not its fault.)
                if _local_health(port) and not health_check(current_url):
                    log(f"tunnel {current_url} not reachable — forcing restart")
                    proc.kill()
                    time.sleep(1)
                    break
                time.sleep(30)

        # If we fell through (no URL / early exit), small backoff then retry
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[tunnel-supervisor] stopped", flush=True)
        sys.exit(0)
