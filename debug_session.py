#!/usr/bin/env python3
"""Debug: show what _build_openai_messages would produce for a session."""
import json
from pathlib import Path

with open("/data/data/com.termux/files/home/.hermes-mobile-server/messages_e30e6178-e28b-4adb-afd8-193fcafe1e96.json") as f:
    msgs = json.load(f)

UPLOADS_DIR = Path("/data/data/com.termux/files/home/.hermes-mobile-server/uploads")

for m in msgs[-20:]:
    role = m["role"]
    attach_url = m.get("attachment_url", "")
    content = m["content"][:60]

    if attach_url:
        parts = attach_url.split("/")
        fpath = UPLOADS_DIR / parts[2] / "/".join(parts[3:])
        exists = fpath.exists()
        if exists:
            size = fpath.stat().st_size
            raw = fpath.read_bytes()[:200]
            is_text = False
            for enc in ("utf-8", "latin-1"):
                try:
                    decoded = raw.decode(enc)
                    printable = sum(1 for c in decoded if str(c).isprintable() or c in "\n\r\t")
                    ratio = printable / max(len(decoded), 1)
                    if ratio > 0.85:
                        is_text = True
                        break
                except:
                    continue
            print(f'{role}: "{content}"')
            print(f"  FILE: {attach_url} -> exists, {size} bytes, is_text={is_text}")
        else:
            print(f'{role}: "{content}"')
            print(f"  FILE: {attach_url} -> NOT FOUND")
    else:
        print(f'{role}: "{content}"')
    print()
