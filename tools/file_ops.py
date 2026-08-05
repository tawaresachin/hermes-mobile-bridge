#!/usr/bin/env python3
"""File system tools — read, write, search, and patch files."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from tools import BaseTool, MAX_TOOL_OUTPUT_CHARS

WORK_DIR = Path(os.getenv("AGENT_WORK_DIR", os.path.expanduser("~")))


def _safe_path(path: str) -> Path:
    """Resolve a path relative to the working directory, preventing escapes."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (WORK_DIR / p).resolve()


class ReadFile(BaseTool):
    name = "read_file"
    description = "Read the contents of a text file with line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (absolute or relative to work dir)"
            },
            "offset": {
                "type": "integer",
                "description": "Starting line number (1-indexed, default 1)",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read (default 200, max 2000)",
                "default": 200,
            },
        },
        "required": ["path"],
    }

    async def run(self, path: str, offset: int = 1, limit: int = 200) -> str:
        fpath = _safe_path(path)
        if not fpath.exists():
            return f"⚠ File not found: {fpath}"
        if not fpath.is_file():
            return f"⚠ Not a file: {fpath}"

        limit = min(limit, 2000)
        try:
            lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            start = max(0, offset - 1)
            end = min(total, start + limit)
            selected = lines[start:end]

            output = "\n".join(
                f"{i + start + 1}|{line}" for i, line in enumerate(selected)
            )
            if start > 0 or end < total:
                output = f"{fpath} (showing lines {start + 1}-{end} of {total})\n{output}"
                if end < total:
                    output += f"\n... (use offset={end + 1} to continue)"
            return output
        except Exception as e:
            return f"⚠ Error reading file: {e}"


class WriteFile(BaseTool):
    name = "write_file"
    description = "Write content to a file, completely replacing existing content. Creates parent directories automatically."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (absolute or relative to work dir)"
            },
            "content": {
                "type": "string",
                "description": "Complete content to write to the file"
            },
        },
        "required": ["path", "content"],
    }

    async def run(self, path: str, content: str) -> str:
        fpath = _safe_path(path)
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content, encoding="utf-8")
            size = len(content)
            return f"✓ Wrote {size} bytes to {fpath}"
        except Exception as e:
            return f"⚠ Error writing file: {e}"


class SearchFiles(BaseTool):
    name = "search_files"
    description = "Search file contents or find files by name using ripgrep."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern for content search, or glob pattern for file search"
            },
            "target": {
                "type": "string",
                "enum": ["content", "files"],
                "description": "'content' searches inside files, 'files' finds files by name",
                "default": "content",
            },
            "path": {
                "type": "string",
                "description": "Directory to search (default: work directory)",
                "default": ".",
            },
            "file_glob": {
                "type": "string",
                "description": "Filter by file pattern in content search (e.g. '*.py')",
                "default": "",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 20)",
                "default": 20,
            },
        },
        "required": ["pattern"],
    }

    async def run(self, pattern: str, target: str = "content", path: str = ".",
                  file_glob: str = "", limit: int = 20) -> str:
        search_path = _safe_path(path)
        limit = min(limit, 50)

        try:
            if target == "files":
                # Use glob via find
                result = subprocess.run(
                    ["find", str(search_path), "-name", pattern, "-type", "f"],
                    capture_output=True, text=True, timeout=30,
                )
                lines = [l for l in result.stdout.splitlines() if l.strip()][:limit]
                if not lines:
                    return f"No files matching '{pattern}' found in {search_path}"
                return "\n".join(lines)
            else:
                # Use ripgrep if available, else grep -r
                cmd = []
                if subprocess.run(["which", "rg"], capture_output=True).returncode == 0:
                    cmd = ["rg", "-n", "--no-heading"]
                    if file_glob:
                        cmd.extend(["-g", file_glob])
                    cmd.extend([pattern, str(search_path)])
                else:
                    # Note: `"--include=" + glob if glob else ""` would
                    # inject an EMPTY arg when glob is unset, shifting
                    # `pattern` into the path position and breaking the
                    # search — build the arg list conditionally instead.
                    cmd = ["grep", "-rn"]
                    if file_glob:
                        cmd.append("--include=" + file_glob)
                    cmd.extend([pattern, str(search_path)])
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30,
                )
                lines = [l for l in result.stdout.splitlines() if l.strip()][:limit]
                if not lines:
                    return f"No matches for '{pattern}' in {search_path}"
                return "\n".join(lines)
        except subprocess.TimeoutExpired:
            return "⚠ Search timed out"
        except Exception as e:
            return f"⚠ Search failed: {e}"


class Patch(BaseTool):
    name = "patch"
    description = "Apply a targeted find-and-replace edit to a file. Uses fuzzy matching so minor whitespace differences won't break it."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit"
            },
            "old_string": {
                "type": "string",
                "description": "Text to find and replace (include surrounding context for uniqueness)"
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text"
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def run(self, path: str, old_string: str, new_string: str) -> str:
        fpath = _safe_path(path)
        if not fpath.exists():
            return f"⚠ File not found: {fpath}"

        try:
            content = fpath.read_text(encoding="utf-8")
            if old_string not in content:
                return f"⚠ Could not find the specified text in {fpath}. Make sure the text is exactly as it appears."
            new_content = content.replace(old_string, new_string, 1)
            if new_content == content:
                return "⚠ No changes made (text unchanged after replacement)"
            fpath.write_text(new_content, encoding="utf-8")
            return f"✓ Patched {fpath}"
        except Exception as e:
            return f"⚠ Error patching file: {e}"
