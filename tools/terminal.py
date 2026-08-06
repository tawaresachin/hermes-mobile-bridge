#!/usr/bin/env python3
"""Terminal execution tool — runs shell commands on the host."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from tools import BaseTool

# Working directory — defaults to user's home, can be overridden via env
WORK_DIR = Path(os.getenv("AGENT_WORK_DIR", os.path.expanduser("~")))


class Terminal(BaseTool):
    name = "terminal"
    description = (
        "Execute a shell command on the host machine. "
        "The command runs in a persistent working directory. "
        "Use this to install packages, run scripts, build projects, and "
        "interact with the filesystem."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute"
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default 60, max 300)",
                "default": 60,
            },
            "workdir": {
                "type": "string",
                "description": "Working directory (default: current)",
                "default": "",
            },
        },
        "required": ["command"],
    }

    async def run(self, command: str, timeout: int = 60, workdir: str = "") -> str:
        cwd = Path(workdir) if workdir else WORK_DIR
        # Security: block dangerous commands that modify system
        blocked_prefixes = [
            "rm -rf /", "rm -rf /*", "mkfs", "dd if=", "> /dev/",
            ":(){ :|:& };:", "chmod -R 000 /",
        ]
        for prefix in blocked_prefixes:
            if command.strip().startswith(prefix):
                return f"⚠ Command blocked for safety: {prefix}"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env={**os.environ},
                shell=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=min(timeout, 300)
                )
            except asyncio.TimeoutError:
                proc.kill()
                return f"⚠ Command timed out after {timeout}s"

            output = ""
            if stdout:
                output += stdout.decode(errors="replace")
            if stderr:
                if output:
                    output += "\n--- stderr ---\n"
                output += stderr.decode(errors="replace")

            # Add exit code
            if proc.returncode != 0:
                output += f"\n[exit code: {proc.returncode}]"

            return output.strip()

        except Exception as e:
            return f"⚠ Failed to execute command: {e}"
