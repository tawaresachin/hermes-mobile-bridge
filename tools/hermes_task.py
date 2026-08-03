#!/usr/bin/env python3
"""Hermes delegation tool — hands a task to the FULL Hermes Agent.

Runs `hermes chat -q "<task>"` as a subprocess and returns its complete
reply. This gives the bridge the SAME capabilities as the real Hermes
(memory, skills, vision, subagents, plugins, all tools) WITHOUT the bridge
re-implementing any of them — the bridge stays a thin, lightweight shell.
"""
from __future__ import annotations

import asyncio
import os

from tools import BaseTool

# Path to the Hermes CLI. Override via HERMES_BIN if not on PATH.
HERMES_BIN = os.getenv("HERMES_BIN", "hermes")


class HermesTask(BaseTool):
    name = "hermes_task"
    description = (
        "Delegate a task to the FULL Hermes Agent on this machine. This has "
        "far more capability than this bridge's own tools: persistent memory, "
        "learned skills, vision, subagent delegation, plugins, and every "
        "Hermes tool. Use it for heavy or complex multi-step work, coding "
        "projects, research, or any task that needs the complete agent. It "
        "takes a while (tens of seconds to minutes) but returns the agent's "
        "full, best answer. Pass the whole task as one clear string."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The full task/instruction to give the Hermes agent."
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default 600, max 1800)",
                "default": 600,
            },
        },
        "required": ["task"],
    }

    async def run(self, task: str, timeout: int = 600) -> str:
        if not task or not task.strip():
            return "⚠ No task provided."
        timeout = max(10, min(timeout, 1800))

        # Spawn `hermes chat -q "<task>"` — argv list avoids shell escaping bugs.
        cmd = [HERMES_BIN, "chat", "-q", task.strip()]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ},
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return f"⚠️ Hermes task timed out after {timeout}s. Try a smaller task."

            out = stdout.decode(errors="replace")
            err_out = stderr.decode(errors="replace")

            # Drop the trailing "Session: ...  Duration: ... " summary block that
            # `hermes chat` appends — it's not part of the answer.
            lines = out.splitlines()
            trimmed = []
            skip = False
            for ln in lines:
                if ln.strip().startswith("Session:"):
                    skip = True
                    continue
                if skip and not ln.strip():
                    continue
                if ln.strip():
                    skip = False
                trimmed.append(ln)
            out = "\n".join(trimmed).strip()

            if not out and err_out:
                return f"⚠️ Hermes task produced no stdout.\nstderr:\n{err_out.strip()}"
            return out

        except FileNotFoundError:
            return (
                "⚠️ Hermes CLI not found. Set HERMES_BIN to the hermes path, "
                "or install Hermes Agent on this machine."
            )
        except Exception as e:
            return f"⚠️ Hermes task failed: {e}"