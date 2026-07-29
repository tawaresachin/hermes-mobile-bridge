#!/usr/bin/env python3
"""
Hermes feature integration: skills loading, command parsing, rules,
and system prompt assembly — matching the Telegram gateway behaviour.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Runtime config (set by server.py on startup)
AI_MODEL = "unknown"
AI_BASE_URL = "unknown"


def set_config(model: str, base_url: str) -> None:
    """Update config from server.py at startup."""
    global AI_MODEL, AI_BASE_URL
    AI_MODEL = model
    AI_BASE_URL = base_url


logger = logging.getLogger(__name__)

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
HERMES_AGENT_DIR = HERMES_HOME / "hermes-agent"

# ─── Skills ──────────────────────────────────────────────────────────────


def _hermes_venv_python() -> Optional[str]:
    """Locate the Hermes venv python for importing Hermes modules."""
    candidates = [
        HERMES_AGENT_DIR / "venv" / "bin" / "python3",
        HERMES_AGENT_DIR / "venv" / "bin" / "python",
        HERMES_HOME / "venv" / "bin" / "python3",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


_skills_cache: str | None = None


def load_skills_system_prompt() -> str:
    """Load Hermes skills as a system-prompt section using Hermes's own builder."""
    global _skills_cache
    if _skills_cache is not None:
        return _skills_cache

    venv_python = _hermes_venv_python()
    if not venv_python:
        logger.warning("Hermes venv not found — skills disabled")
        _skills_cache = ""
        return ""

    import subprocess
    script = """import sys
sys.path.insert(0, {agent_dir!r})
try:
    from agent.prompt_builder import build_skills_system_prompt
    result = build_skills_system_prompt()
    print(result)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(0)
"""
    try:
        proc = subprocess.run(
            [venv_python, "-c", script.format(agent_dir=repr(str(HERMES_AGENT_DIR)))],
            capture_output=True, text=True, timeout=30,
        )
        result = proc.stdout.strip()
        if result:
            _skills_cache = result
            logger.info(f"Loaded skills system prompt ({len(result)} chars)")
            return result
        else:
            # Check stderr for errors
            stderr_text = proc.stderr.strip()
            if stderr_text:
                logger.warning(f"Skills loader stderr: {stderr_text[:200]}")
            else:
                logger.warning("Skills loader returned empty result")
            _skills_cache = ""
            return ""
    except Exception as e:
        logger.warning(f"Skills loader failed: {e}")
        _skills_cache = ""
        return ""


# ─── Commands ────────────────────────────────────────────────────────────


COMMANDS = {
    "/help": "Show available commands and tips.",
    "/reset": "Start a fresh conversation (clears history).",
    "/new": "Same as /reset.",
    "/retry": "Regenerate the last response.",
    "/model": "Show the current AI model.",
    "/clear": "Clear the current session.",
    "/skills": "List available Hermes skills.",
    "/version": "Show version info.",
    "/info": "Show session info.",
}


def handle_command(query: str, session_id: str) -> tuple[bool, str | None]:
    """Check if query is a slash command. Returns (handled, response_text).
    If handled=True, the caller should send response_text as the assistant reply.
    """
    cmd = query.strip().split()[0].lower()

    if cmd == "/help":
        lines = ["**Available Commands**", ""]
        for c, desc in COMMANDS.items():
            lines.append(f"`{c}` — {desc}")
        return True, "\n".join(lines)

    if cmd in ("/reset", "/new"):
        return True, "✅ Session reset. Starting fresh."

    if cmd == "/retry":
        return True, (
            "⚠️ Retry is handled by the mobile app — tap the last message "
            "and select 'Retry' from the menu. The server will regenerate."
        )

    if cmd == "/model":
        return True, f"**Model:** {AI_MODEL}\n**Provider:** {AI_BASE_URL}"

    if cmd == "/clear":
        return True, "✅ Session cleared."

    if cmd == "/skills":
        skill_list = []
        skills_dir = HERMES_HOME / "skills"
        if skills_dir.exists():
            for cat in sorted(skills_dir.iterdir()):
                if cat.is_dir():
                    for skill_file in cat.glob("*/SKILL.md"):
                        name = skill_file.parent.name
                        desc = ""
                        try:
                            content = skill_file.read_text()
                            m = re.search(r'description:\s*"(.*?)"', content)
                            if m:
                                desc = m.group(1)
                        except Exception:
                            pass
                        skill_list.append(f"  • **{name}** — {desc}")
        lines = ["**Installed Skills**", ""] + skill_list if skill_list else ["No skills installed."]
        return True, "\n".join(lines)

    if cmd == "/version":
        return True, "Hermes Mobile Bridge v2 — AgentLoop v3"

    if cmd == "/info":
        return True, f"**Session:** `{session_id}`\n**Status:** Connected ✅"

    return False, None


# ─── Rules / Project Context ─────────────────────────────────────────────


def load_project_rules() -> str:
    """Load project context rules from AGENTS.md / CLAUDE.md if present."""
    rules_parts = []
    for filename in ("AGENTS.md", "CLAUDE.md", ".cursorrules"):
        for root in [Path.cwd(), HERMES_HOME]:
            f = root / filename
            if f.exists():
                try:
                    text = f.read_text(encoding="utf-8")
                    # Only include if it's not too long
                    if len(text) < 5000:
                        rules_parts.append(text)
                    else:
                        rules_parts.append(text[:4996] + "\n... [project rules truncated]")
                except Exception:
                    pass
    combined = "\n\n".join(rules_parts) if rules_parts else ""
    if combined:
        logger.info(f"Loaded project rules ({len(combined)} chars)")
    return combined


# ─── Full System Prompt Assembly ─────────────────────────────────────────


def build_system_prompt(base_prompt: str, session_dir: str = "") -> str:
    """Assemble the complete system prompt: base + skills + rules."""
    parts = [base_prompt]

    # Add skills
    skills_part = load_skills_system_prompt()
    if skills_part:
        parts.append("\n\n" + skills_part)

    # Add rules
    rules_part = load_project_rules()
    if rules_part:
        parts.append("\n\n## Project Rules\n" + rules_part)

    # Add commands reference
    parts.append(
        "\n\n## Available Commands\n"
        "Users can type slash commands in the chat. "
        "The server handles these automatically — you will not see them. "
        "Commands: " + ", ".join(f"`{c}`" for c in COMMANDS)
    )

    return "\n\n".join(parts)
