#!/usr/bin/env python3
"""
Hermes feature integration: skills loading, command parsing, rules,
memory, personalities, model management, and system prompt assembly —
matching the Telegram gateway behaviour.

Expanded to support full Telegram feature parity.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import urllib.request
import urllib.error
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

# ─── Config Helpers ────────────────────────────────────────────────────


def _read_config_yaml() -> dict:
    """Read and parse the Hermes config.yaml (simple YAML subset parser)."""
    config_path = HERMES_HOME / "config.yaml"
    if not config_path.exists():
        return {}
    # Prefer PyYAML if available — the hand-rolled parser can't handle lists
    try:
        import yaml
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    text = config_path.read_text(encoding="utf-8")
    # Simple nested key parser — handles basic YAML
    result = {}
    current_key = ""
    current_value: dict | str = {}
    stack = [result]
    indent_stack = [-1]
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        # Pop back to correct indent level
        while indent_stack and indent <= indent_stack[-1]:
            stack.pop()
            indent_stack.pop()
        if ":" in stripped.lstrip():
            key, _, val = stripped.lstrip().partition(":")
            key = key.strip()
            val = val.strip()
            # Array item?
            if key.startswith("-"):
                continue
            current_container = stack[-1]
            if val:
                current_container[key] = val.strip("\"'")
            else:
                new_dict = {}
                current_container[key] = new_dict
                stack.append(new_dict)
                indent_stack.append(indent)
    return result


def _get_hf_token() -> str:
    """Get the HuggingFace token from config or env."""
    # Try env first
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        return hf_token
    # Try from config.yaml via custom provider
    config_path = HERMES_HOME / "config.yaml"
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        for m in re.finditer(r'api_key:\s*["\']?(hf_[A-Za-z0-9]+)["\']?', text):
            return m.group(1)
    # Try huggingface token file
    hf_file = Path.home() / ".huggingface" / "token"
    if hf_file.exists():
        return hf_file.read_text().strip()
    return ""


def _get_provider_key() -> str:
    """Get the API key for the currently configured AI provider.

    Order: server's AI_API_KEY → OmniRoute key env → HF token.
    """
    key = os.environ.get("AI_API_KEY", "") or os.environ.get("HERMES_API_KEY", "")
    if key and key != "hermes123":  # skip the static bridge key — it's not a provider key
        return key
    or_key = os.environ.get("HERMES_CUSTOM_LOCALHOST_20128_API_KEY", "")
    if or_key:
        return or_key
    return _get_hf_token()


def _get_supermemory_key() -> str:
    """Get the supermemory API key from config."""
    config_path = HERMES_HOME / "config.yaml"
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        m = re.search(r'Authorization: Bearer (sm_[A-Za-z0-9_]+)', text)
        if m:
            return m.group(1)
    return os.environ.get("SUPERMEMORY_API_KEY", "")


def _get_personalities() -> dict[str, str]:
    """Return dict of {name: prompt} from config."""
    config_path = HERMES_HOME / "config.yaml"
    if not config_path.exists():
        return {}
    text = config_path.read_text(encoding="utf-8")
    # Find the personalities block using regex
    # Format: indent name: start of prompt\n  continuation lines...
    personalities = {}
    in_block = False
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "personalities:":
            in_block = True
            base_indent = len(line) - len(line.lstrip())
            i += 1
            continue
        if in_block:
            curr_indent = len(line) - len(line.lstrip())
            # Check if we left the block
            if stripped and curr_indent <= base_indent and ":" in stripped and not stripped.startswith("-"):
                in_block = False
                i += 1
                continue
            # Extract personality name and prompt
            if stripped and ":" in stripped and curr_indent > base_indent:
                parts = stripped.split(":", 1)
                name = parts[0].strip()
                prompt = parts[1].strip()
                # Collect continuation lines (more indented than this line)
                prompt_indent = len(line) - len(line.lstrip())
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    next_stripped = next_line.strip()
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_stripped and next_indent > prompt_indent and not next_stripped.startswith("-"):
                        prompt += " " + next_stripped
                        j += 1
                    else:
                        break
                if name:
                    personalities[name] = prompt
        i += 1
    return personalities


def _get_current_personality() -> str:
    """Get the currently active personality name."""
    config_path = HERMES_HOME / "config.yaml"
    if not config_path.exists():
        return "default"
    text = config_path.read_text(encoding="utf-8")
    m = re.search(r'^\s*personality:\s*(\S+)', text, re.MULTILINE)
    return m.group(1) if m else "default"


# ─── Supermemory MCP API ──────────────────────────────────────────────


def _supermemory_call(method: str, args: dict) -> dict | None:
    """Call the supermemory MCP server via JSON-RPC over HTTP."""
    api_key = _get_supermemory_key()
    if not api_key:
        return None
    payload = {
        "jsonrpc": "2.0",
        "id": int(time.time()),
        "method": "tools/call",
        "params": {
            "name": method,
            "arguments": args,
        },
    }
    req = urllib.request.Request(
        "https://mcp.supermemory.ai/mcp",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        logger.warning("Supermemory API error: %s", e)
        return None


def _memory_recall(query: str) -> str:
    """Search memories. Returns formatted text."""
    result = _supermemory_call("recall", {"query": query, "containerTag": "hermes-mobile"})
    if not result:
        return "⚠ Memory search failed (supermemory not configured)"
    content = result.get("content", [])
    if not content:
        # Try result  structure
        if "result" in result:
            content = result["result"]
    if not content:
        return "No memories found."
    # Format the response
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        lines = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", item.get("content", str(item)))
                lines.append(f"• {text[:500]}")
            else:
                lines.append(f"• {str(item)[:500]}")
        return "\n".join(lines) if lines else "No memories found."
    return str(content)[:2000]


def _memory_save(content: str) -> str:
    """Save a memory via supermemory."""
    result = _supermemory_call("memory", {
        "action": "save",
        "content": content,
        "containerTag": "hermes-mobile",
    })
    if not result:
        return "⚠ Failed to save memory (supermemory not configured)"
    return "✅ Memory saved!"


def _memory_forget(text: str) -> str:
    """Forget a memory via supermemory."""
    result = _supermemory_call("memory", {
        "action": "forget",
        "content": text,
        "containerTag": "hermes-mobile",
    })
    if not result:
        return "⚠ Failed to forget memory (supermemory not configured)"
    return "✅ Memory forgotten!"


# ─── Model Management ──────────────────────────────────────────────────


def _list_available_models() -> list[str]:
    """List available models from the current provider."""
    base_url = AI_BASE_URL.rstrip("/")
    # Use the configured provider API key, falling back to HF token for HF Router
    token = _get_provider_key()

    if not token:
        # Try unauthenticated (OmniRoute / OpenCode Zen need no auth)
        try:
            req = urllib.request.Request(
                f"{base_url}/models",
                headers={"User-Agent": "HermesMobileBridge/2.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            models_list = data.get("data", data) if isinstance(data, dict) else data
            return [
                m.get("id", m) if isinstance(m, dict) else m
                for m in models_list
            ]
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to list models (no auth): %s", e)
            return []

    # Authenticated request with the provider's key
    try:
        req = urllib.request.Request(
            f"{base_url}/models",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "HermesMobileBridge/2.0",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models_list = data.get("data", data) if isinstance(data, dict) else data
        return [
            m.get("id", m) if isinstance(m, dict) else m
            for m in models_list
        ]
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to list models: %s", e)
        return []


def _set_model(model_name: str) -> str:
    """Switch the model by updating the bridge .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        if "AI_MODEL=" in content:
            content = re.sub(r"AI_MODEL=.*", f"AI_MODEL={model_name}", content)
        else:
            content += f"\nAI_MODEL={model_name}"
        env_path.write_text(content, encoding="utf-8")
        return f"✅ Model switched to `{model_name}`. Restart bridge to apply."
    else:
        # Create .env with the model
        env_path.write_text(f"AI_MODEL={model_name}\n", encoding="utf-8")
        return f"✅ Model switched to `{model_name}`. Restart bridge to apply."


# ─── Skills ────────────────────────────────────────────────────────────


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
            logger.info("Loaded skills system prompt (%d chars)", len(result))
            return result
        else:
            stderr_text = proc.stderr.strip()
            if stderr_text:
                logger.warning("Skills loader stderr: %s", stderr_text[:200])
            else:
                logger.warning("Skills loader returned empty result")
            _skills_cache = ""
            return ""
    except Exception as e:
        logger.warning("Skills loader failed: %s", e)
        _skills_cache = ""
        return ""


def _invalidate_skills_cache():
    """Force reload of skills prompt on next call."""
    global _skills_cache
    _skills_cache = None


def _load_skill(name: str) -> str:
    """Load a specific skill by name into the cache then reload."""
    # Find the skill directory
    skills_dir = HERMES_HOME / "skills"
    if not skills_dir.exists():
        return f"⚠ No skills directory found at {skills_dir}"

    found = False
    for cat in sorted(skills_dir.iterdir()):
        if not cat.is_dir():
            continue
        for skill_dir in cat.glob(f"{name}/SKILL.md"):
            found = True
            break
        if found:
            break

    if not found:
        # Try case-insensitive search
        for cat in sorted(skills_dir.iterdir()):
            if not cat.is_dir():
                continue
            for skill_dir in cat.iterdir():
                if skill_dir.is_dir() and skill_dir.name.lower() == name.lower():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        found = True
                        name = skill_dir.name
                        break
            if found:
                break

    if not found:
        return f"⚠ Skill '{name}' not found. Use `/skills` to list available skills."

    _invalidate_skills_cache()
    # Trigger reload
    result = load_skills_system_prompt()
    if result:
        return f"✅ Skill `{name}` loaded into context. It will be active on your next message."
    return f"✅ Skill `{name}` found but failed to load. Try `/reset` to refresh."


def _list_skills() -> str:
    """List all installed skills."""
    skills_dir = HERMES_HOME / "skills"
    if not skills_dir.exists():
        return "No skills directory found."

    skill_list = []
    for cat in sorted(skills_dir.iterdir()):
        if not cat.is_dir():
            continue
        category = cat.name
        for skill_file in cat.glob("*/SKILL.md"):
            sname = skill_file.parent.name
            desc = ""
            try:
                content = skill_file.read_text()
                m = re.search(r'description:\s*"(.*?)"', content)
                if m:
                    desc = m.group(1)
            except Exception:
                pass
            skill_list.append(f"  • `{sname}` — {desc}")

    if not skill_list:
        return "No skills installed."

    lines = ["**Installed Skills**", ""] + skill_list
    lines.append("")
    lines.append("Use `/skills load <name>` to activate a skill.")
    return "\n".join(lines)


# ─── Personality Management ────────────────────────────────────────────


def _list_personalities() -> str:
    """Return formatted list of available personalities."""
    personalities = _get_personalities()
    if not personalities:
        return "No personalities configured."
    current = _get_current_personality()
    lines = ["**Available Personalities**", ""]
    for name in sorted(personalities.keys()):
        marker = "⭐ " if name == current else "  "
        prompt_preview = personalities[name][:60].replace("\n", " ")
        lines.append(f"{marker}`{name}` — {prompt_preview}…")
    lines.append("")
    lines.append(f"Current: **{current}**")
    lines.append("Use `/personality <name>` to switch.")
    return "\n".join(lines)


def _set_personality(name: str) -> str:
    """Switch personality by updating config.yaml via hermes CLI."""
    personalites = _get_personalities()
    if name not in personalites:
        available = ", ".join(sorted(personalites.keys()))
        return f"⚠ Unknown personality '{name}'. Available: {available}"

    try:
        result = subprocess.run(
            ["hermes", "config", "set", "agent.personality", name],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return f"✅ Personality switched to **{name}**! It will apply on your next message."
        else:
            return f"⚠ Failed to switch personality: {result.stderr.strip() or result.stdout.strip()}"
    except Exception as e:
        return f"⚠ Error: {e}"


# ─── Hermes CLI Delegation ────────────────────────────────────────────


def _hermes_cli(*args: str) -> str:
    """Run a hermes CLI command and return its output."""
    try:
        result = subprocess.run(
            ["hermes", *args],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "✅ Done (no output)"
        else:
            return result.stderr.strip() or result.stdout.strip() or f"⚠ Command failed (exit {result.returncode})"
    except subprocess.TimeoutExpired:
        return "⚠ Command timed out"
    except FileNotFoundError:
        return "⚠ `hermes` CLI not found"
    except Exception as e:
        return f"⚠ Error: {e}"


# ─── Status ────────────────────────────────────────────────────────────


def _status() -> str:
    """Full system status."""
    personality = _get_current_personality()
    lines = [
        "**System Status**",
        "",
        f"🤖 **Model:** `{AI_MODEL}`",
        f"🔗 **Provider:** `{AI_BASE_URL}`",
        f"🎭 **Personality:** `{personality}`",
    ]
    # Add memory status
    sm_key = _get_supermemory_key()
    lines.append(f"🧠 **Memory:** {'Online' if sm_key else 'Not configured'}")
    # Add skills status
    lines.append(f"📚 **Skills:** {'Loaded' if _skills_cache else 'Not loaded'}")
    # Add uptime info from ps
    try:
        pid_path = Path(__file__).parent / "server.pid"
        if pid_path.exists():
            pid = pid_path.read_text().strip()
            lines.append(f"⚡ **Uptime:** PID {pid}")
    except Exception:
        pass
    return "\n".join(lines)


# ─── Sessions ──────────────────────────────────────────────────────────


def _list_sessions(limit: int = 10) -> str:
    """List recent sessions via hermes CLI."""
    output = _hermes_cli("sessions", "list", "--limit", str(limit))
    if output.startswith("⚠"):
        return output
    # Format nicely
    lines = output.splitlines()
    if len(lines) <= 1:
        return "No sessions found."
    formatted = ["**Recent Sessions**", ""]
    for line in lines[1:limit + 1]:
        if line.strip():
            formatted.append(f"  {line.strip()}")
    return "\n".join(formatted)


# ─── Config ────────────────────────────────────────────────────────────


def _config_get(key: str = "") -> str:
    """Get a config value."""
    if key:
        output = _hermes_cli("config", "get", key)
        return f"**Config {key}:**\n```\n{output}\n```"
    else:
        output = _hermes_cli("config", "show")
        # Truncate if needed
        if len(output) > 3000:
            output = output[:3000] + "\n... [truncated]"
        return f"**Configuration:**\n```\n{output}\n```"


# ─── Command registry ─────────────────────────────────────────────────


COMMANDS = {
    # Session
    "/help": "Show available commands and tips.",
    "/reset": "Start a fresh conversation (clears history).",
    "/new": "Same as /reset.",
    "/retry": "Retry the last response.",
    "/clear": "Clear the current session.",
    "/sessions": "List recent sessions.",
    # Model & Config
    "/model": "Show/set the AI model. Usage: `/model` or `/model <name>`",
    "/config": "View configuration. Usage: `/config` or `/config <key>`",
    "/status": "Show system status (model, provider, personality, memory).",
    "/info": "Show session info.",
    # Personality
    "/personality": "List/switch personalities. Usage: `/personality` or `/personality <name>`",
    # Memory
    "/memory": "Save/recall/forget memories. Usage: `/memory save <text>`, `/memory recall <query>`, `/memory forget <text>`",
    # Skills
    "/skills": "List installed skills or load one with `/skills load <name>`",
    # Other
    "/version": "Show version info.",
}


def handle_command(query: str, session_id: str) -> tuple[bool, str | None]:
    """Check if query is a slash command. Returns (handled, response_text).
    If handled=True, the caller should send response_text as the assistant reply.
    """
    if not query or not query.strip():
        return False, None

    parts = query.strip().split()
    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []

    # ─── Session Commands ───────────────────────────────────────────

    if cmd in ("/help", "/start"):
        lines = [
            "**Hermes Mobile — Commands**",
            "",
            "**📝 Session**",
            f"  `{'`, `'.join(['/help', '/reset', '/new', '/clear', '/retry', '/sessions'])}`",
            "",
            "**🤖 Model & Config**",
            f"  `{'`, `'.join(['/model', '/config', '/status', '/info'])}`",
            "",
            "**🎭 Personality**",
            f"  `{'`, `'.join(['/personality'])}`",
            "",
            "**🧠 Memory**",
            f"  `{'`, `'.join(['/memory'])}`",
            "",
            "**📚 Skills**",
            f"  `{'`, `'.join(['/skills'])}`",
            "",
            "Use `/command help` or `/command ?` for usage details on any command.",
            "",
            "Pro tip: Just chat naturally — the agent handles tools, web search, and file access automatically!",
        ]
        return True, "\n".join(lines)

    if cmd in ("/reset", "/new"):
        # Actually clear the session history so the model starts fresh
        from server import _messages_path  # circular-safe import
        try:
            msgs_path = _messages_path(session_id)
            if msgs_path and msgs_path.exists():
                msgs_path.unlink()
        except Exception:
            pass
        return True, "✅ Session reset. Starting fresh."

    if cmd == "/retry":
        return True, (
            "⚠️ Retry is handled by the mobile app — tap the last message "
            "and select 'Retry' from the menu. The server will regenerate."
        )

    if cmd == "/clear":
        from server import _messages_path
        try:
            msgs_path = _messages_path(session_id)
            if msgs_path and msgs_path.exists():
                msgs_path.unlink()
        except Exception:
            pass
        return True, "✅ Session cleared."

    # ─── Model ──────────────────────────────────────────────────────

    if cmd == "/model":
        if args and args[0] in ("help", "?"):
            return True, (
                "**/model** — View or change the AI model.\n\n"
                "`/model` — Show current model\n"
                "`/model <name>` — Switch to a named model (requires restart)\n"
                "`/model list` — List available models from the provider"
            )
        if args and args[0] == "list":
            available = _list_available_models()
            if not available:
                return True, "⚠ Could not fetch model list from provider."
            # Filter to show interesting models (skip large ones, highlight small)
            lines = ["**Available Models**", ""]
            for m in available:
                # Highlight vision models
                icon = "🖼️ " if "vl" in m.lower() or "vision" in m.lower() else ""
                lines.append(f"  {icon}`{m}`")
            lines.append("")
            lines.append(f"**Current:** `{AI_MODEL}`")
            lines.append("Use `/model <name>` to switch (requires restart).")
            return True, "\n".join(lines)
        elif args:
            model_name = args[0]
            # Session-scoped switching is now handled at server level.
            # This fallback is for non-streaming paths that hit the agent loop.
            result = _set_model(model_name)
            return True, result.replace("Restart bridge to apply", "Will apply to future queries (use streaming endpoint for instant switch)")
        else:
            return True, f"**Current Model:** `{AI_MODEL}`\n**Provider:** `{AI_BASE_URL}`\n\nUse `/model list` to see available models.\nUse `/model <name>` to switch."

    # ─── Memory ─────────────────────────────────────────────────────

    if cmd == "/memory":
        if not args:
            return True, (
                "**/memory** — Persistent memory management\n\n"
                "`/memory recall <query>` — Search your memories\n"
                "`/memory save <text>` — Save a new memory\n"
                "`/memory forget <text>` — Forget a memory\n"
                "`/memory` — Show this help"
            )
        subcmd = args[0].lower()
        rest = " ".join(args[1:]).strip()

        if subcmd in ("recall", "search", "find"):
            if not rest:
                return True, "Usage: `/memory recall <search query>`"
            result = _memory_recall(rest)
            return True, f"**Memory Recall**\n\n{result}"

        elif subcmd in ("save", "store", "remember"):
            if not rest:
                return True, "Usage: `/memory save <text to remember>`"
            result = _memory_save(rest)
            return True, result

        elif subcmd in ("forget", "delete", "remove"):
            if not rest:
                return True, "Usage: `/memory forget <text to forget>`"
            result = _memory_forget(rest)
            return True, result

        else:
            return True, f"Unknown subcommand: `{subcmd}`. Use `/memory` for help."

    # ─── Personality ────────────────────────────────────────────────

    if cmd == "/personality":
        if args and args[0] in ("help", "?"):
            return True, (
                "**/personality** — List or switch personality.\n\n"
                "`/personality` — Show all personalities\n"
                "`/personality <name>` — Switch to a personality"
            )
        if args:
            return True, _set_personality(args[0])
        else:
            return True, _list_personalities()

    # ─── Status ─────────────────────────────────────────────────────

    if cmd == "/status":
        return True, _status()

    # ─── Skills ─────────────────────────────────────────────────────

    if cmd == "/skills":
        if args and args[0] in ("help", "?"):
            return True, (
                "**/skills** — Manage skills.\n\n"
                "`/skills` — List installed skills\n"
                "`/skills load <name>` — Load a skill into context\n"
            )
        if args and args[0] == "load":
            if len(args) < 2:
                return True, "Usage: `/skills load <skill-name>`"
            return True, _load_skill(args[1])
        else:
            return True, _list_skills()

    # ─── Sessions ───────────────────────────────────────────────────

    if cmd == "/sessions":
        if args and args[0] in ("help", "?"):
            return True, (
                "**/sessions** — Browse recent conversations.\n\n"
                "`/sessions` — List recent sessions\n"
                "`/sessions <N>` — List last N sessions"
            )
        limit = 10
        if args:
            try:
                limit = min(int(args[0]), 50)
            except ValueError:
                pass
        return True, _list_sessions(limit)

    # ─── Config ─────────────────────────────────────────────────────

    if cmd == "/config":
        if args and args[0] in ("help", "?"):
            return True, (
                "**/config** — View configuration.\n\n"
                "`/config` — Show full config\n"
                "`/config <key>` — Show specific config key\n"
                "Example: `/config model.default`"
            )
        key = " ".join(args) if args else ""
        return True, _config_get(key)

    # ─── Info / Version ─────────────────────────────────────────────

    if cmd == "/version":
        return True, "Hermes Mobile Bridge v2 — AgentLoop v3 (Telegram-compatible)"

    if cmd == "/info":
        lines = [
            f"**Session Info**",
            f"  • **ID:** `{session_id}`",
            f"  • **Model:** `{AI_MODEL}`",
            f"  • **Provider:** `{AI_BASE_URL}`",
            f"  • **Personality:** `{_get_current_personality()}`",
            f"  • **Status:** Connected ✅",
        ]
        return True, "\n".join(lines)

    # Unknown command — show help hint
    if cmd.startswith("/"):
        return True, (
            f"⚠ Unknown command: `{cmd}`\n"
            f"Use `/help` to see available commands."
        )

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
        logger.info("Loaded project rules (%d chars)", len(combined))
    return combined


# ─── Full System Prompt Assembly ─────────────────────────────────────────


def build_system_prompt(base_prompt: str, session_dir: str = "") -> str:
    """Assemble the complete system prompt: base + skills + rules."""
    parts = [base_prompt]

    # Add warm personality
    personality_name = _get_current_personality()
    personalities = _get_personalities()
    if personality_name in personalities:
        parts.append(f"\n\n{personalities[personality_name]}")

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
        "Commands: " + ", ".join(f"`{c}`" for c in sorted(COMMANDS.keys()))
    )

    return "\n\n".join(parts)
