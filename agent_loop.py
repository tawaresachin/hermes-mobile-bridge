#!/usr/bin/env python3
"""Agent loop — orchestrates the AI + tool execution cycle."""
from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import AsyncGenerator, Optional

import httpx

from tools import ToolCall, ToolResult, ToolRegistry, registry
from hermes_features import handle_command, build_system_prompt


def _hermes_compression() -> dict:
    """Load Hermes Agent's own compression settings so the bridge enforces the
    SAME per-session context headroom by default. Falls back to safe defaults:
        protect_last_n = 20  (messages kept verbatim)
        threshold      = 0.5 (start compressing once history ratio exceeds 0.5)
        target_ratio   = 0.2 (compress down to ~20% of context)
    Env vars (CONTEXT_RECENT_K / CONTEXT_SUMMARY_BATCH) still override."""
    comp = {"protect_last_n": 20, "threshold": 0.5, "target_ratio": 0.2}
    try:
        import yaml
        path = os.path.expanduser("~/.hermes/config.yaml")
        if os.path.exists(path):
            data = yaml.safe_load(open(path).read()) or {}
            c = data.get("compression") or {}
            if isinstance(c, dict):
                comp = {
                    "protect_last_n": int(c.get("protect_last_n", 20)),
                    "threshold": float(c.get("threshold", 0.5)),
                    "target_ratio": float(c.get("target_ratio", 0.2)),
                }
    except Exception:
        pass
    return comp

# Per-request AgentLoop instances share ONE long-lived client (the old code
# created a fresh AsyncClient per turn and never closed it — connection
# pools/fds accumulated until GC). 300s matches the original per-request
# timeout; the connection pool is reused across turns.
_LOOP_CLIENT: httpx.AsyncClient | None = None


def _get_shared_client() -> httpx.AsyncClient:
    global _LOOP_CLIENT
    if _LOOP_CLIENT is None or _LOOP_CLIENT.is_closed:
        _LOOP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
    return _LOOP_CLIENT


# Module-level per-session summary locks. AgentLoop is constructed per
# request, so instance-level locks never actually serialized anything —
# two concurrent turns on one session raced read→summarise→write on the
# summary file. Keyed by session_id, bounded below.
_SUMMARY_LOCKS: dict[str, asyncio.Lock] = {}

DEFAULT_SYSTEM_PROMPT = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. "
    "You assist users with a wide range of tasks including answering questions, "
    "writing and editing code, analyzing information, creative work, and executing "
    "actions via your tools. You communicate clearly, admit uncertainty when "
    "appropriate, and prioritize being genuinely useful over being verbose. "
    "Be targeted and efficient in your responses."
    "\n\n"
    "You run on the Hermes Mobile Bridge server and have access to real system tools: "
    "terminal (shell commands), file_ops (read/write/patch/search files), "
    "and web (search/extract URLs, multi-source research). "
    "When a task requires external action, use the appropriate tool."
    "\n\n"
    "For HEAVY or complex work — big coding projects, multi-step research, "
    "anything needing memory, skills, vision, or subagents — delegate to "
    "`hermes_task` (the full Hermes Agent on this machine). It is slower "
    "(tens of seconds to minutes) but far more capable. Use it instead of "
    "stumbling through many small tool calls yourself."
)

# Forced by the server at setup (CAVEMAN_STYLE=1 is written into the bridge's
# .env). Short answers = fewer tokens, faster replies, cheaper voice TTS.
CAVEMAN_STYLE_RULE = (
    "\n\n"
    "RESPONSE STYLE: Caveman-speak. Follow every rule."
    "\n"
    "1. Terse. Short sentences. Sentence fragments OK. One idea per line."
    "2. Drop filler words: no 'just', 'really', 'basically', 'of course', "
    "'happy to', 'certainly'. Drop articles: no 'a', 'an', 'the' where meaning "
    "stays clear. No hedging."
    "3. Short words, plain. No fancy word, no big word, no complicated grammar."
    "4. NO decorations: no emoji, no smiley, no kaomoji, no stars, no hearts, "
    "no arrow glyphs, no decorative markdown tables. Plain words + punctuation "
    "only. Keep it simple."
    "5. Keep code, function names, API names, CLI commands, and error strings "
    "VERBATIM. Never touch them."
    "6. If the user writes in another language, reply in that language, "
    "caveman style. Do not force English."
    "\n"
    "EXCEPT: For SECURITY warnings, irreversible/destructive actions, or any "
    "step where compression could mislead, use FULL clear sentences. Do not "
    "compress those."
)

# ─── SSE Event Types ────────────────────────────────────────────────────


def sse_text(content: str) -> str:
    return f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"


def sse_reasoning(content: str) -> str:
    """Stream DeepSeek's reasoning_content so the caller can persist it —
    DeepSeek requires it echoed back on the next turn."""
    return f"data: {json.dumps({'type': 'reasoning', 'content': content})}\n\n"


def sse_tool_call(tc: ToolCall) -> str:
    return f"data: {json.dumps({
        'type': 'tool_call',
        'id': tc.id,
        'name': tc.name,
        'arguments': json.dumps(tc.arguments),
        'status': tc.status,
    })}\n\n"


def sse_tool_result(tr: ToolResult) -> str:
    event = {
        'type': 'tool_result',
        'id': tr.id,
        'name': tr.name,
        'status': tr.status,
    }
    if tr.output:
        event['output'] = tr.output
    if tr.error:
        event['error'] = tr.error
    return f"data: {json.dumps(event)}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"


# ─── Agent Loop ─────────────────────────────────────────────────────────


class AgentLoop:
    """
    Full agent execution loop:
    1. Build messages from history + new query
    2. Call AI (streaming) — may return text AND tool calls
    3. If tool calls: execute each, feed results back, loop to step 2
    4. If no tool calls: yield final text and stop
    """

    MAX_ITERATIONS = 10  # Prevent infinite tool loops

    # ── Context compaction (learned from the JARVIS voice-assistant
    # architecture: never feed the whole history to the model) ──
    # Keep the most recent RECENT_K messages verbatim; older messages are
    # rolled into a cached summary that gets injected into the system prompt.
    # A new summarisation batch fires only every SUMMARY_BATCH messages, so
    # the added latency is amortised (one extra LLM call per batch).
    RECENT_K = 24
    SUMMARY_BATCH = 12
    SUMMARY_MAX_WORDS = 180

    def __init__(
        self,
        ai_base_url: str,
        ai_api_key: str,
        ai_model: str,
        system_prompt: str = "You are Hermes, a helpful AI assistant with access to tools. Use them when needed. Be concise.",
        tools: Optional[ToolRegistry] = None,
    ):
        self.ai_base_url = ai_base_url
        self.ai_api_key = ai_api_key
        self.ai_model = ai_model
        self.system_prompt = system_prompt
        self.tools = tools or registry
        # Shared client — one per process, NOT one per AgentLoop instance.
        # (Per-request clients leaked connection pools/fds on every turn.)
        self._http = _get_shared_client()
        # Context "headroom" tuning — env-forced at setup, always-on defaults.
        # Enforce Hermes Agent's OWN compression as the per-session default
        # (protect_last_n, threshold, target_ratio) so every app session keeps
        # the same context headroom Hermes uses. Env vars override.
        _headroom = _hermes_compression()
        self.recent_k = int(os.getenv("CONTEXT_RECENT_K", str(_headroom["protect_last_n"])))
        self.summary_batch = int(os.getenv("CONTEXT_SUMMARY_BATCH", str(AgentLoop.SUMMARY_BATCH)))
        # Caveman style: forced ON by the server (CAVEMAN_STYLE=1 written at
        # setup). Set 0 only to disable explicitly.
        self.caveman_style = os.getenv("CAVEMAN_STYLE", "1").lower() not in ("0", "false", "no", "off")

    async def run(
        self,
        messages: list[dict],
        query: str,
        session_id: str = "",
        attachment_url: str = "",
        attachment_type: str = "",
        multi_agent: bool = False,
        model_override: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Run the agent loop for a new user query.
        Yields SSE-formatted events (text, tool_call, tool_result, [DONE]).

        When multi_agent=True, the query is handed to the full Hermes agent
        with the ruflo swarm skill preloaded (8 parallel specialists) — a
        minutes-long run streamed back chunk by chunk.
        """
        # Check for slash commands first
        if not query or not query.strip():
            # Empty query — nothing to process
            yield sse_done()
            return

        # Handle special commands (sessions/config/model). Runs hermes-CLI
        # subprocesses (up to 15s) — keep them off the event loop.
        handled, cmd_response = await asyncio.to_thread(handle_command, query, session_id)
        if handled and cmd_response:
            yield sse_text(cmd_response)
            yield sse_done()
            return

        # ─── Multi-agent mode: ruflo swarm on the full Hermes agent ───
        # The swarm runs with the APP's session model (model_override) so
        # Telegram's hermes config never leaks into app-driven turns — no
        # mix between the two sides.
        if multi_agent:
            async for event in self._run_ruflo(query, model_override):
                yield event
            yield sse_done()
            return

        # Build enhanced system prompt with skills, commands, and rules.
        # Runs skills-loading subprocesses (up to 30s on cold cache) — must
        # NOT block the event loop, so hop to a worker thread.
        enhanced_prompt = await asyncio.to_thread(build_system_prompt, self.system_prompt)
        # Caveman style — server-forced (CAVEMAN_STYLE=1 at setup): short
        # replies save tokens on every turn and keep voice TTS crisp.
        if self.caveman_style:
            enhanced_prompt = enhanced_prompt + CAVEMAN_STYLE_RULE

        # Compact long histories: keep the recent window verbatim, roll the
        # older part into a cached summary appended to the system prompt
        # (JARVIS-style rolling context — keeps lengthy voice sessions fast).
        all_messages, summary_block = await self._compacted_context(list(messages), session_id)
        if summary_block:
            enhanced_prompt = enhanced_prompt + summary_block

        iteration = 0
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            full_response = ""
            full_reasoning = ""
            native_tool_calls: list[ToolCall] = []

            # --- Call AI ---
            async for event in self._call_ai(all_messages, system_prompt=enhanced_prompt):
                event_type = event.get("type", "")
                if event_type == "error":
                    yield sse_text(event.get("content", "AI error"))
                    yield sse_done()
                    return
                elif event_type == "text":
                    full_response += event.get("content", "")
                    yield sse_text(event.get("content", ""))
                elif event_type == "reasoning":
                    full_reasoning += event.get("content", "")
                    # Persist reasoning too — DeepSeek needs it echoed back
                    yield sse_reasoning(event.get("content", ""))
                elif event_type == "tool_call":
                    tc = event["tool_call"]
                    native_tool_calls.append(tc)
                    # Don't yield sse_tool_call yet — we'll do it during execution
                    # to avoid duplicate events (sse_tool_call also emitted there)

            # --- Collect tool calls from both sources ---
            # 1. Native delta.tool_calls from _call_ai()
            tool_calls = list(native_tool_calls)

            # 2. Text-based ```tool_call blocks (fallback for non-native models)
            text_calls = self._parse_tool_calls(full_response)
            # Deduplicate by name+arguments — text blocks could overlap with native
            text_ids = {(tc.name, json.dumps(tc.arguments, sort_keys=True)) for tc in tool_calls}
            for tc in text_calls:
                key = (tc.name, json.dumps(tc.arguments, sort_keys=True))
                if key not in text_ids:
                    tool_calls.append(tc)
                    text_ids.add(key)

            if not tool_calls:
                # AI is done — no more tool calls
                break

            # Persist any assistant text spoken alongside the tool calls —
            # otherwise the next AI iteration loses it (context loss).
            if full_response.strip():
                msg_entry: dict = {
                    "role": "assistant",
                    "content": full_response,
                }
                if full_reasoning:
                    msg_entry["reasoning_content"] = full_reasoning
                all_messages.append(msg_entry)

            # --- Execute each tool ---
            result_texts = []
            for tc in tool_calls:
                yield sse_tool_call(tc)  # status=running

                result = await self.tools.execute(tc)

                yield sse_tool_result(result)
                result_texts.append(f"Tool '{tc.name}' returned: {result.output or result.error}")

                # Add assistant tool call to conversation history
                tc_entry: dict = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }],
                }
                if full_reasoning:
                    tc_entry["reasoning_content"] = full_reasoning
                all_messages.append(tc_entry)
                # Add tool result to conversation history
                all_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.output or result.error or "Tool completed",
                })

            # Loop back to let AI process tool results

        # AI finished without (more) tool calls — we're done
        yield sse_done()

    # ─────────────────────────────────────────────────────────────────────
    # Multi-agent mode: ruflo swarm (full Hermes agent, 8 specialists)
    # ─────────────────────────────────────────────────────────────────────

    async def _run_ruflo(self, query: str, model_override: str = "") -> AsyncGenerator[str, None]:
        """Stream a query through `hermes chat -s ruflo` (the multi-agent
        swarm harness). Runs as an async subprocess, yielding each line as
        SSE text — minutes-long runs arrive progressively, not in a lump.

        Falls back to plain `hermes chat -q` if the ruflo skill is missing,
        so the toggle never hard-fails."""
        hermes_bin = os.getenv("HERMES_BIN", "hermes")
        # Run under the isolated 'swarm' profile: provider pinned to the
        # bridge's own Omnirouter router, so the app's model always resolves
        # correctly and Telegram's hermes config is never read or affected.
        cmd = [hermes_bin, "--profile", "swarm", "chat", "-q", query.strip()]
        if model_override:
            cmd += ["-m", model_override]
        cmd += ["-s", "ruflo"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        try:
            # Stream stdout line-by-line; drop ruflo/TUI noise (box art,
            # skill warnings, the "Session:/Resume" footer) so the app only
            # sees the answer.
            noise_prefixes = (
                "Warning: Unknown toolsets",
                "Query: ",
                "Initializing agent",
                "Resume this session with",
                "hermes --resume",
                "Session:",
                "Duration:",
                "Messages:",
            )
            box_borders = ("╭", "├", "╰", "─", "╮", "╯", "╼", "╽", "│", "┌", "┐", "└", "┘", "┼", "┬", "┴")
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip("\n")
                stripped = line.strip()
                if not stripped or stripped.startswith(box_borders):
                    continue
                if stripped.startswith(noise_prefixes):
                    continue
                yield sse_text(line)
        except asyncio.CancelledError:
            proc.kill()
            raise
        finally:
            try:
                await proc.wait()
            except Exception:
                pass
            # If the run failed (skill missing / CLI error), surface stderr
            # as a plain single-query fallback so the toggle still works.
            if proc.returncode not in (0, None):
                stderr = (await proc.stderr.read()).decode(errors="replace") if proc.stderr else ""
                if "no such skill" in stderr.lower() or "unknown skill" in stderr.lower():
                    yield sse_text("\n[ruflo skill missing — retrying with plain agent]")
                    async for ev in self._run_plain_hermes(query):
                        yield ev

    async def _run_plain_hermes(self, query: str) -> AsyncGenerator[str, None]:
        """Single-query fallback: `hermes chat -q` without skills."""
        hermes_bin = os.getenv("HERMES_BIN", "hermes")
        proc = await asyncio.create_subprocess_exec(
            hermes_bin, "--profile", "swarm", "chat", "-q", query.strip(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        try:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip("\n")
                if line.strip() and not line.strip().startswith("Session:"):
                    yield sse_text(line)
        except asyncio.CancelledError:
            proc.kill()
            raise
        finally:
            try:
                await proc.wait()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────
    # Context compaction: rolling summary for lengthy conversations
    # (JARVIS-style: recent window verbatim + cached older-summary block)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _summary_path(session_id: str):
        import os
        import hashlib
        # SECURITY + collision-safe: hash the client-controlled id so path
        # traversal ("../../x") and sanitised collisions ("abc/def" vs
        # "abcdef") are both impossible. Same store dir as the messages.
        store = os.getenv("STORE_PATH", os.path.expanduser("~/.hermes-mobile-server"))
        safe = hashlib.sha256((session_id or "").encode()).hexdigest()[:20]
        return os.path.join(store, f"summary_{safe}.json")

    def _load_summary_state(self, session_id: str) -> tuple[str, int]:
        """Return (summary_text, summarized_count) for the session."""
        try:
            p = self._summary_path(session_id)
            if os.path.exists(p):
                with open(p) as f:
                    data = json.load(f)
                return data.get("summary", ""), int(data.get("count", 0))
        except Exception:
            pass
        return "", 0

    def _save_summary_state(self, session_id: str, summary: str, count: int) -> None:
        try:
            import os
            p = self._summary_path(session_id)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump({"summary": summary, "count": count}, f)
        except Exception:
            pass

    async def _summarize(self, previous: str, chunk: list[dict]) -> str:
        """One LLM call: merge `chunk` into `previous`, return ≤180 words."""
        system = (
            "You are a conversation summariser for a voice assistant. "
            "Compress the conversation below into at most 180 words. "
            "Keep: user preferences, facts, decisions, names, tasks, and anything "
            "needed to continue later. Drop: greetings, filler, assistant deflections. "
            "If a previous summary is given, merge the new messages INTO it "
            "(update, don't repeat). Output ONLY the summary text."
        )
        parts = []
        if previous:
            parts.append(f"PREVIOUS SUMMARY:\n{previous}\n")
        for m in chunk:
            role = m.get("role", "user")
            content = str(m.get("content", "") or "")[:2000]
            if content.strip():
                parts.append(f"{role.upper()}: {content}")
        user_content = "\n".join(parts) or "(empty)"
        try:
            resp = await self._http.post(
                f"{self.ai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.ai_api_key}"},
                json={
                    "model": self.ai_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 500,
                    "stream": False,
                },
                timeout=httpx.Timeout(45.0),
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            words = text.split()
            if len(words) > self.SUMMARY_MAX_WORDS:
                text = " ".join(words[: self.SUMMARY_MAX_WORDS])
            return text.strip()
        except Exception:
            return previous  # fail-open: keep the old summary

    async def _compacted_context(self, messages: list[dict], session_id: str) -> tuple[list[dict], str]:
        """Return (messages_to_send, summary_block_to_append_to_system_prompt).

        When history exceeds RECENT_K + SUMMARY_BATCH, older messages are
        compressed into a rolling summary (cached per session, re-summarised
        in batches of SUMMARY_BATCH). Recent messages stay verbatim.
        """
        if not session_id or len(messages) <= self.recent_k + self.summary_batch:
            return messages, ""

        cutoff = len(messages) - self.recent_k
        older = messages[:cutoff]
        recent = messages[cutoff:]

        lock = _SUMMARY_LOCKS.setdefault(session_id, asyncio.Lock())
        async with lock:  # serialise read→summarise→write per session
            summary, count = self._load_summary_state(session_id)
            new_batch = older[count:]
            if len(new_batch) >= self.summary_batch:
                # Strip reasoning/tool-noise before summarising (keeps the call cheap)
                clean = [
                    {k: v for k, v in m.items() if k in ("role", "content")}
                    for m in new_batch
                ]
                summary = await self._summarize(summary, clean)
                self._save_summary_state(session_id, summary, len(older))
        # Bound the lock map (dicts preserve insertion order — drop the oldest
        # entries once it gets large; active sessions re-create their lock).
        if len(_SUMMARY_LOCKS) > 256:
            _SUMMARY_LOCKS.clear()

        if not summary:
            # Nothing summarised yet — keep a trimmed tail rather than drop
            # everything: the oldest SUMMARY_BATCH messages still fit.
            return recent, ""

        block = (
            "\n\n[EARLIER CONVERSATION SUMMARY — compressed context from earlier "
            "in this session. Use it to stay consistent; do not repeat what it "
            "already covers]:\n"
            + summary
        )
        return recent, block

    async def _call_ai(self, messages: list[dict], system_prompt: str | None = None) -> AsyncGenerator[dict, None]:
        """Stream from the AI provider, yielding events.
        Handles both delta.content (text) and delta.tool_calls (native tool calling).
        Accumulates tool call deltas across chunks by index.
        """
        headers = {"Content-Type": "application/json"}
        if self.ai_api_key:
            headers["Authorization"] = f"Bearer {self.ai_api_key}"

        sp = system_prompt or self.system_prompt

        payload = {
            "model": self.ai_model,
            "messages": [
                {"role": "system", "content": sp},
                *[m for m in messages if m.get("role") != "system"],
            ],
            "stream": True,
            "tools": self.tools.openai_tools(),
            "tool_choice": "auto",
        }

        try:
            accumulated_tool_calls: dict[int, dict] = {}

            def _flush_accumulated_tool_calls() -> list[dict]:
                """Emit fully-accumulated native tool calls as events.
                Only complete calls (with a function name) are flushed —
                a partial call cut off mid-stream is dropped rather than
                executed with garbage arguments."""
                events: list[dict] = []
                for tc_data in accumulated_tool_calls.values():
                    fn = tc_data.get("function", {})
                    fn_name = fn.get("name", "")
                    if not fn_name:
                        continue
                    tc_id = tc_data.get("id", str(uuid.uuid4())[:8])
                    fn_args_str = fn.get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
                    except json.JSONDecodeError:
                        fn_args = {}
                    events.append({
                        "type": "tool_call",
                        "tool_call": ToolCall(
                            id=tc_id,
                            name=fn_name,
                            arguments=fn_args,
                        ),
                    })
                return events

            async with self._http.stream(
                "POST",
                f"{self.ai_base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    yield {"type": "error", "content": f"API error {resp.status_code}: {error_text.decode()[:200]}"}
                    return

                # Accumulate tool call deltas across chunks
                # Keyed by index, each is {id, function: {name, arguments}}
                done_received = False

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        # Emit any fully accumulated tool calls before finishing
                        for ev in _flush_accumulated_tool_calls():
                            yield ev
                        done_received = True
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # Extract text content
                        content = delta.get("content", "")
                        if content:
                            yield {"type": "text", "content": content}

                        # Extract reasoning content (DeepSeek thinking mode).
                        # Must be echoed back to the API on the next turn,
                        # so it is emitted as its own event for the caller
                        # to persist alongside the message.
                        reasoning = delta.get("reasoning_content", "")
                        if reasoning:
                            yield {"type": "reasoning", "content": reasoning}

                        # Extract tool calls deltas (accumulate across chunks)
                        tool_calls_delta = delta.get("tool_calls")
                        if tool_calls_delta:
                            for tc_chunk in tool_calls_delta:
                                idx = tc_chunk.get("index", 0)
                                if idx not in accumulated_tool_calls:
                                    accumulated_tool_calls[idx] = {}
                                acc = accumulated_tool_calls[idx]

                                # id only appears on first chunk for this index
                                if "id" in tc_chunk:
                                    acc["id"] = tc_chunk["id"]

                                # function delta
                                fn_delta = tc_chunk.get("function", {})
                                if fn_delta:
                                    if "function" not in acc:
                                        acc["function"] = {}
                                    fn_acc = acc["function"]
                                    if "name" in fn_delta:
                                        fn_acc["name"] = fn_delta["name"]
                                    if "arguments" in fn_delta:
                                        fn_acc["arguments"] = fn_acc.get("arguments", "") + fn_delta["arguments"]

                    except json.JSONDecodeError:
                        continue

                # EOF without the [DONE] sentinel (some providers just close
                # the stream) — flush accumulated tool calls so a tool request
                # isn't silently dropped.
                if not done_received:
                    for ev in _flush_accumulated_tool_calls():
                        yield ev

        except Exception as e:
            yield {"type": "error", "content": f"Connection error: {e}"}

    def _parse_tool_calls(self, response_text: str) -> list[ToolCall]:
        """
        Parse tool call blocks from the AI response text.
        Format: ```tool_call\n{"name": "...", "arguments": {...}}\n```
        """
        calls = []
        pattern = r'```tool_call\s*\n?(.*?)\n?```'
        for match in re.finditer(pattern, response_text, re.DOTALL):
            try:
                data = json.loads(match.group(1).strip())
                calls.append(ToolCall(
                    id=data.get("id", str(uuid.uuid4())[:8]),
                    name=data["name"],
                    arguments=data.get("arguments", {}),
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        return calls
