#!/usr/bin/env python3
"""Agent loop — orchestrates the AI + tool execution cycle."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import AsyncGenerator, Optional

import httpx

from tools import ToolCall, ToolResult, ToolRegistry, registry
from hermes_features import handle_command, build_system_prompt, COMMANDS

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
    "and web (search/extract URLs). "
    "When a task requires external action, use the appropriate tool."
)

# ─── SSE Event Types ────────────────────────────────────────────────────


def sse_text(content: str) -> str:
    return f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"


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
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    async def run(
        self,
        messages: list[dict],
        query: str,
        session_id: str = "",
        attachment_url: str = "",
        attachment_type: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Run the agent loop for a new user query.
        Yields SSE-formatted events (text, tool_call, tool_result, [DONE]).
        """
        # Check for slash commands first
        if not query or not query.strip():
            # Empty query — nothing to process
            yield sse_done()
            return

        handled, cmd_response = handle_command(query, session_id)
        if handled and cmd_response:
            yield sse_text(cmd_response)
            yield sse_done()
            return

        # Build enhanced system prompt with skills, commands, and rules
        enhanced_prompt = build_system_prompt(self.system_prompt)

        # Use messages from history — server already saved the query
        all_messages = list(messages)

        iteration = 0
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            full_response = ""
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

            # --- Execute each tool ---
            result_texts = []
            for tc in tool_calls:
                yield sse_tool_call(tc)  # status=running

                result = await self.tools.execute(tc)

                yield sse_tool_result(result)
                result_texts.append(f"Tool '{tc.name}' returned: {result.output or result.error}")

                # Add assistant tool call to conversation history
                all_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }],
                })
                # Add tool result to conversation history
                all_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.output or result.error or "Tool completed",
                })

            # Loop back to let AI process tool results

        # AI finished without (more) tool calls — we're done
        yield sse_done()

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
                accumulated_tool_calls: dict[int, dict] = {}

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        # Emit any fully accumulated tool calls before finishing
                        for tc_data in accumulated_tool_calls.values():
                            tc_id = tc_data.get("id", str(uuid.uuid4())[:8])
                            fn = tc_data.get("function", {})
                            fn_name = fn.get("name", "unknown")
                            fn_args_str = fn.get("arguments", "{}")
                            try:
                                fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
                            except json.JSONDecodeError:
                                fn_args = {}
                            yield {
                                "type": "tool_call",
                                "tool_call": ToolCall(
                                    id=tc_id,
                                    name=fn_name,
                                    arguments=fn_args,
                                ),
                            }
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


# For backward compatibility: simple streaming without tool loop
async def simple_chat_stream(
    messages: list[dict],
    ai_base_url: str,
    ai_api_key: str,
    ai_model: str,
    system_prompt: str = "You are Hermes, a helpful AI assistant. Be concise and accurate.",
) -> AsyncGenerator[str, None]:
    """Simple SSE streaming without tool execution (original behavior)."""
    headers = {"Content-Type": "application/json"}
    if ai_api_key:
        headers["Authorization"] = f"Bearer {ai_api_key}"

    payload = {
        "model": ai_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        try:
            async with client.stream(
                "POST",
                f"{ai_base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    yield f"data: {json.dumps({'type': 'text', 'content': f'⚠ API error {resp.status_code}'})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            yield f"data: {json.dumps({'type': 'text', 'content': f'⚠ Error: {e}'})}\n\n"

    yield "data: [DONE]\n\n"
