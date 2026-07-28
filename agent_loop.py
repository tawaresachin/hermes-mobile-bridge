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
    ) -> AsyncGenerator[str, None]:
        """
        Run the agent loop for a new user query.
        Yields SSE-formatted events (text, tool_call, tool_result, [DONE]).
        """
        # Add user message
        all_messages = list(messages)
        all_messages.append({"role": "user", "content": query})

        iteration = 0
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            full_response = ""

            # --- Call AI ---
            async for event in self._call_ai(all_messages):
                event_type = event.get("type", "")
                if event_type == "error":
                    yield sse_text(event.get("content", "AI error"))
                    yield sse_done()
                    return
                elif event_type == "text":
                    full_response += event.get("content", "")
                    yield sse_text(event.get("content", ""))
                elif event_type == "tool_call":
                    # AI is requesting tool execution
                    yield sse_tool_call(event["tool_call"])
                    # We'll collect tool calls while streaming until we hit a
                    # "text_done" or "tool_calls_complete" signal
                    pass

                # Check if this chunk contains a complete tool call request
                # (The AI signals tool calls in its response text)

            # --- Parse tool calls from the AI response ---
            tool_calls = self._parse_tool_calls(full_response)

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

    async def _call_ai(self, messages: list[dict]) -> AsyncGenerator[dict, None]:
        """Stream from the AI provider, yielding events."""
        headers = {"Content-Type": "application/json"}
        if self.ai_api_key:
            headers["Authorization"] = f"Bearer {self.ai_api_key}"

        payload = {
            "model": self.ai_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
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
                            yield {"type": "text", "content": content}
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
