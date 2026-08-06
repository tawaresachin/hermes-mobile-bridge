#!/usr/bin/env python3
"""Tool registry — base class and discovery for all Hermes agent tools."""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


TOOLS_DIR = Path(__file__).parent
MAX_TOOL_OUTPUT_CHARS = 5000  # Truncate tool results to this length


@dataclass
class ToolCall:
    """A tool invocation request from the AI."""
    id: str
    name: str
    arguments: dict[str, Any]
    status: str = "running"  # running, completed, failed


@dataclass
class ToolResult:
    """The result of executing a tool."""
    id: str
    name: str
    output: str = ""
    error: Optional[str] = None
    status: str = "completed"


class BaseTool(ABC):
    """Abstract base for all agent tools."""

    name: str = ""
    description: str = ""
    parameters: dict = {}  # JSON Schema for the tool's parameters

    @abstractmethod
    async def run(self, **kwargs) -> str:
        """Execute the tool. Returns text output."""
        ...

    def to_openai_tool(self) -> dict:
        """Return the tool definition in OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


# ─── Registry ───────────────────────────────────────────────────────────


class ToolRegistry:
    """Auto-discovers and holds all available tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._discover()

    def _discover(self):
        """Scan the tools directory for all BaseTool subclasses."""
        for importer, modname, ispkg in pkgutil.iter_modules([str(TOOLS_DIR)]):
            if modname.startswith("_") or modname == "__init__":
                continue
            try:
                module = importlib.import_module(f"tools.{modname}")
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and issubclass(obj, BaseTool)
                            and obj is not BaseTool):
                        instance = obj()
                        self._tools[instance.name] = instance
            except Exception as e:
                print(f"Warning: failed to load tool {modname}: {e}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def openai_tools(self) -> list[dict]:
        """Return all tool definitions in OpenAI format."""
        return [t.to_openai_tool() for t in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        tool = self.get_tool(call.name)
        if not tool:
            return ToolResult(
                id=call.id,
                name=call.name,
                error=f"Unknown tool: {call.name}",
                status="failed",
            )
        try:
            output = await tool.run(**call.arguments)
            # Truncate if too long
            if len(output) > MAX_TOOL_OUTPUT_CHARS:
                output = output[:MAX_TOOL_OUTPUT_CHARS] + (
                    f"\n... [truncated {len(output) - MAX_TOOL_OUTPUT_CHARS} chars]"
                )
            return ToolResult(id=call.id, name=call.name, output=output)
        except Exception as e:
            return ToolResult(id=call.id, name=call.name, error=str(e), status="failed")


# Singleton
registry = ToolRegistry()
