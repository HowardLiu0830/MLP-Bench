from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..streams import ToolCall, Usage

__all__ = ["CallResult", "AgentRun", "ToolPolicy"]

_LEAN_BINARY = re.compile(r"(?:^|[;&|]\s*|\bsudo\s+|\benv\s+(?:[A-Z_][A-Z0-9_]*=\S*\s+)*)"
                          r"(?:lake|lean)\b")


def _invokes_lean(command: str) -> bool:
    return bool(_LEAN_BINARY.search(command or ""))


class ToolPolicy:

    AGENTIC = "agentic"
    TOOL_FREE = "tool_free"


@dataclass
class CallResult:
    text: str
    usage: Usage
    returncode: int | None = None
    stderr: str = ""
    timed_out: bool = False
    attempts: int = 1
    tool_calls: list[ToolCall] = field(default_factory=list)
    stream_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and bool(self.text.strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "attempts": self.attempts,
            "usage": self.usage.as_dict(),
            "n_tool_calls": len(self.tool_calls),
            "stderr": self.stderr,
            "stream_path": str(self.stream_path) if self.stream_path else None,
        }


@dataclass
class AgentRun:
    final_text: str
    usage: Usage
    returncode: int | None = None
    stderr: str = ""
    timed_out: bool = False
    attempts: int = 1
    tool_calls: list[ToolCall] = field(default_factory=list)
    workspace: Path | None = None
    stream_path: Path | None = None
    sandboxed: bool = True
    n_actions: int | None = None          # actions (tool calls + shown messages), counted live when a cap is set
    action_cap: int | None = None
    action_cap_hit: bool = False
    cost_usd: float | None = None         # list-price cost, metered live when a cost cap is set
    cost_cap: float | None = None
    cost_cap_hit: bool = False
    # Tool calls that went looking for the CLI's API door (sandbox/egress.py suspect_reasons).
    egress_suspect_reasons: list[str] = field(default_factory=list)

    @property
    def egress_suspect(self) -> bool:
        return bool(self.egress_suspect_reasons)

    @property
    def n_shell_calls(self) -> int:
        return sum(1 for c in self.tool_calls if c.tool in ("shell", "Bash"))

    @property
    def n_lean_calls(self) -> int:
        return sum(1 for c in self.tool_calls
                   if c.tool in ("shell", "Bash") and _invokes_lean(c.command or c.detail))

    def as_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "attempts": self.attempts,
            "usage": self.usage.as_dict(),
            "sandboxed": self.sandboxed,
            "n_tool_calls": len(self.tool_calls),
            "n_shell_calls": self.n_shell_calls,
            "n_lean_calls": self.n_lean_calls,
            "stderr": self.stderr,
            "workspace": str(self.workspace) if self.workspace else None,
            "stream_path": str(self.stream_path) if self.stream_path else None,
            "n_actions": self.n_actions,
            "action_cap": self.action_cap,
            "action_cap_hit": self.action_cap_hit,
            "cost_usd": self.cost_usd,
            "cost_cap": self.cost_cap,
            "cost_cap_hit": self.cost_cap_hit,
            "egress_suspect": self.egress_suspect,
            "egress_suspect_reasons": self.egress_suspect_reasons[:20],
        }
