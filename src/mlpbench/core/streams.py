from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = [
    "Usage",
    "ToolCall",
    "iter_events",
    "parse_claude_stream",
    "parse_codex_stream",
    "text_from_claude_events",
    "claude_final_text",
    "text_from_codex_events",
    "claude_usage",
    "codex_usage",
    "claude_tool_calls",
    "codex_tool_calls",
]


@dataclass(frozen=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_creation: int = 0
    total: int = 0
    cost_usd: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input": self.input, "output": self.output,
            "cache_read": self.cache_read, "cache_creation": self.cache_creation,
            "total": self.total, "cost_usd": self.cost_usd,
        }


@dataclass
class ToolCall:
    tool: str
    detail: str = ""                 # first 300 characters, for display
    returncode: int | None = None
    ok: bool | None = None
    command: str = ""                # the whole shell command, for counting Lean calls


@dataclass
class Trajectory:
    events: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)   # every `result` event, in order


def iter_events(stdout: str) -> Iterator[dict[str, Any]]:
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def parse_claude_stream(stdout: str) -> Trajectory:
    t = Trajectory()
    for obj in iter_events(stdout):
        t.events.append(obj)
        if obj.get("type") == "result":
            t.final = obj
            t.results.append(obj)
    return t


def _claude_blocks(events: list[dict[str, Any]], role: str) -> Iterator[dict[str, Any]]:
    for ev in events:
        if ev.get("type") != role:
            continue
        content = (ev.get("message") or {}).get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict):
                    yield blk


def text_from_claude_events(events: list[dict[str, Any]]) -> str:
    parts = [b.get("text") or "" for b in _claude_blocks(events, "assistant")
             if b.get("type") == "text"]
    return "\n".join(p for p in parts if p)


def claude_final_text(traj: Trajectory) -> str:
    """The session's answer: every `result` text, in order. A session can end in more than one
    turn: when a background task the agent started times out, the CLI wakes the model again and
    that turn ("the task is no longer needed, my answer above stands") is the last `result`,
    while the answer is in an earlier one. Taking only the last dropped the answer (2026-10-20).
    Falls back to the assistant text when no `result` arrived (killed at a cap)."""
    texts = [r["result"] for r in traj.results if isinstance(r.get("result"), str) and r["result"]]
    return "\n".join(texts) or text_from_claude_events(traj.events) or ""


def claude_tool_calls(events: list[dict[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for blk in _claude_blocks(events, "assistant"):
        if blk.get("type") != "tool_use":
            continue
        inp = blk.get("input") or {}
        detail = (inp.get("command") or inp.get("file_path") or inp.get("pattern")
                  or inp.get("path") or inp.get("url") or "")
        calls.append(ToolCall(tool=str(blk.get("name") or "?"), detail=str(detail)[:300],
                              command=str(inp.get("command") or "")))

    errors = sum(1 for b in _claude_blocks(events, "user")
                 if b.get("type") == "tool_result" and b.get("is_error"))
    if errors and calls:
        for c in calls[-errors:]:
            c.ok = False
    return calls


def claude_tool_inputs(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(tool name, full JSON input) of every tool_use, untruncated, for the egress audit."""
    return [(str(b.get("name") or "?"), json.dumps(b.get("input") or {}, ensure_ascii=False))
            for b in _claude_blocks(events, "assistant") if b.get("type") == "tool_use"]


def codex_tool_inputs(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(item type, full command / payload) of every codex tool item, for the egress audit."""
    out: list[tuple[str, str]] = []
    for ev in events:
        if ev.get("type") != "item.started" and ev.get("type") != "item.completed":
            continue
        item = ev.get("item") or {}
        kind = str(item.get("type") or "")
        if kind in ("agent_message", "reasoning", "error", "todo_list", ""):
            continue
        body = item.get("command") if kind == "command_execution" else item
        out.append((kind, body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)))
    return list(dict.fromkeys(out))


def claude_usage(result_event: dict[str, Any]) -> Usage:
    u = (result_event or {}).get("usage") or {}
    inp = int(u.get("input_tokens") or 0)
    out = int(u.get("output_tokens") or 0)
    cr = int(u.get("cache_read_input_tokens") or 0)
    cc = int(u.get("cache_creation_input_tokens") or 0)
    cost = result_event.get("total_cost_usd")
    return Usage(input=inp, output=out, cache_read=cr, cache_creation=cc,
                 total=inp + out + cc,
                 cost_usd=float(cost) if isinstance(cost, (int, float)) else None)


def parse_codex_stream(stdout: str) -> Trajectory:
    t = Trajectory()
    for obj in iter_events(stdout):
        t.events.append(obj)
        if obj.get("type") == "turn.completed":
            t.final = obj
    return t


def text_from_codex_events(events: list[dict[str, Any]]) -> str:
    final = ""
    for ev in events:
        item = ev.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message" \
                and isinstance(item.get("text"), str):
            final = item["text"]
        elif ev.get("type") == "assistant_message" and isinstance(ev.get("text"), str):
            final = ev["text"]
    return final


def codex_tool_calls(events: list[dict[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for ev in events:
        if ev.get("type") not in ("item.completed", "item.started"):
            continue
        item = ev.get("item") or {}
        kind = item.get("type")
        if kind == "command_execution":
            rc = item.get("exit_code")
            rc = int(rc) if isinstance(rc, (int, float)) else None
            cmd = str(item.get("command") or "")
            calls.append(ToolCall(tool="shell", detail=cmd[:300], command=cmd,
                                  returncode=rc, ok=(rc == 0) if rc is not None else None))
        elif kind in ("file_change", "mcp_tool_call", "web_search"):
            detail = item.get("path") or item.get("tool") or item.get("query") or ""
            calls.append(ToolCall(tool=str(kind), detail=str(detail)[:300]))
    seen: dict[tuple[str, str], ToolCall] = {}
    for c in calls:
        key = (c.tool, c.command or c.detail)    # two long commands may share 300 chars
        if key not in seen or c.returncode is not None:
            seen[key] = c
    return list(seen.values())


def codex_usage(turn_completed: dict[str, Any]) -> Usage:
    u = (turn_completed or {}).get("usage") or {}
    inp = int(u.get("input_tokens") or 0)
    out = int(u.get("output_tokens") or 0)
    cr = int(u.get("cached_input_tokens") or u.get("cache_read_input_tokens") or 0)
    return Usage(input=inp, output=out, cache_read=cr, total=inp + out, cost_usd=None)


_TOKENS_USED = re.compile(r"tokens used[:\s]+([\d,]+)", re.I)


