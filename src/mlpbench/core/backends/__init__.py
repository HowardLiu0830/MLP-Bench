from __future__ import annotations

import tempfile
from pathlib import Path

from ..registry import ModelEntry, resolve
from . import claude as _claude
from . import codex as _codex
from .base import AgentRun, ToolPolicy
from .retry import BACKOFF, TRANSIENT, run_with_retry, seconds_until_reset
from .routing import check_proxy, claude_proxy_env, codex_openrouter_overrides

__all__ = [
    "run_agent", "preflight",
    "CallResult", "AgentRun", "ToolPolicy",
    "check_proxy", "claude_proxy_env", "codex_openrouter_overrides",
    "run_with_retry", "seconds_until_reset", "TRANSIENT", "BACKOFF",
]

_DRIVER = {"claude": _claude, "codex": _codex}


def _driver(entry: ModelEntry):
    try:
        return _DRIVER[entry.backend]
    except KeyError:
        raise ValueError(f"no driver for backend {entry.backend!r}") from None


def run_agent(entry: ModelEntry, system: str, user: str, *, workspace: Path, **kw) -> AgentRun:
    return _driver(entry).agent(entry, system, user, workspace=workspace, **kw)


def preflight(alias: str, *, effort: str | None = None, timeout: int = 120) -> dict[str, object]:
    entry = resolve(alias, effort=effort)
    with tempfile.TemporaryDirectory(prefix="mlpbench_preflight_") as tmp:
        run = run_agent(entry, "Reply with exactly: ok", "ok",
                        workspace=Path(tmp) / "ws", timeout=timeout)
    return {
        "alias": entry.alias, "backend": entry.backend, "route": entry.route,
        "model_id": entry.model_id, "effort": entry.effort,
        "ok": run.returncode == 0, "returncode": run.returncode,
        "timed_out": run.timed_out,
        "text": (run.final_text or "").strip()[:200],
        "n_tool_calls": len(run.tool_calls),
        "usage": run.usage.as_dict(),
    }
