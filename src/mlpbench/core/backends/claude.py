from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...errors import BackendUnavailable
from ..config import settings
from ..jsonio import trim_log, write_json_atomic
from ..sandbox import (agent_jail, release_jail, claude_agentic_args, door_reasons, egress_for,
                       suspect_reasons)
from ..streams import (Usage, claude_tool_calls, claude_tool_inputs, claude_usage,
                       claude_final_text, parse_claude_stream)
from .base import AgentRun
from ..pricing import price_for
from .retry import ActionCounter, AnyStop, CostMeter, run_with_retry
from .routing import claude_proxy_env

__all__ = ["agent"]

_BASE = [
    "-p",
    "--output-format", "stream-json", "--verbose",
    "--disable-slash-commands",
    "--exclude-dynamic-system-prompt-sections",
    "--no-session-persistence",
    "--setting-sources", "",
]


def _drop_partials(out: str) -> str:
    """`--include-partial-messages` streams every token delta; keep only the usage events."""
    keep = []
    for ln in out.splitlines(keepends=True):
        if '"stream_event"' in ln and '"message_delta"' not in ln and '"message_start"' not in ln:
            continue
        keep.append(ln)
    return "".join(keep)


def _binary() -> str:
    b = settings().claude_bin
    if not b or not Path(b).exists():
        raise BackendUnavailable(
            "the `claude` CLI is not on PATH. Install it, or set CLAUDE_BIN."
        )
    return b


def _persist(raw_dir: Path | None, stem: str, stdout: str, stderr: str,
             meta: dict[str, Any], *, prompt: str | None = None) -> Path | None:
    if raw_dir is None:
        return None
    raw_dir.mkdir(parents=True, exist_ok=True)
    stream = raw_dir / f"{stem}.stream.jsonl"
    stream.write_text(stdout or "", encoding="utf-8")
    if prompt is not None:
        (raw_dir / f"{stem}.prompt.txt").write_text(prompt, encoding="utf-8")
    write_json_atomic(raw_dir / f"{stem}.meta.json",
                      {**meta, "stderr": trim_log(stderr or "", 4000)})
    return stream


def agent(entry, system: str, user: str, *, workspace: Path, timeout: int = 1800,
          raw_dir: Path | None = None, stem: str = "agent",
          lean_project: Path | None = None, sandbox: bool = True,
          allow_unsandboxed: bool = False, extra_ro: tuple[Path, ...] = (),
          max_actions: int | None = None, max_cost: float | None = None) -> AgentRun:
    cmd = [_binary(), *_BASE, "--system-prompt", system, "--model", entry.model_id,
           "--dangerously-skip-permissions", *claude_agentic_args()]
    meter = None
    if max_cost:
        price = price_for(entry.model_id)
        if price is None:
            raise BackendUnavailable(f"no list price for {entry.model_id}; cannot cap its cost")
        meter = CostMeter(max_cost, price)
        cmd += ["--include-partial-messages"]       # per-reply usage, thinking included
    if entry.effort:
        cmd += ["--effort", entry.effort]

    workspace = Path(workspace)
    workspace = workspace if workspace.is_absolute() else (Path.cwd() / workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if sandbox:
        cmd = agent_jail(cmd, workspace=workspace, with_lean=True,
                         lean_project=lean_project, allow_unsandboxed=allow_unsandboxed,
                         extra_ro=extra_ro, egress=egress_for(entry), cli="claude")

    env = claude_proxy_env() if entry.uses_proxy else os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if entry.effort:
        env["CLAUDE_EFFORT"] = entry.effort

    counter = ActionCounter(max_actions) if max_actions else None
    stop = AnyStop(meter, counter) if (counter or meter) else None
    try:
        rc, out, err, attempts = run_with_retry(
            cmd, stdin=user, cwd=workspace, env=env, timeout=timeout, stop_when=stop,
        )
    finally:
        release_jail(cmd)
    if meter is not None:
        out = _drop_partials(out)

    traj = parse_claude_stream(out)
    text = claude_final_text(traj)
    tools = claude_tool_calls(traj.events)
    usage = claude_usage(traj.final)
    if meter is not None and not traj.final:          # killed: no result event, use the meter
        t = meter.tokens
        usage = Usage(input=t["input"], output=t["output"], cache_read=t["cache_read"],
                      cache_creation=t["cache_creation"],
                      total=t["input"] + t["output"] + t["cache_creation"],
                      cost_usd=round(meter.cost, 6))
    run = AgentRun(final_text=str(text), usage=usage, returncode=rc,
                   stderr=trim_log(err or "", 4000), timed_out=rc is None,
                   attempts=attempts, tool_calls=tools, workspace=workspace,
                   sandboxed=bool(sandbox),
                   egress_suspect_reasons=suspect_reasons(claude_tool_inputs(traj.events),
                                                          "claude")
                   + door_reasons(err, workspace))
    if counter is not None:
        run.n_actions, run.action_cap, run.action_cap_hit = counter.n, counter.cap or None, counter.hit
    if meter is not None:
        run.cost_usd, run.cost_cap, run.cost_cap_hit = round(meter.cost, 6), meter.cap, meter.hit
    run.stream_path = _persist(raw_dir, stem, out, err,
                               {"backend": "claude", "route": entry.route,
                                "model": entry.model_id, "mode": "agent", **run.as_dict()},
                               prompt=f"# system\n{system}\n\n# user\n{user}")
    return run
