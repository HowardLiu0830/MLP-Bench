from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ...errors import BackendUnavailable
from ..config import settings
from ..jsonio import trim_log, write_json_atomic
from ..sandbox import agent_jail, door_reasons, egress_for, release_jail, suspect_reasons
from ..streams import (Usage, codex_tool_calls, codex_tool_inputs, codex_usage,
                       parse_codex_stream,
                       text_from_codex_events)
from .base import AgentRun
from ..paths import workspace_root
from ..pricing import price_for
from .retry import ActionCounter, AnyStop, CostMeter, run_with_retry
from .routing import codex_openrouter_overrides, codex_proxy_env, codex_proxy_overrides

__all__ = ["agent"]


def native_binary_problem(b: str) -> str | None:
    """Why `b` cannot be the codex process inside the jail, or None.

    The sandbox's API door serves only the process the jail execs (sandbox/egress.py). The npm
    package's `codex` is a node launcher that runs the native binary as a CHILD, and every API
    call from that child would be refused, so the run would fail on its first request."""
    try:
        with open(Path(b).resolve(), "rb") as f:
            head = f.read(2)
    except OSError as exc:
        return f"cannot read {b}: {exc}"
    if head == b"#!" or Path(b).resolve().suffix in (".js", ".mjs", ".cjs"):
        return (f"{b} is a script launcher (the npm shim), not the native codex binary; the "
                "sandbox only lets the exec'd process itself reach the API. Point CODEX_BIN at "
                "the native binary (e.g. ~/.codex/packages/standalone/current/bin/codex, or the "
                "npm package's vendor/<target>/codex/codex).")
    return None


def _binary() -> str:
    b = settings().codex_bin
    if not b or not Path(b).exists():
        raise BackendUnavailable("the `codex` CLI is not on PATH. Install it, or set CODEX_BIN.")
    problem = native_binary_problem(b)
    if problem:
        raise BackendUnavailable(problem)
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


# The network is closed by the jail itself (`--unshare-net`; the CLI's API calls leave through
# the allowlisting gateway in sandbox/egress.py). These flags are a second layer. codex 0.160
# ignores `tools.web_search=false`; `web_search="disabled"` is the key that version honours, and
# the old one is kept for older CLIs. Shell commands get a dead proxy instead of the CLI's
# gateway proxy, so even the allowlisted API hosts are not one env variable away.
_DEAD_PROXY = "http://127.0.0.1:9"
NO_NETWORK = (
    "-c", "tools.web_search=false",
    "-c", 'web_search="disabled"',
    *[a for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy",
                    "all_proxy")
      for a in ("-c", f'shell_environment_policy.set.{var}="{_DEAD_PROXY}"')],
    "-c", 'shell_environment_policy.set.NO_PROXY=""',
)

# Features that would let the model run code or fetch anything other than through the user
# shell (which the jail wraps into a network-less namespace): browsers, computer use,
# apps/plugins/MCP, hooks, dependency installers, sub-agents, image tools. None is needed for a
# Lean task.
# What this list does NOT do (codex 0.160): `--disable unified_exec`
# leaves the model's exec_command tool (and its `shell` parameter) in place, because the model
# metadata (shell_type "unified_exec") decides the tool set, and rewriting the cached metadata
# is undone by codex's refresh. code_mode_host must stay on: those models run every command
# from a JS cell (`functions.exec`), and without the host no command runs at all. codex maps
# `shell` to a known shell and execs /usr/bin/bash or /bin/sh -- both the wrapper -- and a
# process that gets past the wrapper is still refused at the door, which serves only sockets owned by the CLI process (sandbox/egress.py BRIDGE_PY).
AGENT_DISABLE: tuple[str, ...] = (
    "unified_exec", "unified_exec_tty",
    "browser_use", "browser_use_external", "browser_use_full_cdp_access", "computer_use",
    "in_app_browser", "in_app_chat", "in_app_local_automation",
    "apps", "plugins", "remote_plugin", "plugin_sharing", "tool_suggest",
    "tool_call_mcp_elicitation", "skill_mcp_dependency_install", "workspace_dependencies",
    "hooks", "multi_agent", "realtime_conversation", "daemon_auto_start", "worktrees",
    "image_generation", "view_image",
)


def agent(entry, system: str, user: str, *, workspace: Path, timeout: int = 1800,
          raw_dir: Path | None = None, stem: str = "agent",
          lean_project: Path | None = None, sandbox: bool = True,
          allow_unsandboxed: bool = False, extra_ro: tuple[Path, ...] = (),
          max_actions: int | None = None, max_cost: float | None = None) -> AgentRun:
    workspace = Path(workspace)
    workspace = workspace if workspace.is_absolute() else (Path.cwd() / workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    cmd = [
        _binary(), "-a", "never", "exec",
        "--cd", str(workspace),
        "--skip-git-repo-check",
        *NO_NETWORK,
        "--json",
    ]
    cmd += (["--dangerously-bypass-approvals-and-sandbox"] if sandbox
            else ["--sandbox", "workspace-write"])
    for feat in AGENT_DISABLE:
        cmd += ["--disable", feat]
    if entry.effort:
        cmd += ["-c", f"model_reasoning_effort={entry.effort}"]
    if entry.is_openrouter:
        cmd += codex_openrouter_overrides()
    elif entry.uses_proxy:
        cmd += codex_proxy_overrides()
    cmd += ["-m", entry.model_id, "-"]

    # The cost meter reads the session's rollout (codex prints usage only at the very end), so
    # a metered run keeps its rollout in a fresh per-run directory bound into the jail; an
    # unmetered run stays --ephemeral and writes none.
    meter = sessions = None
    if max_cost:
        price = price_for(entry.model_id)
        if price is None:
            raise BackendUnavailable(f"no list price for {entry.model_id}; cannot cap its cost")
        if not sandbox:
            raise BackendUnavailable("the codex cost cap needs the sandbox (it reads the rollout)")
        root = workspace_root() / "_codex_sessions"
        root.mkdir(parents=True, exist_ok=True)
        sessions = Path(tempfile.mkdtemp(prefix=f"{stem}.", dir=str(root)))
        meter = CostMeter(max_cost, price, rollout_dir=sessions)
    else:
        cmd.insert(cmd.index("--skip-git-repo-check") + 1, "--ephemeral")

    if sandbox:
        cmd = agent_jail(cmd, workspace=workspace, with_lean=True,
                         lean_project=lean_project, allow_unsandboxed=allow_unsandboxed,
                         extra_ro=extra_ro, codex_sessions=sessions,
                         egress=egress_for(entry), cli="codex")

    env = codex_proxy_env() if entry.uses_proxy else os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    prompt = f"{system}\n\n{user}" if system else user

    counter = ActionCounter(max_actions) if max_actions else None
    stop = AnyStop(meter, counter) if (counter or meter) else None
    try:
        rc, out, err, attempts = run_with_retry(cmd, stdin=prompt, cwd=workspace, env=env,
                                                timeout=timeout, stop_when=stop)
        if meter is not None:
            meter.finish()
        rollouts = sorted(sessions.rglob("rollout-*.jsonl")) if sessions else []
        if raw_dir is not None and rollouts:
            Path(raw_dir).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rollouts[-1], Path(raw_dir) / f"{stem}.rollout.jsonl")
    finally:
        release_jail(cmd)
        if sessions is not None:
            shutil.rmtree(sessions, ignore_errors=True)

    traj = parse_codex_stream(out)
    tools = codex_tool_calls(traj.events)
    usage = codex_usage(traj.final)
    if meter is not None:
        t = meter.tokens
        if not traj.final:                             # killed: no turn.completed, use the meter
            usage = Usage(input=t["input"] + t["cache_read"] + t["cache_creation"],
                          output=t["output"], cache_read=t["cache_read"],
                          cache_creation=t["cache_creation"],
                          total=t["input"] + t["cache_read"] + t["cache_creation"] + t["output"])
        usage = dataclasses.replace(usage, cost_usd=round(meter.cost, 6))
    run = AgentRun(final_text=text_from_codex_events(traj.events), usage=usage, returncode=rc,
                   stderr=trim_log(err or "", 4000), timed_out=rc is None,
                   attempts=attempts, tool_calls=tools, workspace=workspace,
                   sandboxed=bool(sandbox),
                   egress_suspect_reasons=suspect_reasons(codex_tool_inputs(traj.events),
                                                          "codex")
                   + door_reasons(err, workspace))
    if counter is not None:
        run.n_actions, run.action_cap, run.action_cap_hit = counter.n, counter.cap or None, counter.hit
    if meter is not None:
        run.cost_usd, run.cost_cap, run.cost_cap_hit = round(meter.cost, 6), meter.cap, meter.hit
    run.stream_path = _persist(raw_dir, stem, out, err,
                               {"backend": "codex", "route": entry.route,
                                "model": entry.model_id, "mode": "agent", **run.as_dict()},
                               prompt=f"# system\n{system}\n\n# user\n{user}")
    return run
