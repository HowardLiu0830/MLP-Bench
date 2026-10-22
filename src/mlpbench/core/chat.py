"""One tool-free model call: the CoT mode, for every backend.

claude  `claude -p --disallowedTools "*"`, run inside the bwrap jail with no Lean and an
        empty working directory.
codex   `codex exec` with the shell / exec / browser features disabled, inside the same jail.
openai  a plain HTTP POST to an OpenAI-compatible `/chat/completions` (vLLM or OpenRouter).

The jail is belt and braces: the CLI flags already remove the tools, and the jail makes
sure that a flag the CLI silently ignores cannot turn into a read of the benchmark file or a
web fetch (the jail is offline; only the CLI's model API is allowlisted, see sandbox/egress.py).
Every call records the tool calls it saw; a CoT row with n_tool_calls > 0 is not CoT.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import BackendUnavailable
from .backends.base import CallResult
from .backends.retry import BACKOFF, run_with_retry
from .backends.codex import NO_NETWORK
from .backends.routing import (claude_proxy_env, codex_openrouter_overrides, codex_proxy_env,
                               codex_proxy_overrides)
from .config import load_dotenv, settings
from .jsonio import trim_log, write_json_atomic
from .paths import workspace_root
from .registry import ModelEntry
from .sandbox import agent_jail, egress_for, release_jail
from .streams import (Usage, claude_final_text, claude_tool_calls, claude_usage, codex_tool_calls,
                      codex_usage, parse_claude_stream, parse_codex_stream, text_from_codex_events)

__all__ = ["complete", "vllm_base_url", "ChatOptions", "COT_TIMEOUT"]

# Features that give codex a way to act rather than answer.
_CODEX_DISABLE = ("shell_tool", "unified_exec", "browser_use", "computer_use", "in_app_browser",
                  "apps", "plugins", "image_generation", "view_image")


def _default_max_tokens() -> int:
    # Reasoning models (Qwen3.x) think inside this budget, so a small one can end mid-thought
    # with an empty answer. The vLLM server's
    # window (serve_vllm.sh MAXLEN=65536) leaves room for 32k after the longest prompt.
    return int(os.environ.get("MLPBENCH_MAX_TOKENS", "32768"))


@dataclass
class ChatOptions:
    max_tokens: int = field(default_factory=_default_max_tokens)
    temperature: float | None = None
    # vLLM: Qwen3-style chat templates think by default; the reasoning goes to a separate
    # field when the server runs with --reasoning-parser, so leaving it on is harmless.
    extra_body: dict[str, Any] = field(default_factory=dict)


def vllm_base_url() -> str:
    load_dotenv()
    return os.environ.get("MLPBENCH_VLLM_URL") or f"http://127.0.0.1:{8000 + os.getuid() % 1000}/v1"


def _persist(raw_dir: Path | None, stem: str, *, prompt: str, stdout: str,
             meta: dict[str, Any]) -> Path | None:
    if raw_dir is None:
        return None
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{stem}.prompt.txt").write_text(prompt, encoding="utf-8")
    stream = raw_dir / f"{stem}.stream.jsonl"
    stream.write_text(stdout or "", encoding="utf-8")
    write_json_atomic(raw_dir / f"{stem}.meta.json", meta)
    return stream


def _scratch() -> Path:
    root = workspace_root() / "chat"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="cot_", dir=str(root)))


def _claude(entry: ModelEntry, system: str, user: str, timeout: int) -> tuple[CallResult, str]:
    b = settings().claude_bin
    if not b:
        raise BackendUnavailable("the `claude` CLI is not on PATH. Install it, or set CLAUDE_BIN.")
    cmd = [b, "-p", "--setting-sources", "", "--model", entry.model_id,
           "--output-format", "stream-json", "--verbose", "--no-session-persistence",
           "--disable-slash-commands", "--exclude-dynamic-system-prompt-sections",
           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           # "*" does not reach every built-in; name the two that survive it.
           "--disallowedTools", "*", "ListAgents", "ReportFindings",
           "--system-prompt", system]
    if entry.effort:
        cmd += ["--effort", entry.effort]
    ws = _scratch()
    cmd = agent_jail(cmd, workspace=ws, with_lean=False, egress=egress_for(entry), cli="claude")
    env = claude_proxy_env() if entry.uses_proxy else os.environ.copy()
    try:
        rc, out, err, attempts = run_with_retry(cmd, stdin=user, cwd=ws, env=env, timeout=timeout)
    finally:
        release_jail(cmd)
        shutil.rmtree(ws, ignore_errors=True)
    traj = parse_claude_stream(out)
    text = claude_final_text(traj)
    res = CallResult(text=str(text), usage=claude_usage(traj.final), returncode=rc,
                     stderr=trim_log(err or "", 4000), timed_out=rc is None, attempts=attempts,
                     tool_calls=claude_tool_calls(traj.events))
    return res, out


def _codex(entry: ModelEntry, system: str, user: str, timeout: int) -> tuple[CallResult, str]:
    b = settings().codex_bin
    if not b:
        raise BackendUnavailable("the `codex` CLI is not on PATH. Install it, or set CODEX_BIN.")
    ws = _scratch()
    cmd = [b, "-a", "never", "exec", "--cd", str(ws), "--skip-git-repo-check", "--ephemeral",
           *NO_NETWORK, "--json",
           # codex's own sandbox cannot start inside bwrap; the jail is the sandbox here,
           # and with the exec features disabled there is nothing to sandbox anyway.
           "--dangerously-bypass-approvals-and-sandbox"]
    for feat in _CODEX_DISABLE:
        cmd += ["--disable", feat]
    if entry.effort:
        cmd += ["-c", f"model_reasoning_effort={entry.effort}"]
    if entry.is_openrouter:
        cmd += codex_openrouter_overrides()
    elif entry.uses_proxy:
        cmd += codex_proxy_overrides()
    cmd += ["-m", entry.model_id, "-"]
    cmd = agent_jail(cmd, workspace=ws, with_lean=False, egress=egress_for(entry), cli="codex")
    env = codex_proxy_env() if entry.uses_proxy else os.environ.copy()
    prompt = f"{system}\n\n{user}" if system else user
    try:
        rc, out, err, attempts = run_with_retry(cmd, stdin=prompt, cwd=ws, env=env, timeout=timeout)
    finally:
        release_jail(cmd)
        shutil.rmtree(ws, ignore_errors=True)
    traj = parse_codex_stream(out)
    res = CallResult(text=text_from_codex_events(traj.events), usage=codex_usage(traj.final),
                     returncode=rc, stderr=trim_log(err or "", 4000), timed_out=rc is None,
                     attempts=attempts, tool_calls=codex_tool_calls(traj.events))
    return res, out


def _http(entry: ModelEntry, system: str, user: str, timeout: int,
          opts: ChatOptions) -> tuple[CallResult, str]:
    load_dotenv()
    if entry.route == "vllm":
        base, key = vllm_base_url(), os.environ.get("MLPBENCH_VLLM_KEY", "dummy")
    else:
        base, key = settings().openrouter_base_url, os.environ.get("OPENROUTER_API_KEY", "")
    extra = dict(opts.extra_body)
    if entry.route == "vllm" and os.environ.get("MLPBENCH_VLLM_THINKING", "1") == "0":
        # Qwen3.x chat templates think by default and can spend the whole budget doing it;
        # the paper's open-weights runs turned it off the same way (proxy extra_body).
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    body: dict[str, Any] = {
        "model": entry.model_id,
        "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": user}],
        "max_tokens": opts.max_tokens,
        **extra,
    }
    if opts.temperature is not None:
        body["temperature"] = opts.temperature
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})

    raw, err, rc, attempts = "", "", 1, 0
    deadline = time.time() + timeout
    for i, pause in enumerate((0,) + BACKOFF):
        if pause:
            time.sleep(min(pause, max(0.0, deadline - time.time())))
        attempts = i + 1
        left = deadline - time.time()
        if left <= 0:
            return CallResult(text="", usage=Usage(), returncode=None, timed_out=True,
                              stderr=err or "timeout", attempts=attempts), raw
        try:
            with urllib.request.urlopen(req, timeout=left) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            rc = 0
            break
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            err, rc = f"HTTP {exc.code}: {raw[:500]}", exc.code
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            err, rc = f"{type(exc).__name__}: {exc}", 1
            if isinstance(exc, urllib.error.URLError) and "refused" in str(exc).lower():
                err += f" -- is the server at {base} up? (running/open_source/serve_vllm.sh)"
                break

    text, usage = "", Usage()
    if rc == 0:
        try:
            data = json.loads(raw)
            choice = data["choices"][0]
            msg = choice["message"]
            text = msg.get("content") or ""
            if choice.get("finish_reason") == "length" and not text.strip():
                err = (f"hit max_tokens={opts.max_tokens} while still reasoning; no answer was "
                       f"written (raise MLPBENCH_MAX_TOKENS)")
            u = data.get("usage") or {}
            usage = Usage(input=int(u.get("prompt_tokens") or 0),
                          output=int(u.get("completion_tokens") or 0),
                          total=int(u.get("total_tokens") or 0))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            err, rc = f"unparseable response: {exc}: {raw[:300]}", 1
    return CallResult(text=text, usage=usage, returncode=rc, stderr=err,
                      attempts=attempts), raw


# CoT has no wall-clock limit (paper, App. B.1): a reply is bounded only by the model's own output
# cap. The subprocess and HTTP drivers need a number, so the default is one no call reaches.
COT_TIMEOUT = 7 * 24 * 3600


def complete(entry: ModelEntry, system: str, user: str, *, timeout: int = COT_TIMEOUT,
             raw_dir: Path | None = None, stem: str = "call",
             options: ChatOptions | None = None) -> CallResult:
    """One tool-free call. The raw stream, the prompt and a meta record go to raw_dir."""
    if entry.backend == "claude":
        res, raw = _claude(entry, system, user, timeout)
    elif entry.backend == "codex":
        res, raw = _codex(entry, system, user, timeout)
    elif entry.backend == "openai":
        res, raw = _http(entry, system, user, timeout, options or ChatOptions())
    else:
        raise ValueError(f"no CoT driver for backend {entry.backend!r}")
    res.stream_path = _persist(
        raw_dir, stem, prompt=f"# system\n{system}\n\n# user\n{user}", stdout=raw,
        meta={"backend": entry.backend, "route": entry.route, "model": entry.model_id,
              "effort": entry.effort, "mode": "cot", **res.as_dict()})
    return res
