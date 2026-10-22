from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass, replace

from ..errors import BackendUnavailable, ProxyUnreachable, UnknownModelError
from .config import have, settings

__all__ = ["ModelEntry", "REGISTRY", "resolve", "aliases"]

CODEX_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class ModelEntry:
    alias: str
    backend: str
    route: str
    model_id: str
    efforts: tuple[str, ...]
    default_effort: str | None = None
    effort: str | None = None
    note: str = ""

    @property
    def is_openrouter(self) -> bool:
        return self.route == "openrouter"

    @property
    def uses_proxy(self) -> bool:
        """A CLI entry whose traffic goes through the LiteLLM proxy: every `local:` model
        (codex), and the claude CLI on OpenRouter (`orc:`)."""
        return self.route == "local" or (self.backend == "claude" and self.route == "openrouter")

    @property
    def is_http(self) -> bool:
        """An OpenAI-compatible HTTP model (vLLM or OpenRouter): CoT only, no CLI agent."""
        return self.backend == "openai"


def _claude(alias: str, model_id: str, *, default: str | None = "medium",
            note: str = "") -> ModelEntry:
    return ModelEntry(alias, "claude", "native", model_id, CLAUDE_EFFORTS, default, None, note)


def _codex(alias: str, model_id: str, *, default: str | None = "medium",
           note: str = "") -> ModelEntry:
    return ModelEntry(alias, "codex", "native", model_id, CODEX_EFFORTS, default, None, note)


def _or_codex(alias: str, model_id: str, *, note: str = "") -> ModelEntry:
    return ModelEntry(alias, "codex", "openrouter", model_id, CODEX_EFFORTS, None, None, note)


def _or_claude(alias: str, model_id: str, *, note: str = "") -> ModelEntry:
    return ModelEntry(alias, "claude", "openrouter", model_id, CLAUDE_EFFORTS, None, None, note)


_ENTRIES: tuple[ModelEntry, ...] = (
    _claude("claude-opus-5", "claude-opus-5"),
    _claude("claude-sonnet-5", "claude-sonnet-5"),
    _claude("claude-opus-4-8", "claude-opus-4-8"),
    _claude("claude-sonnet-4-6", "claude-sonnet-4-6"),

    _codex("gpt-5.5", "gpt-5.5"),
    _codex("gpt-5.6-sol", "gpt-5.6-sol"),
    _codex("gpt-5.6-luna", "gpt-5.6-luna"),
    _codex("gpt-5.6-terra", "gpt-5.6-terra"),
    _codex("gpt-5.4-mini", "gpt-5.4-mini"),
    _codex("gpt-6.1-sol", "gpt-6.1-sol"),
    _codex("gpt-6-sol", "gpt-6-sol"),
    _codex("gpt-6-luna", "gpt-6-luna"),
    _codex("gpt-6-astra", "gpt-6-astra"),

    _or_codex("or:deepseek-v4-flash", "deepseek/deepseek-v4-flash",
              note="18 providers at 7.6-78.8 tok/s; pin provider.sort in the proxy config"),
    _or_claude("orc:deepseek-v4-flash", "openrouter/deepseek/deepseek-v4-flash",
               note="via the LiteLLM proxy"),
)

REGISTRY: dict[str, ModelEntry] = {e.alias: e for e in _ENTRIES}

# Open-weights / any OpenAI-compatible model, named on the command line rather than listed:
#   vllm:<hf-id>        HTTP to a local vLLM server (MLPBENCH_VLLM_URL)        -- CoT
#   openrouter:<id>     HTTP to OpenRouter (OPENROUTER_API_KEY)                -- CoT
#   local:<hf-id>       codex CLI -> LiteLLM proxy -> vLLM (MLPBENCH_PROXY_URL)  -- agentic
OPEN_PREFIXES = ("vllm:", "openrouter:", "local:")


def _open_entry(alias: str) -> ModelEntry | None:
    prefix, _, model = alias.partition(":")
    if not model or f"{prefix}:" not in OPEN_PREFIXES:
        return None
    if prefix == "vllm":
        return ModelEntry(alias, "openai", "vllm", model, (), None, None, "local vLLM server")
    if prefix == "openrouter":
        return ModelEntry(alias, "openai", "openrouter-api", model, (), None, None, "OpenRouter")
    return ModelEntry(alias, "codex", "local", f"local/{model}", CODEX_EFFORTS, None, None,
                      "codex CLI via the LiteLLM proxy to a local vLLM server")


def aliases() -> list[str]:
    return sorted(REGISTRY)


def _check_route(entry: ModelEntry) -> None:
    cfg = settings()
    if entry.backend == "openai":
        if entry.route == "openrouter-api" and not have("OPENROUTER_API_KEY"):
            raise UnknownModelError(
                f"{entry.alias} needs OPENROUTER_API_KEY; put it in .env or export it.")
        return
    bin_ = cfg.claude_bin if entry.backend == "claude" else cfg.codex_bin
    if not bin_:
        raise BackendUnavailable(
            f"{entry.alias} needs the `{entry.backend}` CLI, which is not on PATH.\n"
            f"Install it, or set {entry.backend.upper()}_BIN."
        )

    if entry.route == "native" and "/" in entry.model_id:
        raise UnknownModelError(
            f"{entry.alias}: model id {entry.model_id!r} is vendor-qualified but the route is "
            "'native'. A vendor-qualified id belongs on route=openrouter."
        )

    if entry.route == "local":
        _check_proxy(cfg.proxy_url)
        return
    if entry.route != "openrouter":
        return

    if not have("OPENROUTER_API_KEY"):
        raise UnknownModelError(
            f"{entry.alias} routes through OpenRouter but OPENROUTER_API_KEY is not set.\n"
            "Put it in a .env beside the benchmark data, or export it."
        )
    if entry.backend == "claude":
        _check_proxy(cfg.proxy_url)


def _check_proxy(url: str, timeout: float = 5.0) -> None:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health/liveliness", timeout=timeout):
            return
    except (urllib.error.URLError, OSError) as exc:
        raise ProxyUnreachable(
            f"the LiteLLM proxy at {url} is not answering ({exc}).\n"
            "Start it with `mlpbench proxy start`, or set MLPBENCH_PROXY_URL."
        ) from exc


def resolve(alias: str, *, effort: str | None = None, check: bool = True) -> ModelEntry:
    if not alias or not alias.strip():
        raise UnknownModelError("no model given. Legal aliases: " + ", ".join(aliases()))

    alias = alias.strip()
    entry = REGISTRY.get(alias) or _open_entry(alias)
    if entry is None:
        raise UnknownModelError(
            f"unknown model {alias!r}.\n"
            f"Legal aliases: {', '.join(aliases())}, or {'|'.join(OPEN_PREFIXES)}<model-id>\n"
            "A model that is not listed is reachable with "
            "--model-spec backend=...,route=...,id=... -- but say exactly what it is; "
            "this package does not guess a provider from the name."
        )

    chosen = effort if effort is not None else entry.default_effort
    if chosen is not None and chosen not in entry.efforts:
        raise UnknownModelError(
            f"{alias} ({entry.backend}) does not accept effort {chosen!r}. "
            f"Accepted: {', '.join(entry.efforts)}.\n"
            "The two CLIs differ here: 'minimal' is codex-only, 'max' is claude-only."
        )

    entry = replace(entry, effort=chosen)
    if check:
        _check_route(entry)
    return entry


def describe(entry: ModelEntry) -> dict[str, object]:
    return {
        "alias": entry.alias,
        "backend": entry.backend,
        "route": entry.route,
        "model_id": entry.model_id,
        "effort": entry.effort,
        "openrouter_base_url": settings().openrouter_base_url if entry.is_openrouter else None,
        "proxy_url": settings().proxy_url if entry.uses_proxy else None,
        "cli_versions": _cli_versions(entry.backend),
    }


def _cli_versions(backend: str) -> dict[str, str]:
    import subprocess

    if backend == "openai":
        return {}
    cfg = settings()
    bin_ = cfg.claude_bin if backend == "claude" else cfg.codex_bin
    if not bin_:
        return {}
    try:
        out = subprocess.run([bin_, "--version"], capture_output=True, text=True, timeout=30)
        return {backend: (out.stdout or out.stderr).strip()[:120]}
    except (OSError, subprocess.SubprocessError):
        return {}


def table() -> list[dict[str, object]]:
    return [
        {"alias": e.alias, "backend": e.backend, "route": e.route, "model_id": e.model_id,
         "efforts": ", ".join(e.efforts), "default_effort": e.default_effort, "note": e.note}
        for e in sorted(_ENTRIES, key=lambda e: (e.route, e.backend, e.alias))
    ]
