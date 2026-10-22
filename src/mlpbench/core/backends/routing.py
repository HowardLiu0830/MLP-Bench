from __future__ import annotations

import os
import urllib.error
import urllib.request

from ...errors import ProxyUnreachable
from ..config import load_dotenv, settings

__all__ = ["codex_openrouter_overrides", "codex_proxy_overrides", "codex_proxy_env",
           "claude_proxy_env", "check_proxy"]


def codex_openrouter_overrides(base_url: str | None = None) -> list[str]:
    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise KeyError(
            "OPENROUTER_API_KEY is not set, and this model routes through OpenRouter.\n"
            "Put it in a .env beside the benchmark data, or export it."
        )
    url = base_url or settings().openrouter_base_url
    return [
        "-c", "model_providers.openrouter.name=OpenRouter",
        "-c", f"model_providers.openrouter.base_url={url}",
        "-c", "model_providers.openrouter.env_key=OPENROUTER_API_KEY",
        "-c", "model_providers.openrouter.wire_api=responses",
        "-c", "model_provider=openrouter",
    ]


def codex_proxy_overrides(proxy_url: str | None = None) -> list[str]:
    """codex -> the LiteLLM proxy's OpenAI Responses endpoint (`local:` models)."""
    url = (proxy_url or settings().proxy_url).rstrip("/")
    return [
        "-c", "model_providers.litellm.name=LiteLLM",
        "-c", f"model_providers.litellm.base_url={url}/v1",
        "-c", "model_providers.litellm.env_key=MLPBENCH_PROXY_KEY",
        "-c", "model_providers.litellm.wire_api=responses",
        "-c", "model_provider=litellm",
    ]


def codex_proxy_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MLPBENCH_PROXY_KEY", "dummy")       # env_key must be set; LiteLLM ignores it
    return env


def claude_proxy_env(proxy_url: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = proxy_url or settings().proxy_url
    env["ANTHROPIC_AUTH_TOKEN"] = env.get("MLPBENCH_PROXY_KEY", "dummy")
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def check_proxy(url: str | None = None, *, timeout: float = 5.0) -> None:
    url = url or settings().proxy_url
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health/liveliness", timeout=timeout):
            return
    except (urllib.error.URLError, OSError) as exc:
        raise ProxyUnreachable(
            f"the LiteLLM proxy at {url} is not answering ({exc}).\n"
            "Start it with `mlpbench proxy start`, or set MLPBENCH_PROXY_URL."
        ) from exc
