from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = ["load_dotenv", "settings", "Settings", "redact", "have", "require"]

_LINE = re.compile(r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$""")

_loaded: set[Path] = set()


def _candidates() -> Iterable[Path]:
    explicit = os.environ.get("MLPBENCH_ENV_FILE")
    if explicit:
        yield Path(explicit).expanduser()
        return
    yield Path.cwd() / ".env"
    try:
        from .paths import data_root

        yield data_root() / ".env"
    except Exception:
        pass
    cfg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(cfg).expanduser() if cfg else Path.home() / ".config"
    yield base / "mlpbench" / "env"


def load_dotenv(*, override: bool = False) -> list[Path]:
    read: list[Path] = []
    for path in _candidates():
        if not path.is_file() or (path in _loaded and not override):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = _LINE.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if override or key not in os.environ:
                os.environ[key] = val
        _loaded.add(path)
        read.append(path)
    return read


def have(var: str) -> bool:
    load_dotenv()
    return bool(os.environ.get(var))


def require(var: str, why: str) -> str:
    load_dotenv()
    val = os.environ.get(var)
    if not val:
        raise KeyError(
            f"{var} is not set, and {why}.\n"
            f"Put it in a .env beside the benchmark data, or export it."
        )
    return val


@dataclass(frozen=True)
class Settings:
    claude_bin: str | None
    codex_bin: str | None
    bwrap_bin: str | None
    openrouter_base_url: str
    proxy_url: str
    has_openrouter_key: bool


def settings() -> Settings:
    load_dotenv()
    return Settings(
        claude_bin=os.environ.get("CLAUDE_BIN") or shutil.which("claude"),
        codex_bin=os.environ.get("CODEX_BIN") or shutil.which("codex"),
        bwrap_bin=os.environ.get("BWRAP_BIN") or shutil.which("bwrap"),
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL",
                                           "https://openrouter.ai/api/v1"),
        # Derived from the UID like the vLLM port, so two users on one box do not collide.
        proxy_url=os.environ.get("MLPBENCH_PROXY_URL",
                                 f"http://127.0.0.1:{4000 + os.getuid() % 1000}"),
        has_openrouter_key=bool(os.environ.get("OPENROUTER_API_KEY")),
    )


_SECRET_KEY = re.compile(r"(_KEY|_TOKEN|_SECRET|_PASSWORD)$", re.I)
_SECRET_VALUE = re.compile(r"\b(sk-or-v1-[A-Za-z0-9]+|sk-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,})")
_REDACTED = "<redacted>"


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: (_REDACTED if isinstance(k, str) and _SECRET_KEY.search(k) else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact(v) for v in obj)
    if isinstance(obj, str):
        return _SECRET_VALUE.sub(_REDACTED, obj)
    return obj
