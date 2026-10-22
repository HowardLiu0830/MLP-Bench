from __future__ import annotations

import os
from pathlib import Path

from ..errors import DataRootNotFound

__all__ = [
    "data_root",
    "runs_root",
    "workspace_root",
    "lean_project",
    "forward_construction_jsonl",
    "backward_verification_jsonl",
    "auto_formalization_jsonl",
]

# The repo root is the directory holding both benchmark files.
_MARKERS: tuple[tuple[str, ...], ...] = (
    ("Benchmarks/forward-construction.jsonl",),
    ("Benchmarks/backward-verification.jsonl",),
)


def _marker_names() -> str:
    return ", ".join(" or ".join(alts) for alts in _MARKERS)


_MAX_ASCENT = 8


def _looks_like_data_root(d: Path) -> bool:
    return all(any((d / m).is_file() for m in alts) for alts in _MARKERS)


def _ascend(start: Path) -> Path | None:
    d = start if start.is_dir() else start.parent
    for _ in range(_MAX_ASCENT):
        if _looks_like_data_root(d):
            return d
        if d.parent == d:
            break
        d = d.parent
    return None


def data_root() -> Path:
    env = os.environ.get("MLPBENCH_DATA_ROOT")
    if env:
        d = Path(env).expanduser().resolve()
        if not _looks_like_data_root(d):
            raise DataRootNotFound(
                f"MLPBENCH_DATA_ROOT={d} does not hold the benchmark files.\n"
                f"Expected: {_marker_names()}"
            )
        return d

    for start in (Path.cwd(), Path(__file__).resolve()):
        found = _ascend(start)
        if found is not None:
            return found

    raise DataRootNotFound(
        "Could not locate the benchmark data.\n"
        f"Looked upward from {Path.cwd()} and from the installed package for a directory "
        f"holding: {_marker_names()}\n"
        "Set MLPBENCH_DATA_ROOT, or run from inside the repository.\n"
        "Note: a non-editable install cannot find the data by itself -- "
        "run ./setup.sh (editable install), or set the variable."
    )


def runs_root() -> Path:
    env = os.environ.get("MLPBENCH_RUNS_ROOT")
    return Path(env).expanduser().resolve() if env else (Path.cwd() / "runs").resolve()


def workspace_root() -> Path:
    env = os.environ.get("MLPBENCH_WORKSPACE")
    if env:
        return Path(env).expanduser().resolve()
    cache = os.environ.get("XDG_CACHE_HOME")
    if cache:
        return (Path(cache).expanduser() / "mlpbench" / "workspace").resolve()
    return Path(f"/tmp/mlpbench-{os.getuid()}/workspace").resolve()


def lean_project() -> Path:
    """The built Lean project: $MLPBENCH_LEAN_PROJECT, else ~/.local/share/mlpbench/lean_project.

    Deliberately outside the repo. The agent sandbox binds parts of this project, and bwrap
    materialises every parent of a bind; a project under the repo root would make the repo
    root itself appear inside the jail. `setup.sh --lean` builds it from `lean/`.
    """
    env = os.environ.get("MLPBENCH_LEAN_PROJECT")
    if env:
        return Path(env).expanduser().resolve()
    share = os.environ.get("XDG_DATA_HOME")
    base = Path(share).expanduser() if share else Path.home() / ".local" / "share"
    return (base / "mlpbench" / "lean_project").resolve()


def safe_upstream() -> Path:
    """The clone of Safe (github.com/liuchengwucn/Safe) whose `prompts/formalize.txt` fills the
    FL system prompt: $MLPBENCH_SAFE_UPSTREAM, else ~/.local/share/mlpbench/Safe, where
    `setup.sh` puts it. Safe ships no licence, so it is fetched, not vendored."""
    env = os.environ.get("MLPBENCH_SAFE_UPSTREAM")
    if env:
        return Path(env).expanduser().resolve()
    share = os.environ.get("XDG_DATA_HOME")
    base = Path(share).expanduser() if share else Path.home() / ".local" / "share"
    return (base / "mlpbench" / "Safe").resolve()


def forward_construction_jsonl() -> Path:
    return data_root() / "Benchmarks" / "forward-construction.jsonl"


def backward_verification_jsonl() -> Path:
    return data_root() / "Benchmarks" / "backward-verification.jsonl"


def auto_formalization_jsonl() -> Path:
    return data_root() / "Benchmarks" / "auto-formalization.jsonl"

