from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from ..core import backends as B
from ..core.registry import ModelEntry, resolve
from ..errors import UnknownModelError

__all__ = ["complete_agentic", "resolve_model", "egress_audit"]

# complete_agentic returns (rc, text) to the adapter; the egress audit of every session an item
# opens is collected here instead, per worker thread, and read back by verify.task.
_AUDIT = threading.local()


def egress_audit(reset: bool = False) -> list[str]:
    """egress_suspect_reasons of the agent sessions run on this thread since the last reset."""
    got = list(getattr(_AUDIT, "reasons", []))
    if reset:
        _AUDIT.reasons = []
    return got


def resolve_model(backend: str | None, model: str | None, effort: str | None) -> ModelEntry:
    if not model:
        raise UnknownModelError(
            "no model given. Pass a registry alias (see `mlpbench models`)."
        )
    entry = resolve(model, effort=effort)
    if backend and backend != entry.backend:
        raise UnknownModelError(
            f"--backend {backend} disagrees with the registry, which drives {model!r} "
            f"through {entry.backend}. Drop --backend; the registry decides."
        )
    return entry


def _session_paths(session_path: str | Path | None) -> tuple[Path | None, str]:
    if not session_path:
        return None, "session"
    p = Path(session_path)
    return p.parent, p.name.split(".")[0] or "session"


def _repl_paths() -> tuple[Path, ...]:
    """What the seeded `lean_check_cli.py` needs inside the jail: the REPL binary's directory
    and, if staged, the node-local Mathlib packages. Without them the session's checker cannot
    start and the agent falls back to `lake env lean` (the adapter's verdicts are unaffected:
    it re-checks everything outside the jail)."""
    from .common import lean_repl as LR

    out = [LR.REPL_BIN.parent] if LR.REPL_BIN.is_file() else []
    if LR.LOCAL_PACKAGES.is_dir():
        out.append(LR.LOCAL_PACKAGES)
    return tuple(out)


def _pin_repl_env() -> None:
    """Point the seeded checker at the host's REPL by path. Its default, /dev/shm/$USER-lean,
    breaks in the jail: the user name may not resolve there (no NSS service, e.g. an LDAP
    account), and codex's login shell then resets $USER to the uid. The CLI passes these
    variables on to its shells."""
    from .common import lean_repl as LR

    os.environ.setdefault("LEAN_LOCAL_ROOT", str(LR.LOCAL_ROOT))
    if LR.REPL_BIN.is_file():
        os.environ.setdefault("LEAN_REPL_BIN", str(LR.REPL_BIN))


def complete_agentic(prompt: str, system: str = "", backend: str | None = None,
                     model: str | None = None, effort: str | None = None,
                     timeout: int = 3600, seed_files: dict[str, str] | None = None,
                     session_path: str | Path | None = None,
                     lean_project: Path | None = None,
                     **_ignored: Any) -> tuple[int, str]:
    entry = resolve_model(backend, model, effort)
    raw_dir, stem = _session_paths(session_path)
    _pin_repl_env()

    with tempfile.TemporaryDirectory(prefix=f"mlpbench_fl_{entry.backend}_") as tmp:
        ws = Path(tmp)
        for name, content in (seed_files or {}).items():
            (ws / name).write_text(content, encoding="utf-8")
        run = B.run_agent(entry, system, prompt, workspace=ws, timeout=timeout,
                          raw_dir=raw_dir, stem=stem, lean_project=lean_project,
                          extra_ro=_repl_paths())
    _AUDIT.reasons = getattr(_AUDIT, "reasons", []) + run.egress_suspect_reasons
    return (run.returncode if run.returncode is not None else -1), run.final_text
