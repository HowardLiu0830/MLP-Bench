from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import data_root, lean_project, workspace_root
from .bwrap import (
    AGENT_STATE_HIDE_DIRS,
    AGENT_STATE_HIDE_FILES,
    CLAUDE_STATE_RW,
    CODEX_STATE_RO,
    CODEX_STATE_RW,
    LEAN_PROJECT_EXPOSE,
    Egress,
    OFFLINE,
    agent_jail,
    release_jail,
    bwrap_bin,
    require_bwrap,
)
from .egress import denied as egress_denied, door_reasons, egress_for, suspect_reasons
from .tools import (
    AGENTIC_BLOCKED,
    claude_agentic_args,
)

__all__ = [
    "agent_jail", "release_jail", "require_bwrap", "bwrap_bin", "Egress", "OFFLINE", "egress_for", "egress_denied",
    "suspect_reasons", "door_reasons",
    "claude_agentic_args",
    "AGENTIC_BLOCKED", "LEAN_PROJECT_EXPOSE", "AGENT_STATE_HIDE_DIRS", "AGENT_STATE_HIDE_FILES",
    "CODEX_STATE_RW", "CODEX_STATE_RO", "CLAUDE_STATE_RW",
    "SelfTest", "selftest",
]


@dataclass
class SelfTest:
    ok: bool
    sandboxed: bool
    hidden: list[str] = field(default_factory=list)
    visible: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    network: list[str] = field(default_factory=list)    # destinations reached from inside

    def as_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "sandboxed": self.sandboxed, "hidden": self.hidden,
                "visible": self.visible, "network": self.network, "errors": self.errors}


_PROBE = r"""
import json, os, sys
import socket
out = {"visible": [], "hidden": [], "network": []}
for p in sys.argv[1:]:
    (out["visible"] if os.path.exists(p) else out["hidden"]).append(p)
for host in ("1.1.1.1", "github.com"):
    try:
        socket.create_connection((host, 443), timeout=3).close()
        out["network"].append(host + ":443")
    except OSError:
        pass
print("MLPBENCH_PROBE " + json.dumps(out))
"""


def selftest(*, workspace: Path | None = None, must_not_see: list[Path] | None = None,
             timeout: int = 120) -> SelfTest:
    if not bwrap_bin():
        return SelfTest(ok=False, sandboxed=False,
                        errors=["bubblewrap is not available; the run is not sandboxed"])

    ws = Path(workspace) if workspace else (workspace_root() / "_selftest")
    ws.mkdir(parents=True, exist_ok=True)

    if must_not_see is None:
        try:
            root = data_root()
            must_not_see = [root / "Benchmarks" / "forward-construction.jsonl",
                            root / "Benchmarks" / "backward-verification.jsonl",
                            root]
        except Exception as exc:
            return SelfTest(ok=False, sandboxed=True, errors=[f"cannot locate the data: {exc}"])

    targets = [str(p) for p in must_not_see]
    interp = Path(getattr(sys, "_base_executable", None) or sys.executable).resolve()
    interp_ro = [Path(sys.base_prefix).resolve(), interp.parent]
    cmd = agent_jail([str(interp), "-c", _PROBE, *targets],
                     workspace=ws, with_lean=True, lean_project=lean_project(),
                     extra_ro=interp_ro)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return SelfTest(ok=False, sandboxed=True, errors=[f"probe failed to run: {exc}"])
    finally:
        release_jail(cmd)

    line = next((l for l in proc.stdout.splitlines() if l.startswith("MLPBENCH_PROBE ")), None)
    if line is None:
        return SelfTest(ok=False, sandboxed=True,
                        errors=[f"probe produced no verdict (rc={proc.returncode}): "
                                f"{(proc.stderr or proc.stdout)[:300]}"])

    import json

    seen = json.loads(line[len("MLPBENCH_PROBE "):])
    visible, net = seen.get("visible", []), seen.get("network", [])
    errors = [f"the benchmark is readable inside the sandbox: {visible}"] if visible else []
    if net:   # the jail has no network namespace of its own: never run an agent in it
        errors.append(f"the sandbox reached the network: {net}")
    return SelfTest(
        ok=not visible and not net,
        sandboxed=True,
        hidden=seen.get("hidden", []),
        visible=visible,
        network=net,
        errors=errors,
    )
