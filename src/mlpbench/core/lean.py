from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..errors import LeanUnavailable
from .cpu import cpu_list, lean_threads
from .jsonio import trim_log
from .leantext import build_axiom_check_file
from .paths import lean_project as default_lean_project
from .paths import workspace_root

__all__ = ["LeanResult", "DEFAULT_LEAN_CMD", "STANDARD_AXIOMS",
           "compile_file", "print_axioms", "parse_axioms", "preflight",
           "agent_lean_command"]

DEFAULT_LEAN_CMD = "lake env lean"

STANDARD_AXIOMS = frozenset({
    "Classical.choice", "Quot.sound", "propext",
    "lcProof", "lcCast", "lcUnreachable",
})

_SORRY_DIAGNOSTICS = ("declaration uses 'sorry'", "declaration uses `sorry`")


@dataclass
class LeanResult:
    is_success: bool
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None
    lean_file: str | None = None

    @property
    def combined(self) -> str:
        return (self.stdout or "") + ("\n" if self.stdout and self.stderr else "") + (self.stderr or "")

    def as_dict(self) -> dict[str, Any]:
        return {"is_success": self.is_success, "returncode": self.returncode,
                "stdout": self.stdout, "stderr": self.stderr,
                "timed_out": self.timed_out, "error": self.error}


def _argv(lean_cmd: str, lean_file: Path, *, json: bool = False, threads: bool = False) -> list[str]:
    cmd = shlex.split(lean_cmd)
    if json:
        cmd = cmd + ["--json"]
    if threads:
        cmd = cmd + ["-j", str(lean_threads())]
    cmd = cmd + [str(lean_file)]
    if shutil.which("taskset"):
        cmd = ["taskset", "-c", cpu_list()] + cmd
    return cmd


def _run(source: str, *, lean_project: Path, lean_cmd: str, timeout: int,
         scratch_dir: Path, stem: str, json: bool, threads: bool) -> LeanResult:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("LEAN_NUM_THREADS", str(lean_threads()))
    with tempfile.TemporaryDirectory(prefix=f"{stem}.", dir=str(scratch_dir)) as tmp:
        lean_file = Path(tmp) / "check.lean"
        try:
            lean_file.write_text(source if source.endswith("\n") else source + "\n",
                                 encoding="utf-8")
        except OSError as exc:
            return LeanResult(is_success=False, error=f"failed to write the Lean file: {exc}")
        try:
            proc = subprocess.run(_argv(lean_cmd, lean_file, json=json, threads=threads),
                                  cwd=str(lean_project), capture_output=True, text=True,
                                  timeout=timeout, env=env)
        except subprocess.TimeoutExpired as exc:
            return LeanResult(is_success=False, returncode=None,
                              stdout=trim_log(exc.stdout or ""),
                              stderr=trim_log(exc.stderr or "timeout"),
                              timed_out=True)
        except OSError as exc:
            return LeanResult(is_success=False, error=f"could not run Lean: {exc}")

        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        success = proc.returncode == 0 and not any(d in out for d in _SORRY_DIAGNOSTICS)
        return LeanResult(is_success=success, returncode=proc.returncode,
                          stdout=trim_log(proc.stdout or ""), stderr=trim_log(proc.stderr or ""),
                          lean_file=str(lean_file))


def compile_file(source: str, *, lean_project: Path | None = None,
                 lean_cmd: str = DEFAULT_LEAN_CMD, timeout: int = 300,
                 scratch_dir: Path | None = None, stem: str = "check",
                 json: bool = False) -> LeanResult:
    return _run(source, lean_project=Path(lean_project or default_lean_project()),
                lean_cmd=lean_cmd, timeout=timeout,
                scratch_dir=Path(scratch_dir or (workspace_root() / "lean")),
                stem=stem, json=json, threads=True)


def print_axioms(source: str, names: Sequence[str], *, lean_project: Path | None = None,
                 lean_cmd: str = DEFAULT_LEAN_CMD, timeout: int = 600,
                 scratch_dir: Path | None = None, stem: str = "axioms") -> LeanResult:
    return _run(build_axiom_check_file(source, list(names)),
                lean_project=Path(lean_project or default_lean_project()),
                lean_cmd=lean_cmd, timeout=timeout,
                scratch_dir=Path(scratch_dir or (workspace_root() / "lean")),
                stem=stem, json=False, threads=True)


def parse_axioms(out: str, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    res: dict[str, dict[str, Any]] = {}
    for name in names:
        m = re.search(
            rf"'([\w.]*\b{re.escape(name)})' "
            rf"(?:does not depend on any axioms|depends on axioms:\s*(\[.*?\]))",
            out, re.DOTALL,
        )
        if not m:
            unknown = re.search(rf"[Uu]nknown (?:constant|identifier) `?{re.escape(name)}`?", out)
            res[name] = {
                "solved": False,
                "reason": ("target did not elaborate (unknown constant)" if unknown
                           else "no axiom trace found (elaboration or parse failure)"),
            }
            continue

        axlist = m.group(2)
        if axlist is None:
            res[name] = {"solved": True, "reason": "no axioms (fully constructive)"}
            continue
        if "sorryAx" in axlist:
            res[name] = {"solved": False,
                         "reason": "proof term uses sorryAx (the target or a helper is sorry)"}
            continue

        axioms = sorted(a for a in set(re.findall(r"[\w.]+", axlist)) if a)
        injected = [a for a in axioms if a not in STANDARD_AXIOMS]
        if injected:
            res[name] = {
                "solved": False, "kernel_unsound": True,
                "reason": f"REJECTED: non-standard axiom(s) {injected} "
                          "(kernel bypass or injected axiom)",
            }
        else:
            res[name] = {"solved": True, "reason": f"depends only on standard axioms {axioms}"}
    return res


_PROBE = "import Mathlib\ntheorem mlpbench_preflight : 2 + 2 = 4 := by norm_num\n"


def preflight(*, lean_project: Path | None = None, lean_cmd: str = DEFAULT_LEAN_CMD,
              timeout: int = 900, scratch_dir: Path | None = None) -> None:
    project = Path(lean_project or default_lean_project())
    if not project.is_dir():
        raise LeanUnavailable(
            f"the Lean project {project} does not exist.\n"
            "Build it once with `code/lean/bootstrap_lean.sh`, or point "
            "MLPBENCH_LEAN_PROJECT at an existing lake project with Mathlib."
        )
    res = compile_file(_PROBE, lean_project=project, lean_cmd=lean_cmd, timeout=timeout,
                       scratch_dir=scratch_dir, stem="preflight")
    if not res.is_success:
        raise LeanUnavailable(
            f"Lean could not compile a two-line Mathlib probe in {project}.\n"
            f"rc={res.returncode} timed_out={res.timed_out}\n"
            f"{trim_log(res.combined, 1200)}\n"
            "Without a working Mathlib every proof fails to elaborate and the run scores "
            "zero for a reason that has nothing to do with the models."
        )


def agent_lean_command(lean_file: str, *, lean_project: Path | None = None) -> str:
    project = Path(lean_project or default_lean_project())
    return f"cd {project} && {DEFAULT_LEAN_CMD} {lean_file}"
