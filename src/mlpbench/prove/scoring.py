from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..benchmarks.forward_construction import Row
from ..core import lean as L
from ..core.leantext import altered_definition, statement_preserved

__all__ = ["DeclVerdict", "TargetVerdict", "RowScore", "score_file", "score_from_output"]

_DIAGNOSTIC = re.compile(r"(?m)^(?=\S+\.lean:\d+:\d+: (?:error|warning|information):)")

_AXIOM_BANNER = re.compile(r"(?m)^'[\w.]+' (?:depends on axioms|does not depend).*$")


@dataclass
class DeclVerdict:
    name: str
    solved: bool
    reason: str
    kernel_unsound: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "solved": self.solved, "reason": self.reason,
                "kernel_unsound": self.kernel_unsound}


@dataclass
class TargetVerdict:
    target_id: str
    label: str
    decls: tuple[str, ...]
    solved: bool
    reason: str
    kernel_unsound: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"target_id": self.target_id, "label": self.label, "decls": list(self.decls),
                "per_target_solved": self.solved, "per_target_reason": self.reason,
                "kernel_unsound": self.kernel_unsound}


@dataclass
class RowScore:
    row_id: str
    source: str
    decls: list[DeclVerdict] = field(default_factory=list)
    targets: list[TargetVerdict] = field(default_factory=list)
    whole_file_ok: bool = False
    checked: bool = False
    error: str | None = None
    lean: dict[str, Any] = field(default_factory=dict)

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def n_solved(self) -> int:
        return sum(1 for t in self.targets if t.solved)

    @property
    def kernel_unsound(self) -> bool:
        return any(t.kernel_unsound for t in self.targets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.row_id, "source": self.source,
            "checked": self.checked, "error": self.error,
            "success": self.whole_file_ok,
            "n_targets": self.n_targets, "n_targets_solved": self.n_solved,
            "kernel_unsound": self.kernel_unsound,
            "decls": [d.as_dict() for d in self.decls],
            "targets": [t.as_dict() for t in self.targets],
            "lean": self.lean,
        }


def _error_names_decl(out: str, name: str) -> bool:
    diagnostics_only = _AXIOM_BANNER.sub("", out or "")
    for chunk in _DIAGNOSTIC.split(diagnostics_only):
        if "error:" in chunk and re.search(rf"\b{re.escape(name)}\b", chunk):
            return True
    return False


def score_from_output(row: Row, completed: str, axiom_output: str) -> RowScore:
    score = RowScore(row_id=row.id, source=row.source, checked=True)
    names = list(row.target_names)

    bad_def = altered_definition(row.lean, completed)
    if bad_def:
        reason = f"REJECTED: the original {bad_def} was altered or removed"
        score.decls = [DeclVerdict(n, False, reason) for n in names]
        score.targets = [TargetVerdict(t.target_id, t.label, t.decls, False, reason)
                         for t in row.targets]
        return score

    raw = L.parse_axioms(axiom_output, names)
    verdicts: dict[str, DeclVerdict] = {}
    for n in names:
        r = raw.get(n, {"solved": False, "reason": "no verdict"})
        v = DeclVerdict(n, bool(r.get("solved")), str(r.get("reason", "")),
                        bool(r.get("kernel_unsound")))
        if v.solved and not statement_preserved(row.lean, completed, n):
            v = DeclVerdict(n, False, "REJECTED: the target statement was altered")
        elif v.solved and _error_names_decl(axiom_output, n):
            v = DeclVerdict(
                n, False,
                "REJECTED: the axiom trace looked clean but an `error:` in the compile "
                "output names this declaration (a hard elaboration failure that was still "
                "committed to the environment)",
            )
        verdicts[n] = v
    score.decls = [verdicts[n] for n in names]

    for t in row.targets:
        parts = [verdicts[d] for d in t.decls]
        solved = all(p.solved for p in parts)
        if len(parts) == 1:
            reason = parts[0].reason
        else:
            reason = "; ".join(f"{p.name}: {p.reason}" for p in parts)
        score.targets.append(TargetVerdict(t.target_id, t.label, t.decls, solved, reason,
                                           any(p.kernel_unsound for p in parts)))
    return score


def score_file(row: Row, completed: str, *, lean_project: Path | None = None,
               lean_cmd: str = L.DEFAULT_LEAN_CMD, timeout: int = 900,
               scratch_dir: Path | None = None,
               whole_file: bool = True) -> RowScore:
    axioms = L.print_axioms(completed, list(row.target_names), lean_project=lean_project,
                            lean_cmd=lean_cmd, timeout=timeout, scratch_dir=scratch_dir,
                            stem=f"ax.{row.id}")
    score = score_from_output(row, completed, axioms.combined)
    score.lean = {"axioms": axioms.as_dict()}

    if whole_file:
        res = L.compile_file(completed, lean_project=lean_project, lean_cmd=lean_cmd,
                             timeout=timeout, scratch_dir=scratch_dir, stem=f"wf.{row.id}")
        score.whole_file_ok = bool(res.is_success) and not score.kernel_unsound
        score.lean["whole_file"] = res.as_dict()
    return score


def totals(scores: Sequence[RowScore]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_rows": len(scores),
        "n_targets": sum(s.n_targets for s in scores),
        "n_decls": sum(len(s.decls) for s in scores),
        "targets_solved": sum(s.n_solved for s in scores),
        "rows_whole_file": sum(1 for s in scores if s.whole_file_ok),
        "rows_kernel_unsound": sum(1 for s in scores if s.kernel_unsound),
        "by_source": {},
    }
    for s in scores:
        b = out["by_source"].setdefault(
            s.source, {"n_rows": 0, "n_targets": 0, "n_decls": 0,
                       "targets_solved": 0, "rows_whole_file": 0})
        b["n_rows"] += 1
        b["n_targets"] += s.n_targets
        b["n_decls"] += len(s.decls)
        b["targets_solved"] += s.n_solved
        b["rows_whole_file"] += int(s.whole_file_ok)
    n = out["n_targets"]
    out["target_solve_rate"] = out["targets_solved"] / n if n else 0.0
    for b in out["by_source"].values():
        b["target_solve_rate"] = b["targets_solved"] / b["n_targets"] if b["n_targets"] else 0.0
    return out
