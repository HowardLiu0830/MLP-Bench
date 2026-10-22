from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..benchmarks.forward_construction import GRADUATE, RESEARCH, Row, render_deps
from ..core import lean as L
from ..core import leantext as LT

__all__ = ["SourceProfile", "SOURCE_PROFILES", "Problem", "build_problem", "NL_MODES",
           "statements_nl"]

# none: the Lean file alone. statement (default): the targets' informal statements, in
# NL_STATEMENT.md beside the Lean file, which the prompt points to. row: statements and the informal proof, in
# NL_PROOF.md. The CoT prompt has no workspace and keeps the statements inline (task.cot_user).
NL_MODES = ("none", "statement", "row")


@dataclass(frozen=True)
class SourceProfile:
    item_noun: str


SOURCE_PROFILES: dict[str, SourceProfile] = {
    RESEARCH: SourceProfile("Paper"),
    GRADUATE: SourceProfile("Theorem"),
}


@dataclass
class Problem:
    row: Row
    profile: SourceProfile
    workfile: Path
    lean_command: str
    lean_project: Path
    nl_mode: str
    nl_source: str | None
    nl_text: str
    nl_filename: str | None

    @property
    def id(self) -> str:
        return self.row.id

    @property
    def multi_target(self) -> bool:
        return self.row.is_multi_target

    def target_list(self) -> str:
        lines = []
        for t in self.row.targets:
            if len(t.decls) > 1:
                lines.append(f"- {t.label}: {' AND '.join(t.decls)} "
                             "(one target; BOTH declarations must be proved)")
            else:
                lines.append(f"- {t.label}: {t.decls[0]}")
        return "\n".join(lines)

    def nl_block(self) -> str:
        if self.nl_mode == "none" or not self.nl_filename:
            return "\n"
        what = ("the informal statements of the targets" if self.nl_mode == "statement"
                else "the informal statement and proof of the target")
        return (f"\n[Natural-language material -- {what}]\n"
                f"{self.nl_filename} (in your working directory)\n")

    def meta(self) -> dict[str, Any]:
        return {"id": self.row.id, "source": self.row.source,
                "n_targets": self.row.n_targets,
                "n_target_decls": len(self.row.target_names),
                "nl_mode": self.nl_mode, "nl_source": self.nl_source,
                "n_auxiliary_sorries": self.row.n_auxiliary_sorries}


def _statement_sections(row: Row) -> list[str]:
    """One section per target: headed by its label, or `## Statement` when the row has one
    target (a Graduate-level label is the declaration's name, which the heading would leak)."""
    parts: list[str] = []
    for t in row.targets:
        head = f"## {t.label}".rstrip() if row.is_multi_target else "## Statement"
        text = t.statement.strip()
        if row.is_multi_target or text:
            parts += [head, "", text, ""]
    return parts


def statements_nl(row: Row) -> str:
    """The targets' informal statements, one section per target label."""
    return "\n".join(_statement_sections(row)).rstrip() + "\n"


def _row_nl(row: Row, *, include_deps: bool) -> str:
    parts: list[str] = [f"# {row.id}", ""] + _statement_sections(row)
    if row.nl_proof.strip():
        parts += ["## Proof", "", row.nl_proof.strip(), ""]
    if include_deps and row.context:
        parts += ["## Cited dependencies", ""]
        parts.append(render_deps(row.context))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def build_problem(row: Row, *, workspace: Path, nl: str = "statement",
                  include_deps: bool = False,
                  lean_project: Path | None = None) -> Problem:
    try:
        profile = SOURCE_PROFILES[row.source]
    except KeyError:
        raise KeyError(
            f"{row.id}: unknown source {row.source!r}. A new source needs a decision about "
            f"where its natural-language material comes from; known: {sorted(SOURCE_PROFILES)}"
        ) from None

    if nl not in NL_MODES:
        raise ValueError(f"--nl must be one of {NL_MODES}, not {nl!r}")

    workspace = Path(workspace)
    workspace = workspace if workspace.is_absolute() else (Path.cwd() / workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    workfile = workspace / f"{row.id}.lean"
    src = row.lean if row.lean.endswith("\n") else row.lean + "\n"
    src = LT.mark_targets(src, {d: t.label for t in row.targets for d in t.decls})
    workfile.write_text(src, encoding="utf-8")

    nl_mode, nl_source, nl_text, nl_filename = "none", None, "", None
    if nl == "statement":
        nl_mode, nl_source = "statement", "benchmark row"
        nl_text = statements_nl(row)
        nl_filename = "NL_STATEMENT.md"
        (workspace / nl_filename).write_text(nl_text, encoding="utf-8")
    elif nl == "row":
        nl_mode, nl_source = "row", "benchmark row"
        nl_text = _row_nl(row, include_deps=include_deps)
        nl_filename = "NL_PROOF.md"
        (workspace / nl_filename).write_text(nl_text, encoding="utf-8")

    return Problem(
        row=row, profile=profile, workfile=workfile,
        lean_command=L.agent_lean_command(str(workfile), lean_project=lean_project),
        lean_project=Path(lean_project) if lean_project else L.default_lean_project(),
        nl_mode=nl_mode, nl_source=nl_source, nl_text=nl_text, nl_filename=nl_filename,
    )
