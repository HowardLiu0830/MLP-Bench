from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import BenchmarkInvalid
from ..core.jsonio import load_jsonl, subsample
from ..core.paths import forward_construction_jsonl

__all__ = ["Target", "Row", "load", "gold", "validate", "stats",
           "N_ROWS", "N_TARGETS", "N_DECLS", "KNOWN_AUXILIARY",
           "RESEARCH", "GRADUATE", "SOURCES", "SOURCE_CHOICES", "canonical_source"]

N_ROWS = 80
N_TARGETS = 120
N_DECLS = 121

GOLD_FIELDS = ("solution",)

# Non-target `sorry`s a row may carry. Empty: every non-target declaration ships with its proof.
KNOWN_AUXILIARY: dict[str, tuple[str, ...]] = {}

_DECL = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance)\s+([A-Za-z0-9_'.]+)",
    re.M,
)
_SORRY = re.compile(r"\bsorry\b")


@dataclass(frozen=True)
class Target:
    """One scored target. `nl` is the rest of its `targets[]` entry: `statement` and, on a
    multi-target row, the ids of its context and of the sibling targets it uses (see autoform)."""
    label: str
    decls: tuple[str, ...]
    row_id: str
    nl: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def target_id(self) -> str:
        return f"{self.row_id}::{self.label}"

    @property
    def statement(self) -> str:
        return str(self.nl.get("statement") or "")

    def as_json(self) -> dict[str, Any]:
        return {"label": self.label, "decls": list(self.decls), **self.nl}


@dataclass(frozen=True)
class Row:
    id: str
    source: str
    targets: tuple[Target, ...]
    nl_proof: str
    context: tuple[Any, ...] | None
    lean: str
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(d for t in self.targets for d in t.decls)

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def n_auxiliary_sorries(self) -> int:
        return max(0, _count_sorries(self.lean) - len(self.target_names))

    @property
    def is_multi_target(self) -> bool:
        return len(self.targets) > 1


def _count_sorries(src: str) -> int:
    return len(_SORRY.findall(src))


def _decls(src: str) -> set[str]:
    return set(_DECL.findall(src))


RESEARCH, GRADUATE = "Research-level", "Graduate-level"
SOURCES = (RESEARCH, GRADUATE)
# Older files and runs say `jmlr` / `lecture` (or carry a `level` field); the CLI keeps
# accepting them, so every comparison goes through `canonical_source`.
SOURCE_ALIASES = {
    "research-level": RESEARCH, "research": RESEARCH, "jmlr": RESEARCH,
    "graduate-level": GRADUATE, "graduate": GRADUATE, "lecture": GRADUATE,
}
SOURCE_CHOICES = (*SOURCES, "jmlr", "lecture")


def canonical_source(name: str | None) -> str | None:
    """`Research-level` / `Graduate-level` for any accepted spelling; None if unknown."""
    return SOURCE_ALIASES.get(str(name).lower()) if name is not None else None


def _to_row(raw: dict[str, Any]) -> Row:
    rid = str(raw["id"])
    targets = tuple(
        Target(label=str(t["label"]), decls=tuple(str(d) for d in t["decls"]), row_id=rid,
               nl={k: v for k, v in t.items() if k not in ("label", "decls")})
        for t in raw["targets"]
    )

    source = canonical_source(raw.get("source") or raw.get("level"))
    if source is None:
        raise BenchmarkInvalid(
            f"{rid}: row has neither `source` nor a known `level` (got "
            f"{raw.get('source') or raw.get('level')!r}); expected one of {list(SOURCES)}"
        )

    known = {"id", "source", "level", "targets", "nl_proof", "context", "lean", *GOLD_FIELDS}
    return Row(
        id=rid,
        source=str(source),
        targets=targets,
        nl_proof=raw.get("nl_proof") or "",
        context=tuple(raw["context"]) if raw.get("context") is not None else None,
        lean=raw["lean"],
        extra={k: v for k, v in raw.items() if k not in known},
    )


def validate(rows: Sequence[Row], *, strict_totals: bool = True) -> list[str]:
    problems: list[str] = []
    warnings: list[str] = []

    for row in rows:
        decls = _decls(row.lean)
        flat = [d for t in row.targets for d in t.decls]

        missing = [n for n in row.target_names if n not in decls]
        if missing:
            problems.append(f"{row.id}: target_names not declared in `lean`: {missing}")

        if len(set(flat)) != len(flat):
            problems.append(f"{row.id}: a declaration is listed under two targets: {flat}")

        unstated = [t.label for t in row.targets if not t.statement.strip()]
        if unstated:
            problems.append(f"{row.id}: targets without a `statement`: {unstated}")

        n_sorry = _count_sorries(row.lean)
        if n_sorry != len(row.target_names):
            aux = KNOWN_AUXILIARY.get(row.id, ())
            extra = n_sorry - len(row.target_names)
            if aux and extra == len(aux):
                warnings.append(
                    f"{row.id}: {extra} sorry beyond the targets -- the known auxiliary "
                    f"{', '.join(aux)}; whole-file 'no sorry' would fail this row unfairly"
                )
            else:
                problems.append(
                    f"{row.id}: {n_sorry} sorry for {len(row.target_names)} target decls, "
                    "and no known auxiliary accounts for it"
                )


    if strict_totals:
        n_targets = sum(len(r.targets) for r in rows)
        n_decls = sum(len(r.target_names) for r in rows)
        if len(rows) != N_ROWS or n_targets != N_TARGETS or n_decls != N_DECLS:
            problems.append(
                f"totals changed: {len(rows)} rows / {n_targets} targets / {n_decls} decls, "
                f"expected {N_ROWS} / {N_TARGETS} / {N_DECLS}"
            )

    if problems:
        raise BenchmarkInvalid(
            "the benchmark failed its load-time invariants:\n  " + "\n  ".join(problems)
        )
    return warnings


def load(path: str | Path | None = None, *, source: str | None = None,
         ids: Iterable[str] | None = None, limit: int | None = None,
         seed: int | None = None, validate_totals: bool | None = None) -> list[Row]:
    p = Path(path) if path else forward_construction_jsonl()
    raw = load_jsonl(p)
    rows = [_to_row(r) for r in raw]

    unfiltered = source is None and ids is None and not limit
    validate(rows, strict_totals=unfiltered if validate_totals is None else validate_totals)

    if source and source != "all":
        want = canonical_source(source)
        if want is None:
            raise BenchmarkInvalid(f"unknown source {source!r}; expected one of {list(SOURCES)}")
        rows = [r for r in rows if r.source == want]
    if ids is not None:
        want = {str(i) for i in ids}
        unknown = want - {r.id for r in rows}
        if unknown:
            raise BenchmarkInvalid(f"no such row id: {sorted(unknown)}")
        rows = [r for r in rows if r.id in want]

    if limit:
        picked = subsample([{"_r": r, "source": r.source} for r in rows],
                           limit=limit, seed=seed, stratify_on="source")
        rows = [d["_r"] for d in picked]
    return rows


def gold(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path) if path else forward_construction_jsonl()
    return {str(r["id"]): {k: r.get(k) for k in GOLD_FIELDS} for r in load_jsonl(p)}


def stats(rows: Sequence[Row] | None = None) -> dict[str, Any]:
    rows = rows if rows is not None else load()
    by_source: dict[str, dict[str, int]] = {}
    for r in rows:
        b = by_source.setdefault(r.source, {"rows": 0, "targets": 0, "decls": 0})
        b["rows"] += 1
        b["targets"] += len(r.targets)
        b["decls"] += len(r.target_names)
    return {
        "rows": len(rows),
        "targets": sum(len(r.targets) for r in rows),
        "decls": sum(len(r.target_names) for r in rows),
        "by_source": by_source,
        "multi_decl_targets": [t.target_id for r in rows for t in r.targets if len(t.decls) > 1],
        "rows_with_auxiliary_sorries": [r.id for r in rows if r.n_auxiliary_sorries],
    }


def render_deps(deps: Sequence[Any] | None) -> str:
    """`context` entries as readable text: one `### name` heading and the verbatim statement per entry.

    The entries' `reason` / `strength` fields are curation metadata and are not shown.
    """
    blocks: list[str] = []
    for d in deps or ():
        if isinstance(d, dict):
            name = str(d.get("name") or d.get("id") or "").strip()
            body = str(d.get("statement") or "").strip()
        else:
            name, body = "", str(d).strip()
        if body:
            blocks.append(f"### {name}\n{body}" if name else body)
    return "\n\n".join(blocks)
