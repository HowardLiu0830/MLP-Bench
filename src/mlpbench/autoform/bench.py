from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..benchmarks.forward_construction import Row, load as load_rows

__all__ = ["as_record", "load", "decls"]

_DROP = ("solution",)


def as_record(row: Row) -> dict[str, Any]:
    return {
        "paper_id": row.id,
        "source": row.source,
        "target_names": list(row.target_names),
        "targets": [t.as_json() for t in row.targets],
        "lean": row.lean,
        "nl_proof": row.nl_proof,
        "context": list(row.context) if row.context else None,
    }


def load(path: str | Path | None = None, *, source: str | None = None,
         ids: Iterable[str] | None = None, limit: int | None = None,
         seed: int | None = None) -> list[dict[str, Any]]:
    rows = load_rows(path, source=source, ids=ids, limit=limit, seed=seed)
    records = [as_record(r) for r in rows]
    for rec in records:
        assert not any(k in rec for k in _DROP), "the gold must not reach this pipeline"
    return records


def decls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        by_decl = {d: t for t in rec["targets"] for d in t["decls"]}
        for name in rec["target_names"]:
            t = by_decl[name]
            out.append({
                "paper_id": rec["paper_id"],
                "source": rec["source"],
                "label": t["label"],
                "target_name": name,
                "target_label_decls": list(t["decls"]),
                "target_names": list(rec["target_names"]),
                "targets": rec["targets"],
                "lean": rec["lean"],
                "nl_proof": rec["nl_proof"],
                "context": rec["context"],
            })
    return out


