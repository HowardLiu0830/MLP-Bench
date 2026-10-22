from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .leansurgery import (
    CLOSERS, OPENERS, _strip_for_scan, find_toplevel,
    split_imports,
    local_decl_names, parse_chunks, parse_decl, qualify,
)


PRED_LEAN_KEYS = ("lean", "pred_lean", "prediction", "formalization", "completed_file", "proof_body")


PRED_STMT_KEYS = ("statement", "theorem", "decl")


def prediction_lean(row: dict[str, Any]) -> str | None:
    for key in PRED_LEAN_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    for key in PRED_STMT_KEYS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return "import Mathlib\n\n" + val
    return None


@dataclass
class Pair:
    pair_id: str
    paper_id: str
    target_name: str
    source: str
    sample_idx: int
    ref_lean: str
    ref_targets: list[str]
    pred_lean: str
    pred_target: str
    pred_targets: list[str]
    note: str = ""


def build_pairs(
    gold: list[dict[str, Any]],
    preds: list[dict[str, Any]] | None,
    *,
    self_check: str | None = None,
) -> tuple[list[Pair], list[dict[str, Any]]]:
    unmatched: list[dict[str, Any]] = []
    pairs: list[Pair] = []

    if self_check == "identity":
        for g in gold:
            pairs.append(
                Pair(
                    pair_id=f"{g['paper_id']}::{g['target_name']}::identity",
                    paper_id=str(g["paper_id"]), target_name=g["target_name"],
                    source=str(g.get("source") or ""), sample_idx=0,
                    ref_lean=g["lean"], ref_targets=g["target_names"],
                    pred_lean=g["lean"], pred_target=g["target_name"],
                    pred_targets=g["target_names"], note="self-check: identity",
                )
            )
        return pairs, unmatched

    if self_check == "rename":
        for g in gold:
            names = local_decl_names(split_imports(g["lean"])[1]) - {g["target_name"]}
            renamed = qualify(g["lean"], names, "") if not names else g["lean"]
            for n in sorted(names, key=len, reverse=True):
                renamed = re.sub(r"(?<![\w.'])" + re.escape(n) + r"(?![\w.'])", n + "R", renamed)
            pairs.append(
                Pair(
                    pair_id=f"{g['paper_id']}::{g['target_name']}::rename",
                    paper_id=str(g["paper_id"]), target_name=g["target_name"],
                    source=str(g.get("source") or ""), sample_idx=0,
                    ref_lean=g["lean"], ref_targets=g["target_names"],
                    pred_lean=renamed, pred_target=g["target_name"],
                    pred_targets=g["target_names"],
                    note=f"self-check: rename ({len(names)} local names)",
                )
            )
        return pairs, unmatched

    if self_check == "drop_hyp":
        for g in gold:
            weaker = drop_last_hypothesis(g["lean"], g["target_name"])
            if weaker is None:
                unmatched.append({"paper_id": g["paper_id"], "target_name": g["target_name"],
                                  "reason": "no droppable hypothesis"})
                continue
            pairs.append(
                Pair(
                    pair_id=f"{g['paper_id']}::{g['target_name']}::drop_hyp",
                    paper_id=str(g["paper_id"]), target_name=g["target_name"],
                    source=str(g.get("source") or ""), sample_idx=0,
                    ref_lean=g["lean"], ref_targets=g["target_names"],
                    pred_lean=weaker, pred_target=g["target_name"],
                    pred_targets=g["target_names"],
                    note="self-check: gold minus one hypothesis (must be rejected)",
                )
            )
        return pairs, unmatched

    if self_check == "shuffle":
        for i, g in enumerate(gold):
            other = gold[(i + 1) % len(gold)]
            if other["paper_id"] == g["paper_id"] and len(gold) > 2:
                other = gold[(i + 2) % len(gold)]
            pairs.append(
                Pair(
                    pair_id=f"{g['paper_id']}::{g['target_name']}::shuffle",
                    paper_id=str(g["paper_id"]), target_name=g["target_name"],
                    source=str(g.get("source") or ""), sample_idx=0,
                    ref_lean=g["lean"], ref_targets=g["target_names"],
                    pred_lean=other["lean"], pred_target=other["target_name"],
                    pred_targets=other["target_names"],
                    note=f"self-check: shuffle <- {other['paper_id']}::{other['target_name']}",
                )
            )
        return pairs, unmatched

    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in preds or []:
        key = (str(row.get("paper_id")), str(row.get("target_name") or row.get("target") or ""))
        index.setdefault(key, []).append(row)

    for g in gold:
        key = (str(g["paper_id"]), str(g["target_name"]))
        rows = index.get(key)
        if not rows:
            unmatched.append({"paper_id": g["paper_id"], "target_name": g["target_name"],
                              "reason": "no prediction"})
            continue
        for j, row in enumerate(rows):
            lean = prediction_lean(row)
            if not lean:
                unmatched.append({"paper_id": g["paper_id"], "target_name": g["target_name"],
                                  "reason": "prediction row has no Lean source"})
                continue
            sample_idx = int(row.get("sample_idx", row.get("sample", j)) or 0)
            pred_target = str(row.get("pred_target_name") or row.get("target_name") or g["target_name"])
            pairs.append(
                Pair(
                    pair_id=f"{g['paper_id']}::{g['target_name']}::{sample_idx}",
                    paper_id=str(g["paper_id"]), target_name=g["target_name"],
                    source=str(g.get("source") or ""), sample_idx=sample_idx,
                    ref_lean=g["lean"], ref_targets=g["target_names"],
                    pred_lean=lean, pred_target=pred_target,
                    pred_targets=list(row.get("target_names") or [pred_target]),
                )
            )
    return pairs, unmatched


HYP_NAME_RE = re.compile(r"^\(\s*h[A-Za-z0-9_']*\s*:")


def drop_last_hypothesis(lean: str, target_name: str) -> str | None:
    imports, body = split_imports(lean)
    chunks = parse_chunks(body)
    hit = next((c for c in chunks if c.kind in ("theorem", "lemma") and c.name == target_name), None)
    if hit is None:
        return None
    decl = parse_decl(hit)
    if decl is None:
        return None
    binders = decl.after_name[: find_toplevel(decl.after_name, ":", skip="=")]
    scan = _strip_for_scan(binders)
    groups, depth, start = [], 0, None
    for i, ch in enumerate(scan):
        if ch in OPENERS:
            if depth == 0:
                start = i
            depth += 1
        elif ch in CLOSERS:
            depth -= 1
            if depth == 0 and start is not None:
                groups.append((start, i + 1))
                start = None
    hyps = [g for g in groups if HYP_NAME_RE.match(binders[g[0]:g[1]])]
    if not hyps:
        return None
    lo, hi = hyps[-1]
    new_binders = (binders[:lo] + binders[hi:]).rstrip()
    new_decl = hit.text.replace(binders, new_binders, 1)
    return lean.replace(hit.text, new_decl, 1)


def resolve_pred_target(pred_lean: str, wanted: str) -> tuple[str, list[str]]:
    _, body = split_imports(pred_lean)
    thms = [c.name for c in parse_chunks(body) if c.kind in ("theorem", "lemma") and c.name]
    if not thms:
        raise ValueError("predicted file declares no theorem")
    return (wanted if wanted in thms else thms[-1]), thms


def summarize(results: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    by_target: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in results:
        by_target.setdefault((r["paper_id"], r["target_name"]), []).append(r)

    out: dict[str, Any] = {
        "n_pairs": len(results),
        "n_targets": len(by_target),
        "n_errors": sum(1 for r in results if r.get("error")),
        "n_timeouts": sum(1 for r in results if r.get("timed_out")),
        "n_trivial": sum(1 for r in results if r.get("trivial_any")),
    }
    for key in keys:
        out[f"{key}@1"] = _rate([rs[0].get(key) for rs in by_target.values() if rs])
        k = max((len(rs) for rs in by_target.values()), default=1)
        if k > 1:
            out[f"{key}@{k}"] = _rate([any(bool(r.get(key)) for r in rs) for rs in by_target.values()])
        out[f"{key}_pairs"] = _rate([r.get(key) for r in results])
    return out


def _rate(values: list[Any]) -> dict[str, Any]:
    vals = [bool(v) for v in values]
    n = len(vals)
    hit = sum(vals)
    return {"hit": hit, "n": n, "rate": (hit / n) if n else 0.0}


def report(results: list[dict[str, Any]], keys: tuple[str, ...], out_dir: Path,
           extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = summarize(results, keys)
    if extra:
        summary.update(extra)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("\n=== summary ===")
    print(f"pairs={summary['n_pairs']}  targets={summary['n_targets']}  "
          f"errors={summary['n_errors']}  timeouts={summary['n_timeouts']}  "
          f"trivial_directions={summary['n_trivial']}")
    for key, val in summary.items():
        if isinstance(val, dict) and "rate" in val:
            print(f"  {key:<22} {val['hit']:>4}/{val['n']:<4} = {val['rate']:.3f}")
    print(f"\nwrote {out_dir/'pairs.jsonl'} and {out_dir/'summary.json'}")
    return summary
