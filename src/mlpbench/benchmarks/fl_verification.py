from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..errors import BenchmarkInvalid, GoldLeak
from ..core.jsonio import load_jsonl, subsample
from ..core.paths import backward_verification_jsonl

__all__ = ["Step", "Proof", "Theorem", "Item", "load", "gold", "stats",
           "render", "check_rendered", "N_ITEMS", "GOLD_FIELDS", "LEAK_TOKENS"]

N_ITEMS = 120

GOLD_FIELDS = ("label", "error_step", "error_class6", "error_explanation")

LEAK_TOKENS: tuple[str, ...] = (
    "is_target", "errata", "as_printed", "gold_verdict", "has_injected_error",
    "all_proofs_correct", "error_step", "error_class", "same_book_filler",
    "__correct", "__perturbation",
)


@dataclass(frozen=True)
class Step:
    n: int
    claim: str


@dataclass(frozen=True)
class Proof:
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Theorem:
    local_id: str
    statement: str
    proof: Proof


@dataclass(frozen=True)
class Item:
    id: str
    context: str
    theorems: tuple[Theorem, ...]
    target_local_id: str | None


def _to_theorem(raw: dict[str, Any]) -> Theorem:
    p = raw.get("proof") or {}
    steps = tuple(Step(n=int(s.get("n") or i + 1), claim=str(s.get("claim") or ""))
                  for i, s in enumerate(p.get("steps") or ()))
    return Theorem(
        local_id=str(raw.get("local_id") or ""),
        statement=str(raw.get("statement") or ""),
        proof=Proof(steps=steps),
    )


def _to_item(raw: dict[str, Any]) -> Item:
    return Item(
        id=str(raw["id"]),
        context=str(raw.get("context") or ""),
        theorems=tuple(_to_theorem(t) for t in raw.get("theorems") or ()),
        target_local_id=(str(raw["target_local_id"])
                         if raw.get("target_local_id") is not None else None),
    )


def load(path: str | Path | None = None, *, ids: Iterable[str] | None = None,
         limit: int | None = None, seed: int | None = None) -> list[Item]:
    p = Path(path) if path else backward_verification_jsonl()
    raw = load_jsonl(p)
    if ids is None and not limit and len(raw) != N_ITEMS:
        raise BenchmarkInvalid(f"expected {N_ITEMS} items, found {len(raw)} in {p}")

    items = [_to_item(r) for r in raw]

    if ids is not None:
        want = {str(i) for i in ids}
        unknown = want - {i.id for i in items}
        if unknown:
            raise BenchmarkInvalid(f"no such item id: {sorted(unknown)}")
        return [i for i in items if i.id in want]

    if limit:
        labels = {str(r["id"]): r.get("label") for r in raw}
        picked = subsample([{"_i": i, "label": labels.get(i.id)} for i in items],
                           limit=limit, seed=seed, stratify_on="label")
        return [d["_i"] for d in picked]
    return items


def gold(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path) if path else backward_verification_jsonl()
    out: dict[str, dict[str, Any]] = {}
    for r in load_jsonl(p):
        g = {k: r.get(k) for k in GOLD_FIELDS if k in r}
        g["target_local_id"] = r.get("target_local_id")
        out[str(r["id"])] = g
    return out


def render(item: Item, *, include_target: bool = False) -> str:
    out: list[str] = []
    if item.context.strip():
        out.append("## Context\n\n" + item.context.strip())
    for th in item.theorems:
        head = f"## {th.local_id}"
        if include_target and item.target_local_id == th.local_id:
            head += "  [TARGET]"
        out.append(head)
        out.append(th.statement.strip())
        out.append("**Proof.**")
        out += [f"{s.n}. {s.claim.strip()}" for s in th.proof.steps]
    return "\n\n".join(out).strip() + "\n"


def check_rendered(item_id: str, rendered: str) -> None:
    found = [tok for tok in LEAK_TOKENS if tok in rendered]
    if item_id and item_id in rendered:
        found.append(f"the item id ({item_id})")
    if found:
        raise GoldLeak(
            f"{item_id}: the rendered bundle carries {found}. "
            "Only context, theorem statements and proof steps may reach the model."
        )


def stats(path: str | Path | None = None) -> dict[str, Any]:
    items = load(path)
    g = gold(path)
    labels: dict[str, int] = {}
    for i in items:
        labels[str(g[i.id].get("label"))] = labels.get(str(g[i.id].get("label")), 0) + 1
    return {
        "items": len(items),
        "labels": labels,
        "theorems_per_item": sorted({len(i.theorems) for i in items}),
        "items_without_a_named_target": sum(1 for i in items if i.target_local_id is None),
        "label_reachable_from_item": any(hasattr(i, f) for i in items[:1] for f in GOLD_FIELDS),
    }
