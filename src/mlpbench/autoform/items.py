from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cache, cached_property
from typing import Any

from ..core.jsonio import load_jsonl
from ..core.paths import auto_formalization_jsonl
from .leansurgery import (
    Target, parse_chunks, split_imports,
)


DECLINE_TOKENS = {"", "none", "-", "no hypothesis", "nothing", "n/a", "null", "无"}


@dataclass
class Hole:
    idx: int
    start: int
    end: int
    gold: str
    kind: str = "other"
    nl_anchor: str = ""
    decoy: bool = False
    """A hole where the gold statement has nothing at all.

    Its span is zero-width -- nothing is removed -- and the template shows an
    empty binder slot ` (<<HOLE k>>)` at that position. The correct answer is to
    decline. Real holes and decoys together turn the task into a judgement
    rather than a translation: filling a decoy is inventing a hypothesis the
    source never stated, declining a real one is dropping a standing assumption,
    and those are the two directions in which formalizations here actually fail.
    """

    def to_json(self) -> dict[str, Any]:
        # The released items have one real hole each: idx is its position, kind/decoy/anchor are
        # constant, so only the span (offsets into the item's `lean`) and its gold are stored.
        return {"start": self.start, "end": self.end, "gold": self.gold}


@dataclass
class MaskedItem:
    item_id: str
    paper_id: str
    target_name: str
    source: str
    holes: list[Hole]
    nl_statement: str = ""
    context: list[Any] | None = None
    nl_notes: str = ""
    lean: str = ""
    target_names: list[str] = field(default_factory=list)
    # Bookkeeping for topic statistics (paper Fig. 2); never rendered into a prompt.
    topic_domain: str = ""
    topic_subtopic: str = ""

    @cached_property
    def _target(self) -> Target:
        return Target.build(self.lean, self.target_name, self.target_names)

    @cached_property
    def _body_start(self) -> int:
        return self.lean.index(self._target.body)

    @property
    def stmt(self) -> str:
        """The file after its imports. Hole offsets are into `lean`, not into this."""
        return self._target.body

    @property
    def masked_stmt(self) -> str:
        return render_mask(self.stmt, _shift(self.holes, self._body_start))

    @property
    def masked_file(self) -> str:
        return render_mask(self.lean, self.holes)

    @property
    def masked_file_alone(self) -> str:
        """`masked_file` without the other targets' declarations (`--sibling-targets hidden`).

        A Research-level row has three or four targets, one item each. With the default
        `--sibling-targets shown` the solver sees the whole file, so the sibling targets' gold
        Lean statements are in view while it writes this one. The definitions, notation and
        `variable`s stay; grading still splices the answer into the full file (`splice_file`),
        siblings included. (`--sibling-targets joint` solves such a row as one `JointItem`.)
        """
        return _drop_decls(self.masked_file, set(self.target_names) - {self.target_name})

    @property
    def masked_header(self) -> str:
        """The target declaration alone, masked, up to the end of its last hole.

        `masked_stmt` is the whole body after the imports, not the declaration (ratio ~0.99 of
        the file), so it cannot double as the "statement to complete" pointer. This is the
        pointer instead.
        """
        lo = self._body_start + self._target.decl_start
        hi = self._body_start + self._target.decl_end
        mine = _shift([h for h in self.holes if lo <= h.start and h.end <= hi], lo)
        if not mine:
            raise ValueError(f"{self.item_id}: no hole inside the target declaration")
        cut = max(h.end for h in mine)
        return render_mask(self.lean[lo : lo + cut], mine).strip()

    def to_json(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "holes"}
        d["holes"] = [h.to_json() for h in self.holes]
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "MaskedItem":
        holes = [Hole(**{"idx": i, **h}) for i, h in enumerate(d.get("holes", []), start=1)]
        if "stmt" in d:  # an older item layout: offsets count from the end of the imports
            holes = _shift(holes, -d["lean"].index(d["stmt"]))
        fields = {k: v for k, v in d.items() if k in MaskedItem.__dataclass_fields__ and k != "holes"}
        return MaskedItem(holes=holes, **fields)


@dataclass
class JointItem:
    """All targets of one Research-level row, solved in one agent session.

    Like `prove`, the agent gets the whole file with every target's statement blanked, so no
    target's gold Lean statement is in view while it writes another. Holes are numbered in file
    order. Each member is still graded on its own: the answer row carries one statement per
    member item (`fills_by_item`), which `expand_joint_answers` turns back into per-item rows.
    """
    members: list[MaskedItem]

    def __post_init__(self) -> None:
        self.members = sorted(self.members, key=lambda m: m.holes[0].start)
        if len({m.paper_id for m in self.members}) != 1 or any(len(m.holes) != 1 for m in self.members):
            raise ValueError("a joint item is one row's single-hole items")

    @property
    def item_id(self) -> str:
        return f"{self.paper_id}::joint"

    @property
    def paper_id(self) -> str:
        return self.members[0].paper_id

    @property
    def source(self) -> str:
        return self.members[0].source

    @property
    def target_names(self) -> list[str]:
        return [m.target_name for m in self.members]

    @property
    def masked_file(self) -> str:
        """The row's file with each member's statement blanked; the other targets of the row
        (not in this run, e.g. under `--ids`) are removed rather than shown."""
        lean = self.members[0].lean
        holes = [Hole(**{**asdict(m.holes[0]), "idx": k})
                 for k, m in enumerate(self.members, start=1)]
        masked = render_mask(lean, holes)
        absent = set(self.members[0].target_names) - set(self.target_names)
        return _drop_decls(masked, absent)


def joint_units(items: list[MaskedItem]) -> list[MaskedItem | JointItem]:
    """Research-level items grouped into one JointItem per row; single-target rows unchanged."""
    groups: dict[str, list[MaskedItem]] = {}
    order: list[MaskedItem | str] = []
    for it in items:
        if len(it.target_names) <= 1:
            order.append(it)
            continue
        if it.paper_id not in groups:
            groups[it.paper_id] = []
            order.append(it.paper_id)
        groups[it.paper_id].append(it)
    return [JointItem(groups[u]) if isinstance(u, str) else u for u in order]


def expand_joint_answers(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A joint answer row -> one row per member item, each with its own `fills`."""
    out = []
    for a in answers:
        if "fills_by_item" not in a:
            out.append(a)
            continue
        for mid, stmt in a["fills_by_item"].items():
            row = {k: v for k, v in a.items() if k not in ("fills_by_item", "fills")}
            out.append({**row, "id": mid, "joint_id": a["id"],
                        "target_name": mid.rsplit("::", 1)[0].split("::", 1)[1],
                        "fills": {"1": stmt} if stmt else {}})
    return out


def _drop_decls(lean: str, names: set[str]) -> str:
    """`lean` without the theorem/lemma declarations named in `names`."""
    if not names:
        return lean
    _, body = split_imports(lean)     # rejoined lines: may lack the final newline
    lo = lean.index(body)
    keep = [c.text for c in parse_chunks(body)
            if not (c.kind in ("theorem", "lemma") and c.name in names)]
    return lean[:lo] + "".join(keep) + lean[lo + len(body):]


def _shift(holes: list[Hole], by: int) -> list[Hole]:
    return [Hole(**{**asdict(h), "start": h.start - by, "end": h.end - by}) for h in holes]


def is_decline(text: str | None) -> bool:
    return text is None or str(text).strip().strip("`").lower() in DECLINE_TOKENS


def render_mask(stmt: str, holes: list[Hole]) -> str:
    out, cursor = [], 0
    for h in sorted(holes, key=lambda x: x.start):
        out.append(stmt[cursor : h.start])
        out.append(f" (<<HOLE {h.idx}>>)" if h.decoy else f"<<HOLE {h.idx}>>")
        cursor = h.end
    out.append(stmt[cursor:])
    return "".join(out)


def splice(stmt: str, holes: list[Hole], fills: dict[int, str]) -> str:
    out, cursor = [], 0
    for h in sorted(holes, key=lambda x: x.start):
        got = fills.get(h.idx, None)
        declined = h.idx in fills and is_decline(got)
        seg = stmt[cursor : h.start]
        if h.decoy:
            out.append(seg)
            out.append("" if is_decline(got) else f" ({got})")
            cursor = h.end
            continue
        if declined:
            lo, hi = h.start, h.end
            if lo > 0 and stmt[lo - 1] in "([{" and hi < len(stmt) and stmt[hi] in ")]}":
                seg = stmt[cursor : lo - 1].rstrip(" \t")
                hi += 1
            out.append(seg)
            cursor = hi
            continue
        out.append(seg)
        out.append(str(got if got is not None else h.gold))
        cursor = h.end
    out.append(stmt[cursor:])
    return "".join(out)


def splice_file(item: MaskedItem, fills: dict[int, str]) -> str:
    return splice(item.lean, item.holes, fills)


def statement_span(decl) -> tuple[int, int]:
    """Where the statement sits in `decl.text`: right after the name, up to the PROOF's `:=`.

    Both ends matter. The name is located on a word boundary (`str.find` would match a short
    name inside the keyword, `t` in `theorem`). The end is the `:=` that introduces the proof
    (`Decl.sig` already stops there), NOT the first top-level `:=`: a statement whose type
    contains `let x := ...` has an earlier one, and cutting there would leave the rest of the
    statement -- conclusion included -- visible in the item.
    """
    if decl is None:
        raise ValueError("no declaration to take a statement from")
    sig = decl.sig
    start, end = len(sig) - len(decl.after_name), len(sig)
    while start < end and sig[start].isspace():
        start += 1
    while end > start and sig[end - 1].isspace():
        end -= 1
    if end <= start:
        raise ValueError(f"{decl.name}: empty statement")
    return start, end


def _label_of(row: dict[str, Any], target_name: str) -> str | None:
    if row.get("label"):
        return str(row["label"])
    for t in row.get("targets") or ():
        if target_name in (t.get("decls") or ()):
            return str(t.get("label"))
    return None


def target_nl(row: dict[str, Any], target_name: str) -> tuple[str, list[Any] | None]:
    """(statement, context) the auto-formalizer sees for ONE target.

    Every `targets` entry carries the target's `statement`. On a multi-target (Research-level) row
    the entry also carries the ids of the `context` entries it needs (`context_ids`) and the
    sibling targets whose statements it uses (`sibling_refs`). A single-target row's context is
    the row's own, so its entry does not repeat it.
    """
    label = _label_of(row, target_name)
    entries = [t for t in row.get("targets") or () if isinstance(t, dict)]
    entry = next((t for t in entries if t.get("label") == label), None)
    if entry is None:
        raise ValueError(f"{row.get('paper_id')}::{target_name}: no targets entry for {label!r}")
    if len(entries) == 1:
        return entry.get("statement") or "", row.get("context")
    if "context_ids" not in entry:
        # The row context of a multi-target row serves three results: never fall back.
        raise ValueError(f"{row.get('paper_id')}::{target_name}: no per-target NL for {label!r}")
    by_id = {d["id"]: d for d in row.get("context") or () if isinstance(d, dict)}
    siblings = {s.get("label"): s for s in entries}
    context: list[Any] = [by_id[i] for i in entry["context_ids"]]
    context += [{"name": f"{lab} (another result of the same paper)",
                 "statement": siblings[lab]["statement"]} for lab in entry.get("sibling_refs") or ()]
    return entry.get("statement") or "", context


@cache
def shipped_notes() -> dict[str, str]:
    """`nl_notes` per `<paper_id>::<declaration>`, read from the shipped item file.

    The notes (how the Lean statement reads or departs from the source) exist only for this
    task, so the item file is their only copy, not `forward-construction.jsonl`.
    A rebuild carries them over; it refuses to run without the file rather than drop them.
    """
    p = auto_formalization_jsonl()
    if not p.is_file():
        raise SystemExit(f"{p} is missing: it holds the only copy of the formalization notes")
    return {f"{r['paper_id']}::{r['target_name']}": r.get("nl_notes") or "" for r in load_jsonl(p)}


def build_item(row: dict[str, Any], target_name: str) -> MaskedItem:
    target = Target.build(row["lean"], target_name, row["target_names"])
    nl_statement, context = target_nl(row, target_name)
    nl_notes = shipped_notes().get(f"{row['paper_id']}::{target_name}", "")
    label = _label_of(row, target_name)
    topic = next((t for t in row.get("targets") or () if t.get("label") == label), {})
    # Only a context entry's name and text reach the model; ids and curation metadata stay out.
    context = [{"name": d.get("name"), "statement": d.get("statement")} if isinstance(d, dict)
               else d for d in context or ()] or None
    stmt = target.decl.text
    # Hole offsets are into the whole file; `target` counts from the end of the imports.
    base = row["lean"].index(target.body)
    holes: list[Hole] = []

    after, end = statement_span(target.decl)
    holes.append(Hole(idx=0, start=base + target.decl_start + after,
                      end=base + target.decl_start + end,
                      gold=stmt[after:end], kind="conclusion",
                      nl_anchor=nl_statement[:200]))
    holes.sort(key=lambda h: h.start)
    for i, h in enumerate(holes, start=1):
        h.idx = i

    return MaskedItem(
        item_id=f"{row['paper_id']}::{target_name}::A-stmt",
        paper_id=str(row["paper_id"]), target_name=target_name,
        source=str(row.get("source") or ""), holes=holes,
        nl_statement=nl_statement, context=context, nl_notes=nl_notes,
        lean=row["lean"], target_names=list(dict.fromkeys(row.get("target_names") or [])),
        topic_domain=str(topic.get("topic_domain") or ""),
        topic_subtopic=str(topic.get("topic_subtopic") or ""),
    )


UNANSWERED = "-- the model produced no statement for this target\n"


def to_predictions(items: dict[str, MaskedItem], answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preds, unanswered = [], 0
    for row in answers:
        item = items.get(row["item_id"])
        if item is None:
            continue
        fills = {int(k): v for k, v in (row.get("fills") or {}).items()}
        if row.get("timed_out"):
            lean, unanswered = UNANSWERED, unanswered + 1
            preds.append({"paper_id": item.paper_id, "target_name": item.target_name,
                          "sample_idx": int(row.get("run_idx", 0)), "lean": lean,
                          "unanswered": True, "timed_out": True,
                          "model": row.get("model", "")})
            continue
        if not fills or not str(fills.get(1, "")).strip():
            lean, unanswered = UNANSWERED, unanswered + 1
        else:
            lean = splice_file(item, fills)
        preds.append({
            "paper_id": item.paper_id, "target_name": item.target_name,
            "sample_idx": int(row.get("run_idx", 0)),
            "lean": lean, "unanswered": lean is UNANSWERED,
            "model": row.get("model", ""),
        })
    if unanswered:
        print(f"[mode-a] {unanswered}/{len(preds)} items had no usable statement "
              f"(none produced, or the turn hit its wall clock); emitted as "
              f"non-elaborating rather than spliced back to gold")
    return preds
