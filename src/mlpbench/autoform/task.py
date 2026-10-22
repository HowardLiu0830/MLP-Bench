"""Auto-formalization: write the Lean statement of a theorem from its natural language.

Items (`Benchmarks/auto-formalization.jsonl`) are built from forward-construction: one per
target declaration, with that declaration's statement blanked to `<<HOLE 1>>` and the
local definitions kept (mask level `stmt`). 121 declarations; the scored unit is the
target LABEL, of which there are 120 (`v23-20-411` Lemma 5 has two declarations and counts
as correct only if both are).

Modes:
  cot    one tool-free call, answer `{"fills": {"1": "<statement>"}}`
  agent  the model edits `answer.lean` in a sandbox with Lean; the FILE is the answer

Evaluation: the predicted file must typecheck, then BEq (both directions, gated, see
`beq.py`) against the gold statement, and a refutation attempt (`refute.py`) on the
prediction alone. Headline: BEq over targets; also the validity rate (compiles and not
refuted -- paper §2.2.1), the typecheck rate and the 4-way relation.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..benchmarks.forward_construction import canonical_source
from ..core.chat import COT_TIMEOUT, complete
from ..core.jsonio import (Appender, extract_json, load_jsonl, subsample, write_json_atomic,
                           write_jsonl_atomic)
from ..core.paths import auto_formalization_jsonl
from ..core.registry import ModelEntry
from . import bench as BN
from . import items as IT
from . import solver_prompts as SP
from .beq import BEqConfig, beq_pair, gates_enabled
from .grading import build_pairs, resolve_pred_target
from .leansurgery import Target, typecheck_file
from .refute import refute_prediction
from .solve import solve_one as solve_agent_one

__all__ = ["build_items", "load_items", "solve", "evaluate", "COT_SYSTEM", "cot_user"]

# The one-shot solver prompt of the paper's original fill pipeline, verbatim. It also covers
# decoy holes; in mode A there is exactly one real hole.
COT_SYSTEM = """You complete a Lean 4 (Mathlib) theorem statement.

You are given a Lean file in which parts of one theorem's statement have been replaced by
placeholders <<HOLE 1>>, <<HOLE 2>>, ... , together with the natural-language mathematics
that the theorem formalizes. Restore the missing text so that the statement says exactly
what the natural language says -- no more general, no more specific.

SOME PLACEHOLDERS STAND FOR NOTHING. A hole written as `(<<HOLE k>>)` in the binder list
may be a slot where the correct statement has no hypothesis at all. Answer `NONE` for those.
Judgement about whether a hypothesis belongs there is part of the task:

* Answering `NONE` where a hypothesis IS required silently weakens nothing and strengthens
  the theorem into something false -- the classic autoformalization error.
* Inventing a hypothesis where the source states none makes the theorem weaker than the
  mathematics it claims to formalize.

Both are wrong. Decide from the natural language, not from how many holes there are.

* Everything else in the file is fixed: use the definitions, notation and variable names
  that are already there.
* Give the text that replaces each placeholder and nothing else -- no `theorem` line, no
  parentheses that were not part of the hole, no proof.
* A hypothesis hole covers the binder's NAME as well as its type: answer `hmeas : Measurable f`,
  not `Measurable f`.
* Each answer must be valid Lean 4 that elaborates in place. Never answer `sorry`.

Answer with JSON only, no prose and no code fence:

{"fills": {"1": "<text for hole 1>", "2": "NONE", "3": "<text for hole 3>"}}"""


def cot_user(item: IT.MaskedItem, nl: str, sibling_targets: str = "hidden") -> str:
    """The Lean appears ONCE: the masked file, then only the target's masked header.

    `sibling_targets` "shown" keeps the row's other targets (gold statements included) in the
    file, "hidden" drops them.
    """
    file = item.masked_file_alone if sibling_targets == "hidden" else item.masked_file
    slots = ", ".join(f"<<HOLE {i}>>" for i in range(1, len(item.holes) + 1))
    return (f"LEAN FILE (with holes):\n```lean\n{file}\n```\n\n"
            f"TARGET THEOREM: `{item.target_name}`\n\nSTATEMENT TO COMPLETE:\n"
            f"```lean\n{item.masked_header}\n```\n\nNATURAL LANGUAGE:\n{nl}\n\nFill: {slots}")


# ------------------------------------------------------------------ items
def build_items(benchmark: Path | None = None) -> list[IT.MaskedItem]:
    built, failed = [], []
    for d in BN.decls(BN.load(benchmark)):
        try:
            built.append(IT.build_item(d, d["target_name"]))
        except Exception as exc:  # a missing item silently drops out of every rate
            failed.append(f"{d['paper_id']}::{d['target_name']}: {exc}")
    if failed:
        raise SystemExit("could not build items:\n  " + "\n  ".join(failed))
    return built


def load_items(path: Path | None = None, *, source: str | None = None, limit: int = 0,
               seed: int | None = None, ids: list[str] | None = None) -> list[IT.MaskedItem]:
    p = Path(path) if path else auto_formalization_jsonl()
    if not p.is_file():
        raise SystemExit(f"{p} is missing; build it with `mlpbench autoform-items`.")
    rows = load_jsonl(p)
    if source:
        want = canonical_source(source)
        if want is None:
            raise SystemExit(f"unknown source {source!r}")
        rows = [r for r in rows if r.get("source") == want]
    if ids:
        want = set(ids)
        rows = [r for r in rows if r["item_id"] in want]
    # The file is one source then the other, so a prefix is not a sample.
    rows = subsample(rows, limit=limit or None, seed=seed, stratify_on="source")
    return [IT.MaskedItem.from_json(r) for r in rows]


# ------------------------------------------------------------------ solving
def _clean(text: str) -> str:
    text = (text or "").strip()
    fence = re.search(r"```(?:lean4?|lean)?\s*\n(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    return text.strip().strip("`").strip()


def _solve_cot(item: IT.MaskedItem, entry: ModelEntry, *, timeout: int, no_nl: bool,
               sessions: Path, sibling_targets: str = "hidden") -> dict[str, Any]:
    nl = ("(withheld -- reconstruct the statement from the Lean context alone)" if no_nl
          else SP.nl_for({"nl_statement": item.nl_statement, "context": item.context,
                          "nl_notes": item.nl_notes}))
    stem = re.sub(r"[^A-Za-z0-9]+", "_", item.item_id)[:60]
    res = complete(entry, COT_SYSTEM, cot_user(item, nl, sibling_targets), timeout=timeout, raw_dir=sessions,
                   stem=stem)
    fills: dict[str, str] = {}
    parse_error = None
    try:
        raw = extract_json(res.text).get("fills")
        if isinstance(raw, dict):
            for k, v in raw.items():
                m = re.search(r"\d+", str(k))
                if m and isinstance(v, str):
                    fills[m.group(0)] = _clean(v)
        else:
            parse_error = "the JSON object has no `fills` mapping"
    except ValueError as exc:
        parse_error = str(exc)[:300]
    return {
        "id": item.item_id, "paper_id": item.paper_id, "target_name": item.target_name,
        "fills": fills, "parse_error": parse_error, "sibling_targets": sibling_targets,
        "returncode": res.returncode, "timed_out": res.timed_out,
        "n_tool_calls": len(res.tool_calls), "tokens": res.usage.as_dict(),
        "error": None if res.ok else (res.stderr or "empty reply")[-500:],
        "status": "answered" if fills.get("1") else "NO ANSWER",
    }


def solve(item: IT.MaskedItem, entry: ModelEntry, mode: str, sessions: Path, *,
          timeout: int | None = None, no_nl: bool = False,
          sibling_targets: str = "hidden") -> dict[str, Any]:
    """`sibling_targets` for a one-target item ("hidden" / "shown"); a JointItem is "joint"."""
    if timeout is None:
        timeout = COT_TIMEOUT if mode == "cot" else 3600
    if sibling_targets == "joint" and not isinstance(item, IT.JointItem):
        # a one-target row under `joint` (every Graduate-level row): no siblings to blank, and
        # its masked_file and masked_file_alone are the same file
        sibling_targets = "hidden"
    if mode == "cot":
        return _solve_cot(item, entry, timeout=timeout, no_nl=no_nl, sessions=sessions,
                          sibling_targets=sibling_targets)
    row = solve_agent_one(item, entry, mode="agent", timeout=timeout, no_nl=no_nl,
                          raw_dir=sessions, sibling_targets=sibling_targets)
    row["id"] = row.pop("item_id")
    if row.get("joint"):
        got = sum(1 for v in row.get("fills_by_item", {}).values() if v)
        n = len(item.members)
        row["status"] = (f"answered {got}/{n} lean={row.get('n_lean_calls')}" if got
                         else "NO ANSWER")
    else:
        row["status"] = (f"answered lean={row.get('n_lean_calls')}" if row.get("fills")
                         else "NO ANSWER")
    return row


# ------------------------------------------------------------------ evaluation
def _slug(c: BEqConfig) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", c.sampler.alias if c.sampler else "none")


PARTIAL = "scores.partial.jsonl"            # one line per graded pair, written as each finishes
PARTIAL_META = "scores.partial.meta.json"   # the grader settings those lines were graded with


def _grading_settings(cfg: BEqConfig, refute: bool,
                      second: BEqConfig | None = None) -> dict[str, Any]:
    """What a resumed grading must share with the lines it reuses."""
    return {"grader": getattr(cfg.sampler, "alias", None),
            **({"double_check": second.sampler.alias} if second and second.sampler else {}),
            "grader_effort": getattr(cfg.sampler, "effort", None), "agentic": cfg.agentic,
            "n": cfg.n, "sampler_timeout": cfg.sampler_timeout, "refute": refute,
            "gates": gates_enabled(cfg)}


def evaluate(run_dir: Path, answers: list[dict[str, Any]], *, items_path: Path | None = None,
             cfg: BEqConfig, jobs: int = 4, refute: bool = True,
             resume: bool = False, second: BEqConfig | None = None) -> dict[str, Any]:
    """Typecheck + BEq + refutation for every answered item; per-target summary.

    The refutation uses the BEq grader (`cfg.sampler`, same agentic setting and
    per-session budget). `refute=False`, or no grader model, leaves `valid` equal to
    `typecheck` -- an upper bound, recorded under `refutation.not_run`.

    `second` (double check, `--double-check`): a second grader runs BEq and the refutation on
    the same pair, alongside the first. A pair is BEq if EITHER grader proves both directions,
    and refuted if EITHER grader's counterexample is accepted. Each grader's own result is kept
    under `by_grader`; the top-level fields come from the grader that decided (the first one
    when neither did), and `summary.per_grader` scores each grader alone."""
    import concurrent.futures as cf

    items = {it.item_id: it for it in load_items(items_path)}
    answers = IT.expand_joint_answers(answers)      # a joint row is graded per member
    run_items = {a["id"]: items[a["id"]] for a in answers if a["id"] in items}
    preds = IT.to_predictions(run_items, [{**a, "item_id": a["id"]} for a in answers])
    gold = [{"paper_id": it.paper_id, "target_name": it.target_name, "source": it.source,
             "lean": it.lean, "target_names": it.target_names} for it in run_items.values()]
    pairs, unmatched = build_pairs(gold, preds)

    def one(pair) -> dict[str, Any]:
        base = {"paper_id": pair.paper_id, "target_name": pair.target_name, "source": pair.source}
        if not pair.pred_lean.strip() or pair.pred_lean.startswith("-- the model produced no"):
            return {**base, "typecheck": False, "beq": False, "valid": False,
                    "relation": "unanswered"}
        tc = typecheck_file(pair.pred_lean, lean_project=cfg.lean_project, timeout=cfg.timeout,
                            work_dir=cfg.work_dir)
        if not tc["ok"]:
            return {**base, "typecheck": False, "beq": False, "valid": False,
                    "relation": "does_not_elaborate", "typecheck_detail": tc}
        stem = re.sub(r"[^A-Za-z0-9]+", "_", pair.pair_id)[:60]
        try:
            ref = Target.build(pair.ref_lean, pair.target_name, pair.ref_targets)
            name, thms = resolve_pred_target(pair.pred_lean, pair.pred_target)
            pred = Target.build(pair.pred_lean, name, thms)
        except Exception as exc:
            ref = pred = None
            built_error = f"{type(exc).__name__}: {exc}"

        def grade(c: BEqConfig, stem: str) -> tuple[dict[str, Any], dict[str, Any]]:
            if pred is None:
                return ({"beq": False, "relation": "harness_error", "error": built_error},
                        {"ran": False, "refuted": False,
                         "reason": "harness error before refutation"})
            try:
                res = beq_pair(ref, pred, c, stem=stem)
            except Exception as exc:
                res = {"beq": False, "relation": "harness_error",
                       "error": f"{type(exc).__name__}: {exc}"}
            # Validity is independent of the gold: it runs on every prediction that compiles,
            # BEq-equivalent ones included (a refuted equivalent would mean a refuted gold).
            if not refute:
                return res, {"ran": False, "refuted": False,
                             "reason": "refutation disabled (--no-refute)"}
            try:
                return res, refute_prediction(pred, c, stem=stem + ".refute")
            except Exception as exc:  # noqa: BLE001 - recorded; valid stays an upper bound
                return res, {"ran": False, "refuted": False,
                             "reason": f"harness error: {type(exc).__name__}: {exc}"}

        extra: dict[str, Any] = {}
        if second is None:
            res, ref_res = grade(cfg, stem)
        else:
            graders = [cfg, second]
            with cf.ThreadPoolExecutor(max_workers=2) as ex:
                both = list(ex.map(lambda c: grade(c, f"{stem}.{_slug(c)}"), graders))
            extra["by_grader"] = {
                c.sampler.alias: {**r, "refute": rr, "refuted": bool(rr.get("refuted"))}
                for c, (r, rr) in zip(graders, both)}
            res = next((r for r, _ in both if r.get("beq")), both[0][0])
            ref_res = next((rr for _, rr in both if rr.get("refuted")), both[0][1])
        return {**base, "typecheck": True, **res, "refute": ref_res,
                "refuted": bool(ref_res.get("refuted")),
                "valid": not ref_res.get("refuted"), **extra}

    # Each graded pair goes to PARTIAL the moment it finishes (fsync'd), and one progress line
    # is printed, flushed, so a long BEq + refutation pass is visible in a log file and a killed
    # pass loses only the pairs in flight. `resume` reuses PARTIAL's lines when the grader
    # settings match PARTIAL_META; scores.jsonl is written, in pair order, once all are done.
    import threading
    import time

    settings = _grading_settings(cfg, refute, second)
    part, meta = run_dir / PARTIAL, run_dir / PARTIAL_META
    if part.exists() and not resume:
        raise SystemExit(f"{part} holds graded pairs from an unfinished grading; pass --resume "
                         f"to reuse them (same grader settings) or delete it to start over")
    if resume and part.exists() and meta.exists():
        old = json.loads(meta.read_text(encoding="utf-8"))
        if old != settings:
            raise SystemExit(f"--resume would mix grader settings ({old} vs {settings}); "
                             f"delete {part} to start over")
    write_json_atomic(meta, settings)
    app = Appender(part, id_key="pair_id", resume=True)
    reused = {str(r["pair_id"]): r for r in load_jsonl(part)} if part.exists() else {}
    todo = [p for p in pairs if p.pair_id not in reused]

    lock, done, t0 = threading.Lock(), [len(pairs) - len(todo)], time.time()
    print(f"[autoform-eval] {len(pairs)} pairs ({len(pairs) - len(todo)} already graded), "
          f"jobs={jobs}, grader={settings['grader']}"
          + (f" + {settings['double_check']} (double check)" if second else "")
          + f", refute={refute}", flush=True)

    def timed(pair) -> dict[str, Any]:
        t = time.time()
        s = {**one(pair), "pair_id": pair.pair_id}
        with lock:
            app.write(s)
            done[0] += 1
            ref = s.get("refute") or {}
            refuted = ("-" if not ref.get("ran") else int(bool(s.get("refuted"))))
            print(f"[autoform-eval] {done[0]}/{len(pairs)} {pair.pair_id[:58]:58s} "
                  f"tc={int(bool(s.get('typecheck')))} beq={int(bool(s.get('beq')))} "
                  f"refuted={refuted} {s.get('relation', '')}  "
                  f"({time.time() - t:.0f}s; {(time.time() - t0) / 60:.0f}m elapsed)", flush=True)
        return s

    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        fresh = {s["pair_id"]: s for s in ex.map(timed, todo)}
    scored = [reused.get(p.pair_id) or fresh[p.pair_id] for p in pairs]
    write_jsonl_atomic(run_dir / "scores.jsonl", scored)
    part.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)

    # Per target LABEL: v23-20-411's two Lemma-5 declarations count once, AND-ed.
    by_decl = {f"{s['paper_id']}::{s['target_name']}": s for s in scored}
    records = {r["paper_id"]: r for r in BN.load()}
    targets: dict[str, dict[str, bool]] = {}
    alone: dict[str, dict[str, int]] = {}         # double check: each grader on its own
    for pid in {it.paper_id for it in run_items.values()}:
        for t in records[pid]["targets"]:
            keys = [f"{pid}::{d}" for d in t["decls"]]
            if not all(k in by_decl for k in keys):
                continue          # a label only partly in this run is not scored
            targets[f"{pid}::{t['label']}"] = {
                m: all(bool(by_decl[k].get(m)) for k in keys)
                for m in ("typecheck", "beq", "valid")}
            for c in ([cfg, second] if second else []):
                g = c.sampler.alias
                one_g = [(by_decl[k].get("by_grader") or {}).get(g) for k in keys]
                acc = alone.setdefault(g, {"beq": 0, "valid": 0})
                acc["beq"] += all(bool(r and r.get("beq")) for r in one_g)
                acc["valid"] += all(bool(by_decl[k].get("typecheck"))
                                    and not (r and r.get("refuted"))
                                    for k, r in zip(keys, one_g))
    n = len(targets)

    def rate(m: str) -> float:
        return (sum(v[m] for v in targets.values()) / n) if n else 0.0

    directions = [s[d] for s in scored for d in ("ref_from_pred", "pred_from_ref")
                  if isinstance(s.get(d), dict) and not s[d].get("skipped")]
    refutes = [s["refute"] for s in scored if isinstance(s.get("refute"), dict)]
    summary = {
        "task": "autoform", "n_items": len(answers), "n_targets": n,
        "beq": sum(v["beq"] for v in targets.values()),
        "valid": sum(v["valid"] for v in targets.values()),
        "typecheck": sum(v["typecheck"] for v in targets.values()),
        "beq_rate": rate("beq"),                  # post-gate whenever the gates are on
        "valid_rate": rate("valid"),              # compiles AND no accepted refutation
        "typecheck_rate": rate("typecheck"),
        "relation": dict(Counter(s.get("relation") for s in scored)),
        # Directions whose only compiling candidates a gate threw out (pre-gate yes, post no).
        "gate_rejected_directions": sum(1 for d in directions
                                        if d.get("ok_before_gates") and not d.get("beq")),
        "refutation": {
            "attempted": sum(1 for r in refutes if r.get("ran")),
            "refuted": sum(1 for r in refutes if r.get("refuted")),
            "not_run": dict(Counter(r.get("reason") for r in refutes if not r.get("ran"))),
            "rejected": dict(Counter((r.get("reason") or "")[:60] for r in refutes
                                     if r.get("ran") and not r.get("refuted"))),
            "agent_rows_without_lean": sum(1 for r in refutes if r.get("agentic")
                                           and r.get("n_lean_calls") == 0),
        },
        "agent_rows_without_lean": sum(1 for a in answers if a.get("n_lean_calls") == 0),
        "agent_rows_egress_suspect": [a.get("item_id") or a.get("id")
                                      for a in answers if a.get("egress_suspect")],
        "cot_rows_with_tools": sum(1 for a in answers if (a.get("n_tool_calls") or 0) > 0
                                   and a.get("mode") == "cot"),
        "unmatched": unmatched,
        "grader": {"sampler": cfg.sampler.alias if cfg.sampler else None,
                   "effort": cfg.sampler.effort if cfg.sampler else None,
                   "agentic": cfg.agentic, "n": cfg.n, "gates": gates_enabled(cfg),
                   "refute": bool(refute), "session_timeout": cfg.sampler_timeout,
                   **({"double_check": second.sampler.alias} if second else {})},
        **({"per_grader": alone} if second else {}),
        "per_target": targets,
    }
    write_json_atomic(run_dir / "summary.json", summary)
    return summary
