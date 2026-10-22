"""Strict scoring for FL verification (strict table + triviality gate), as in the paper.

Buckets (every item lands in one, a mislocated flawed item in two):

    flawed, flagged at the gold theorem AND step       -> TP
    correct, not flagged                               -> TN   (TN_cov if coverage == 1.0)
    flawed, a REAL flag at some other step             -> FP and FN   (mislocated)
    flawed, not flagged                                -> FN
    correct, flagged                                   -> FP

Rows (the FL pipeline) are re-decided from their per-step Lean states, not from the
pipeline's own summary:

  * vacuity gate: a `Success of Proof` whose statement assumes its own conclusion is not
    evidence; it is demoted to unproved (matters for the `prove` task).
  * malf rule `region` (the default): a `Failed Formalization` never
    accuses a step, but a correct item with ANY malformed step is not certified clean (FP),
    and a flawed item earns TP only if nothing at or before the gold step was malformed.
  * no-parse rule: a session that died or whose answer did not parse is
    a failure on both halves -- FP on a correct item, FN on a flawed one -- instead of falling
    through to a free "no error found".

The paper's NL baseline is an ablation and is not shipped.
"""
from __future__ import annotations

import collections
import re
from typing import Any

__all__ = ["tautology", "agent_flag", "score_rows"]


PROVED, UNPROVED, MALFORMED = "Success of Proof", "Failure of Proof", "Failed Formalization"
REFUTED = ("Refuted", "Entailment Refuted")

_BINDER = re.compile(r"^\s*(∀|forall|\\u2200)\s")


# --------------------------------------------------------------------- vacuity (triviality.py)
def _split_top(s: str, sep: str) -> list[str]:
    out, buf, d, i = [], [], 0, 0
    while i < len(s):
        c = s[i]
        if c in "([{⟨":
            d += 1
        elif c in ")]}⟩":
            d -= 1
        if d == 0 and s.startswith(sep, i):
            out.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        buf.append(c)
        i += 1
    out.append("".join(buf))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip("()").strip())


def _body(stmt: str) -> str:
    s = stmt.split(":=")[0]
    i = s.find(":")
    return s[i + 1:] if i >= 0 else s


def _strip_binders(b: str) -> str:
    prev = None
    while prev != b:
        prev = b
        b = b.strip()
        if _BINDER.match(b):
            j = b.find(",")
            if j > 0:
                b = b[j + 1:]
    return b


def tautology(stmt: str) -> str | None:
    """Is the conclusion literally one of the hypotheses (or a conjunction of them)?

    Syntactic, so a lower bound on vacuous statements."""
    b = _strip_binders(_body(stmt))
    parts = [p for p in _split_top(b, "→") if p.strip()]
    if len(parts) < 2:
        parts = [p for p in _split_top(b, "->") if p.strip()]
    if len(parts) < 2:
        return None
    hyps, concl = {_norm(p) for p in parts[:-1]}, _norm(parts[-1])
    if concl in hyps:
        return "conclusion IS a hypothesis"
    conj = [_norm(c) for c in _split_top(parts[-1], "∧")]
    if len(conj) > 1 and all(c in hyps for c in conj):
        return "conclusion is a conjunction of hypotheses"
    return None


# --------------------------------------------------------------------- agent rows
def agent_flag(per_theorem: list[dict[str, Any]], gold: tuple[str, Any], *, task: str = "refute",
               gate: bool = True, malf_rule: str = "region") -> tuple[bool, bool, bool, Any]:
    """(said, hit, accused, flagged_at) for one FL row, as strict_table.counts decides them."""
    flat = []
    for th in per_theorem:
        for s in sorted(th["steps"], key=lambda z: z["n"]):
            st = s["state"]
            if gate and st == PROVED and tautology(s.get("formal_statement") or ""):
                st = UNPROVED
            flat.append((str(th["local_id"]), s["n"], st))
    real = (lambda x: x == UNPROVED) if task == "prove" else (lambda x: x in REFUTED)
    gold_i = next((i for i, (t, n, _) in enumerate(flat) if (t, n) == gold), None)
    malf_i = [i for i, (_, _, st) in enumerate(flat) if st == MALFORMED]

    if malf_rule == "slot":
        flag_i = next((i for i, (_, _, st) in enumerate(flat) if real(st) or st == MALFORMED), None)
        said = flag_i is not None
        hit = said and flag_i == gold_i
        accused = said and not hit
    else:
        flag_i = next((i for i, (_, _, st) in enumerate(flat) if real(st)), None)
        if malf_rule == "item":
            blocked = bool(malf_i)
        elif malf_rule == "region":
            blocked = gold_i is not None and any(m <= gold_i for m in malf_i)
        else:
            blocked = False
        said = flag_i is not None or (malf_rule != "none" and bool(malf_i))
        hit = flag_i is not None and flag_i == gold_i and not blocked
        accused = flag_i is not None and flag_i != gold_i
    at = flat[flag_i][:2] if flag_i is not None else None
    return said, hit, accused, at


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score_rows(rows: list[dict[str, Any]], *, malf_rule: str = "region", gate: bool = True,
               parse_penalty: bool = True) -> tuple[dict[str, Any], list[dict]]:
    c: collections.Counter = collections.Counter()
    per_item = []
    for r in rows:
        g = r.get("gold") or {}
        flawed = g.get("label") == "flawed"
        c["flawed" if flawed else "correct"] += 1
        gold = (str(g.get("target_local_id")), g.get("error_step"))

        dead = r.get("parse_ok") is False or r.get("rc") not in (0, None)
        if dead:
            c["noparse"] += 1
        if dead and parse_penalty:
            bucket = "FN(no answer)" if flawed else "FP(no answer)"
            c["FN" if flawed else "FP"] += 1
            per_item.append({"id": r.get("id"), "label": g.get("label"), "bucket": bucket})
            continue

        said, hit, accused, at = agent_flag(r["per_theorem"], gold, gate=gate,
                                            malf_rule=malf_rule,
                                            task=r.get("fl_task", "refute"))
        if not flawed:
            if said:
                bucket = "FP(false alarm)"
                c["FP"] += 1
                c["FA"] += 1
            else:
                bucket = "TN"
                c["TN"] += 1
                c["TN_cov"] += int((r.get("coverage") or 0) >= 1.0)
        elif hit:
            bucket = "TP"
            c["TP"] += 1
        elif accused:
            bucket = "FP+FN(mislocated)"
            c["FP"] += 1
            c["FN"] += 1
            c["mis"] += 1
        else:
            bucket = "FN(missed)"
            c["FN"] += 1
        per_item.append({"id": r.get("id"), "label": g.get("label"), "bucket": bucket,
                         "flagged_at": list(at) if at else None,
                         "gold_at": [gold[0], gold[1]] if flawed else None})

    n = len(rows)
    TP, FP, FN, TN = c["TP"], c["FP"], c["FN"], c["TN"]
    p, r, f1 = _prf(TP, FP, FN)
    # THE DEFAULT HEADLINE: TP by location (the FL pipeline never classifies), raw accuracy
    # (TP + TN)/n with every TN counted, checked or not.
    summary = {
        "precision": p, "recall": r, "F1": f1,
        "accuracy": (TP + TN) / n if n else 0.0,
        "headline_basis": "strict, TP by location, raw accuracy",
        "n": n, "n_flawed": c["flawed"], "n_correct": c["correct"],
        "TP": TP, "FP": FP, "FN": FN, "TN": TN, "TN_cov": c["TN_cov"],
        "mislocated": c["mis"], "false_alarms": c["FA"], "no_answer": c["noparse"],
        "acc_cov": (TP + c["TN_cov"]) / n if n else 0.0,
        "FA_rate": c["FA"] / c["correct"] if c["correct"] else 0.0,
        "MIS_rate": c["mis"] / c["flawed"] if c["flawed"] else 0.0,
        "rules": {"malf_rule": malf_rule, "vacuity_gate": gate, "parse_penalty": parse_penalty},
    }
    return summary, per_item
