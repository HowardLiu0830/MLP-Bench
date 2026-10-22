"""Validity: does a predicted statement survive a Lean-checked attempt to refute it?

Paper §2.2.1: "the proportion of generated statements that compile and that neither agent
refutes"; B.2: the evaluator's "sole function is to search for Lean proofs ... a refutation
for validity, all of which are accepted or rejected by the Lean kernel".

For a prediction that compiles, an evaluator agent (the BEq grader: same model, same
sandbox, same per-session budget) gets the predicted file with one theorem added right after
the target, in the target's own scope,

    theorem af_refutation : ¬ (∀ <file variables in scope> <target binders>, <conclusion>)

and tries to prove it. A refutation counts only if, after the harness has rebuilt the file
from ITS OWN text (definitions, the target, the `af_refutation` statement) plus the agent's
proof and any helper lemmas, Lean compiles it and `#print axioms af_refutation` shows no
`sorryAx` and nothing beyond the standard axioms. `native_decide` and `plausible` are
refused outright (the first adds `Lean.ofReduceBool`, the second closes goals it merely
failed to falsify), as are kernel-check switches and inline metaprograms (`BANNED`).
valid = compiles AND no accepted refutation.

`valid` is an UPPER bound: no refutation within budget is not a proof of truth.

Follows the refutation gate of the paper's original pipeline. Deliberate differences, all of which can only
make a refutation harder to accept, never easier:
  * acceptance is `#print axioms` on `af_refutation` (original: compile with no `sorry`
    diagnostic plus a whole-file integrity check), so unrelated `sorry` targets may stay
    in the file and the context is the prediction's file unchanged;
  * the harness re-assembles the checked file from its own text, keeping only the agent's
    proof and new `theorem`/`lemma`/`def`/`abbrev` helpers (no `instance`, no notation, no
    name that the statement mentions), so no edit to the context can change what the goal
    means;
  * file `variable`s are quantified only where they are in SCOPE at the target (the original
    took every `variable` line above it, including those of closed sections, which
    does not elaborate when a later section reuses the names);
  * the refutation sits after the target inside its namespace and sections (the original
    rebuilt the file without the other `sorry` targets), and a preceding `open ... in`
    is repeated so the goal elaborates under the same `open`s as the target;
  * the prompt does not state a time budget; when the agent leaves the file unchanged, the last ```lean block of its
    reply is tried instead (the original scored that as zero).
"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..core import backends as B
from ..core.chat import complete
from ..core.jsonio import trim_log
from ..core.lean import parse_axioms, print_axioms
from ..core.leantext import normalize_ws, strip_lean_comments
from ..core.paths import workspace_root
from . import leansurgery as A

__all__ = ["REFUTE_NAME", "RefuteError", "scoped_variables", "negation_goal",
           "build_skeleton", "assemble", "axioms_clean", "refute_prediction",
           "SYSTEM", "USER_TEMPLATE"]

REFUTE_NAME = "af_refutation"
REFUTE_FILE = "refutation.lean"
# Never acceptable in the agent's text, whatever `#print axioms` says: proof holes, the
# compiler-trusting deciders, and the meta-level escape hatches (kernel-check switches and
# inline metaprograms) that could reach past the statement.
BANNED = re.compile(r"(?<![\w.'])(sorry|admit|native_decide|plausible|axiom|extern)(?![\w'])"
                    r"|(ofReduceBool|skipKernelTC|implemented_by|run_tac|run_cmd|run_elab)")
HELPER_KINDS = ("theorem", "lemma", "def", "abbrev")


class RefuteError(ValueError):
    """The negation goal could not be built; the item is not refuted (valid stays an
    upper bound) and the reason is recorded."""


# ------------------------------------------------------------------ negation goal
def scoped_variables(body: str, offset: int) -> list[str]:
    """The `variable` binders in scope at `offset`, in order.

    Each `section`/`namespace` opens a scope and `end` closes it, discarding the variables
    declared inside; a variable from a closed section is not a hypothesis of the target and
    usually does not even elaborate there (its type mentions names of that section)."""
    if re.search(r"^\s*include\s", strip_lean_comments(body[:offset]), re.M):
        # `include` would make a variable a hypothesis of `af_refutation` itself.
        raise RefuteError("the file uses `include` above the target")
    stack: list[list[str]] = [[]]
    for c in A.parse_chunks(body):
        if c.start >= offset:
            break
        if c.kind in ("section", "namespace"):
            stack.append([])
        elif c.kind == "end":
            if len(stack) > 1:
                stack.pop()
        elif c.kind in ("variable", "variables"):
            text = strip_lean_comments(c.text)
            m = re.search(r"\bvariables?\b", text)
            binders = normalize_ws(text[m.end():] if m else "")
            if binders:
                stack[-1].append(binders)
    return [v for scope in stack for v in scope]


def _open_in_prefix(body: str, decl_start: int) -> str:
    """`open X in` / `set_option o v in` lines directly above the target: they scope over
    that one declaration, so the refutation must repeat them to read the same."""
    chunks = [c for c in A.parse_chunks(body) if c.start < decl_start]
    pre: list[str] = []
    for c in reversed(chunks):
        text = strip_lean_comments(c.text).strip()
        if not text:
            continue
        if c.kind in ("open", "set_option") and re.search(r"\bin$", text):
            pre.insert(0, text)
            continue
        break
    return "".join(t + "\n" for t in pre)


def negation_goal(target: A.Target) -> str:
    """`¬ (∀ <variables in scope> <binders>, <conclusion>)` for the target as stated.

    Both telescopes go INSIDE the `¬`. Leaving the file variables free would refute
    `∀ E, ¬ P E` -- the statement fails for EVERY carrier -- which is strictly stronger
    than the statement being false (the original gate quantifies them the same way). Quantifying a
    variable the target does not use is harmless in the safe direction: it can only make
    the negation harder to prove."""
    d = target.decl
    if d is None:
        raise RefuteError("the target has no parsable signature")
    j = A.find_toplevel(d.after_name, ":", skip="=")
    if j < 0 or not d.conclusion:
        raise RefuteError("could not split the target's binders from its conclusion")
    binders = normalize_ws(strip_lean_comments(d.after_name[:j]))
    concl = strip_lean_comments(d.conclusion).strip()
    telescope = " ".join(x for x in (*scoped_variables(target.body, target.decl_start),
                                     binders) if x).strip()
    inner = f"∀ {telescope},\n      {concl}" if telescope else concl
    return f"¬ (\n    {inner})"


def _signature(goal: str, prefix: str) -> str:
    return f"{prefix}theorem {REFUTE_NAME} :\n  {goal}"


def build_skeleton(target: A.Target, *, heartbeats: int = 0,
                   proof: str = "  sorry", helpers: str = "") -> dict[str, Any]:
    """The predicted file with `af_refutation` inserted right after the target.

    Returns the parts the checker reassembles from (`pre`, `sig`, `post`) and the text the
    agent is given (`text`), with `proof` as the refutation's tactic block."""
    goal = negation_goal(target)
    prefix = _open_in_prefix(target.body, target.decl_start)
    imports = "\n".join(target.imports) or "import Mathlib"
    head = imports + "\n\n"
    if heartbeats:
        head += f"set_option maxHeartbeats {heartbeats}\n\n"
    pre = head + target.body[: target.decl_end].rstrip("\n") + "\n\n"
    sig = _signature(goal, prefix)
    post = "\n" + target.body[target.decl_end:]
    ns = A.enclosing_namespace(target.body, target.decl_start)
    text = pre + (helpers.rstrip() + "\n\n" if helpers.strip() else "") + \
        sig + " := by\n" + proof.rstrip() + "\n" + post
    return {"goal": goal, "pre": pre, "sig": sig, "post": post, "text": text,
            "full_name": f"{ns}.{REFUTE_NAME}" if ns else REFUTE_NAME}


# ------------------------------------------------------------------ the agent's answer
def _identifiers(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_Ͱ-Ͽᴀ-ᶿ][\w'Ͱ-Ͽ]*", text))


def assemble(skel: dict[str, Any], completed: str) -> tuple[str | None, str]:
    """(file to check, why) from the agent's file. The file is the harness's own
    `pre` + `sig` + `post` around the agent's proof and new helper declarations; None and
    a reason when the answer cannot be scored.

    A declaration of the original file -- same name, or same text for an anonymous one --
    is not a helper: the harness's copy is used, so editing it changes nothing."""
    orig = A.parse_chunks(A.split_imports(skel["text"])[1])
    original_names = {c.name for c in orig if c.name}
    original_texts = {normalize_ws(strip_lean_comments(c.text)) for c in orig}
    _, body = A.split_imports(completed)
    chunks = A.parse_chunks(body)
    ref = next((c for c in chunks if c.kind in ("theorem", "lemma") and c.name == REFUTE_NAME),
               None)
    if ref is None:
        return None, f"no `{REFUTE_NAME}` declaration in the answer"
    k = A.find_toplevel(ref.text, ":=")
    if k < 0:
        return None, f"`{REFUTE_NAME}` has no proof"
    head = re.sub(r"^(open|set_option)\b.*\bin\s*$", "", strip_lean_comments(skel["sig"]),
                  flags=re.M)
    if normalize_ws(strip_lean_comments(ref.text[:k])) != normalize_ws(head):
        return None, f"the statement of `{REFUTE_NAME}` was changed"
    proof = ref.text[k + 2:].rstrip()
    if not proof.strip():
        return None, "empty proof"
    goal_ids = _identifiers(skel["goal"])
    helpers: list[str] = []
    for c in chunks:
        if c.start >= ref.start:
            break
        if c.kind not in A.DECL_KEYWORDS or (c.name and c.name in original_names) \
                or normalize_ws(strip_lean_comments(c.text)) in original_texts:
            continue
        if c.kind not in HELPER_KINDS:
            return None, (f"helper `{c.name or '(anonymous)'}` is a `{c.kind}`; only "
                          f"{', '.join(HELPER_KINDS)} are kept")
        if re.search(r"@\[[^\]]*\binstance\b", c.text):
            return None, f"helper `{c.name}` is an instance"
        if not c.name or c.name.split(".")[-1] in goal_ids:
            return None, f"helper `{c.name}` shadows a name the statement uses"
        helpers.append(c.text.rstrip())
    agent_text = proof + "\n" + "\n".join(helpers)
    bad = BANNED.search(strip_lean_comments(agent_text))
    if bad:
        return None, f"banned token `{bad.group(1) or bad.group(2)}`"
    text = skel["pre"] + ("\n\n".join(helpers) + "\n\n" if helpers else "") + \
        skel["sig"] + " :=" + proof + "\n" + skel["post"]
    return text, ""


def axioms_clean(out: str, full_name: str) -> tuple[bool, str]:
    """`#print axioms` verdict for the refutation: no sorryAx, standard axioms only."""
    v = parse_axioms(out, [full_name])[full_name]
    return bool(v.get("solved")), str(v.get("reason") or "")


def _last_lean_block(text: str) -> str | None:
    blocks = re.findall(r"```(?:lean4?)?\s*\n(.*?)```", text or "", flags=re.S)
    return blocks[-1] if blocks else None


# ------------------------------------------------------------------ prompt
# The original gate's audit prompt, verbatim except for the file name and the
# helper rule (the harness keeps only proofs and new lemmas/defs, see `assemble`).
SYSTEM = """You are an expert Lean 4 / Mathlib proof engineer working AUTONOMOUSLY.

You are given a Lean statement. Your ONLY job is to decide whether it is FALSE, by proving its
negation in Lean 4 with Mathlib:

    theorem af_refutation : ¬ (<the statement>) := by
      <your proof>

The standard shape is to `intro` the universally quantified claim, instantiate it at concrete
witnesses that violate it, and derive `False` with `norm_num` / `linarith` / `nlinarith` / `simp`.

Rules:
  * NO `sorry`, no `admit`, no new axioms, no `native_decide`, no `plausible`. The proof must be
    complete: the harness runs `#print axioms af_refutation` and rejects anything resting on
    `sorryAx` or a non-standard axiom.
  * The `theorem af_refutation : ...` line is FIXED. Do not reword it, do not delete it, do not
    replace it with anything else. A modified or missing statement scores zero.
  * Everything else already in the file is FIXED too: the harness rebuilds the file from its own
    copy and keeps only your proof of `af_refutation` and any NEW `theorem`/`lemma`/`def`
    declarations you place directly above it. Edits to existing definitions, `open`s,
    `variable`s, instances or notation are discarded.
  * DO NOT try to prove the statement itself. A positive proof, a partial positive proof, or
    supporting lemmas aimed at one are all OFF TASK -- they are not scored, they are not wanted,
    and they burn the budget you need for witnesses. You are looking for a counterexample and
    nothing else.
  * You may add auxiliary lemmas, but only ones used by the refutation, and everything must compile.
  * You have a Lean project and a shell. Use `lake env lean` to compile-check as often as you like.
    Do not stop at your first failure: try several witnesses and several strategies. Degenerate
    instances are usually the most productive -- the empty set, the zero function, a one-point
    space, n = 0, equal arguments -- as are hypotheses that are vacuous or contradictory.
  * If you exhaust your ideas, STOP and report `VERDICT: not-false` with a one-paragraph note on
    the witnesses you tried and why each failed. Do not spend remaining time on a positive proof.
    Leave the file with `af_refutation` intact and its proof as your last attempt.

When done, report your final answer as a fenced ```lean block containing the complete file you
compiled, followed by one line: `VERDICT: refuted` or `VERDICT: not-false`."""

USER_TEMPLATE = """[Context file]
The statement lives in this file. Its imports, definitions and `variable` binders are available
to you.

```lean
{src}
```

[The statement to refute: `{decl}`]
```lean
{signature}
```

[Provenance]
model prediction -- an auto-formalization of a natural-language theorem, written by a model

This statement was NOT deliberately falsified, so there is no planted weak point to aim at. It may
well be true. Your task is unchanged: look for a counterexample, and report `not-false` if you
cannot find one. Attack the places formalizations usually break -- a missing measurability,
integrability, non-emptiness, positivity or domain condition; a quantifier that is too strong; an
edge case at zero or at the empty carrier.

Write a complete Lean file that compiles against this project and proves `¬` of the statement above.
Do not attempt the positive direction."""

WORKSPACE_NOTE = """

[Your workspace]
- The file to complete is `{f}`. It already contains the exact goal, right after `{decl}`:
    theorem af_refutation :
      {goal} := by sorry
- Replace the `sorry` with a real proof. DO NOT change the statement of `af_refutation` in any
  way -- the harness re-checks it and a modified statement scores zero.
- Compile with: `cd {proj} && lake env lean $OLDPWD/{f}`
- YOU MUST WRITE YOUR FINAL ANSWER INTO `{f}` ON DISK. The harness reads the file, not your
  message. A file left unchanged scores zero."""


# ------------------------------------------------------------------ one prediction
def refute_prediction(pred: A.Target, cfg, *, stem: str = "refute") -> dict[str, Any]:
    """One refutation attempt on a compiling prediction. `cfg` is the BEq `BEqConfig`: the
    same grader model, agentic setting, per-session budget, Lean project and timeouts."""
    entry = cfg.sampler
    if entry is None:
        return {"ran": False, "refuted": False, "reason": "no grader model (`--grader-model none`)"}
    try:
        skel = build_skeleton(pred, heartbeats=cfg.heartbeats)
    except RefuteError as exc:
        return {"ran": False, "refuted": False, "reason": f"negation goal: {exc}"}
    signature = (pred.decl.sig if pred.decl else "").strip()
    user = USER_TEMPLATE.format(src=pred.lean.strip(), decl=pred.target_name,
                                signature=signature)
    agentic = bool(cfg.agentic and entry.backend in ("claude", "codex"))
    meta: dict[str, Any] = {"agentic": agentic}
    source = None
    try:
        if agentic:
            user += WORKSPACE_NOTE.format(f=REFUTE_FILE, proj=cfg.project, decl=pred.target_name,
                                          goal=skel["goal"].replace("\n", "\n      "))
            root = workspace_root() / "af_refute"
            root.mkdir(parents=True, exist_ok=True)
            scratch = Path(tempfile.mkdtemp(prefix="ref_", dir=str(root)))
            try:
                (scratch / REFUTE_FILE).write_text(skel["text"], encoding="utf-8")
                run = B.run_agent(entry, SYSTEM, user, workspace=scratch,
                                  timeout=cfg.sampler_timeout, raw_dir=cfg.raw_dir, stem=stem,
                                  lean_project=cfg.project)
                reply = run.final_text or ""
                on_disk = (scratch / REFUTE_FILE).read_text(encoding="utf-8", errors="replace")
                meta.update(n_lean_calls=run.n_lean_calls, agent_timed_out=run.timed_out)
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            completed = None
            if on_disk.strip() != skel["text"].strip():
                completed, source = on_disk, "file"
            else:
                completed = _last_lean_block(reply)
                source = "reply" if completed else None
        else:
            res = complete(entry, SYSTEM, user, timeout=cfg.sampler_timeout, raw_dir=cfg.raw_dir,
                           stem=stem)
            reply = res.text or ""
            completed = _last_lean_block(reply)
            source = "reply" if completed else None
            meta.update(n_lean_calls=0, agent_timed_out=res.timed_out)
    except Exception as exc:  # noqa: BLE001 - a dead session is "not refuted", recorded
        return {"ran": True, "refuted": False, "reason": f"grader error: {exc}",
                "negation_goal": skel["goal"], **meta}
    out: dict[str, Any] = {"ran": True, "refuted": False, "negation_goal": skel["goal"],
                           "answer_from": source,
                           "verdict_claimed": (re.findall(r"VERDICT:\s*([\w-]+)", reply) or [None])[-1],
                           **meta}
    if not completed:
        return {**out, "reason": "no attempt: file unchanged and no ```lean block in the reply"}
    text, why = assemble(skel, completed)
    if text is None:
        return {**out, "reason": why}
    lean = print_axioms(text, [skel["full_name"]], lean_project=cfg.project,
                        timeout=cfg.timeout, scratch_dir=cfg.work_dir, stem=stem + ".ax")
    if lean.timed_out:
        return {**out, "reason": "lean timeout"}
    if lean.returncode != 0:
        return {**out, "reason": "does not compile",
                "lean_error": trim_log(lean.combined, 1500)}
    ok, why = axioms_clean(lean.combined, skel["full_name"])
    if not ok:
        return {**out, "reason": why}
    return {**out, "refuted": True, "reason": why, "final_file": text}
