#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path

from ..benchmarks.fl_verification import check_rendered
from ..core.paths import safe_upstream
from .rubric_text import ERROR_TAXONOMY
from .llm import complete_agentic
from .common import lean_repl
from .common.lean_repl import LeanRepl

_HERE = Path(__file__).resolve().parent
CHECKER = _HERE / "common" / "lean_check_cli.py"

UPSTREAM = safe_upstream()

STATES = ("No Verification Required", "Failed Formalization",
          "Success of Proof", "Failure of Proof", "Refuted", "Entailment Refuted")
NVR, MALFORMED, PROVED, UNPROVED, REFUTED, ENT_REFUTED = STATES


_HDR_STEP = re.compile(r"\[\[\[\s*theorem\s*=\s*(?P<t>[^\s\]]+)\s+step\s*=\s*(?P<n>\d+)\s*\]\]\]"
                       r"(?P<body>.*?)(?=\[\[\[|\Z)", re.DOTALL)
_FENCE = re.compile(r"```(?:lean4?|Lean4?)?\s*\n(.*?)```", re.DOTALL)
_NEGATIVE = re.compile(r"(?i)\A\s*(false|none|n/?a|skip|no_claim)\s*\Z")


def _block_body(body: str) -> str | None:
    m = _FENCE.search(body)
    code = textwrap.dedent(m.group(1) if m else body).strip()
    return None if not code or _NEGATIVE.match(code) else code


_SECTION = re.compile(
    r"(?im)^\s*(STATEMENT|PROOF|REFUTATION|ENTAILMENT|ENTAILMENT_REFUTATION)"
    r"(?:\s*\(premises:\s*(?P<prem>.*)\))?\s*:?\s*$")

_CANONICAL = re.compile(
    r"\A\s*(?:theorem|lemma|example)\s+\w*\s*:\s*(?P<prop>.+?)\s*:=\s*(?:by\s+)?sorry\s*\Z",
    re.DOTALL)


_FN_BINDER = re.compile(r"(?:forall|∀)\s*\(?\s*\w+\s*:\s*[^,)]*(?:->|→)")


def quantifies_a_function(prop: str) -> bool:
    return bool(_FN_BINDER.search(prop or ""))


def _indent(tactics: str, pad: str = "  ") -> str:
    return "\n".join(pad + ln if ln.strip() else ln for ln in (tactics or "").splitlines())


_LEADING_BY = re.compile(r"\A\s*by\b[ \t]*\n?")


def _tactics(block: str, pad: str = "  ") -> str:
    return _indent(_LEADING_BY.sub("", block or ""), pad)


def negation_of(statement: str) -> str | None:
    m = _CANONICAL.match((statement or "").strip())
    if not m:
        return None
    prop = m.group("prop").strip()
    return f"theorem refutation : Not (\n{prop}\n) := by\n"


def parse_session_batch(raw: str | None) -> dict:
    out = {}
    for m in _HDR_STEP.finditer(raw or ""):
        body = m.group("body")
        parts = _SECTION.split(body)
        got = {"statement": None, "proof": None, "refutation": None,
               "entailment": None, "entailment_refutation": None, "premises": None}
        if len(parts) > 1:
            for label, prem, chunk in zip(parts[1::3], parts[2::3], parts[3::3]):
                got[label.lower()] = _block_body(chunk)
                if prem and label.upper() == "ENTAILMENT":
                    got["premises"] = [x.strip() for x in prem.split(",") if x.strip()]
        else:
            got["statement"] = _block_body(body)
        if got["statement"] or got["proof"] or got["entailment"]:
            key = (str(m.group("t")).strip().strip('"\''), int(m.group("n")))
            got["statement"] = got["statement"] or got["proof"]
            out[key] = got
    return out


_T_ID = re.compile(r"T(\d+)")


def bind_theorem_ids(emitted: dict, item: dict) -> tuple[dict, list[str]]:
    """The contract asks for `theorem=<local_id>` and the bundle heads each theorem `# Theorem
    T1`, but agents sometimes write the bare number, `theorem=1`. Matched literally, such a
    session lost every step and read as "no error found" (opus: 14-28 items per run). When every
    id of the item is `T<n>`, a bare `n` is bound to `T<n>`; a label the item uses verbatim
    always wins over a rebound one. Returns the bound batch and the rebound labels ("1/3")."""
    ids = {str(t["local_id"]) for t in item["theorems"]}
    if not ids or not all(_T_ID.fullmatch(i) for i in ids):
        return emitted, []
    out = {k: v for k, v in emitted.items() if k[0] in ids}
    rebound = []
    for (t, n), got in emitted.items():
        if t in ids:
            continue
        key = (f"T{int(t)}", n) if t.isdigit() else None
        if key and key[0] in ids and key not in out:
            out[key] = got
            rebound.append(f"{t}/{n}")
        else:
            out[(t, n)] = got           # stays unmatched, and is reported as such
    return out, rebound


def require_upstream() -> Path:
    """Safe's own `formalize.txt`, as in the paper. Safe ships no licence, so
    it is not vendored here; a missing clone is an error, not a quiet switch to `adapted`."""
    f = UPSTREAM / "prompts" / "formalize.txt"
    if not f.is_file():
        raise SystemExit(
            f"FL verification needs Safe's upstream prompt at {f}. Run ./setup.sh, which "
            f"fetches it, or point MLPBENCH_SAFE_UPSTREAM at a clone of "
            f"https://github.com/liuchengwucn/Safe")
    return f


def rules_head() -> str:
    head = require_upstream().read_text(encoding="utf-8")
    i = head.find("<question>")
    if i != -1:
        j = head.rfind("\n#", 0, i)
        head = head[:j if j != -1 else i].rstrip() + "\n"
    return head


def bundle_text(item: dict) -> str:
    parts = [f"# Question\n{(item.get('context') or '').strip()}"]
    for t in item["theorems"]:
        steps = "\n".join(f"Step {s['n']}: {s['claim']}" for s in t["proof"]["steps"])
        parts.append(f"# Theorem {t['local_id']}\n{t['statement']}\n\n"
                     f"## Steps of its proof\n{steps}")
    text = "\n\n".join(parts)
    check_rendered(str(item.get("id") or ""), text)
    return text


_LEAN = None
_LEAN_LOCK = __import__("threading").Lock()


def lean() -> "LeanRepl":
    global _LEAN
    with _LEAN_LOCK:
        if _LEAN is None:
            _LEAN = LeanRepl()
            print(f"[lean] Mathlib environment ready in {_LEAN.load_seconds}s", flush=True)
    return _LEAN


def preflight_lean(boot: bool) -> None:
    if not boot:
        print("[lean] --no-lean: NO STATEMENT IS CHECKED. This run is plumbing only and its "
              "rows are not a result.", flush=True)
        return
    if not lean_repl.REPL_BIN.is_file():
        sys.exit(
            f"[fatal] Lean REPL binary not found.\n{lean_repl.describe()}\n"
            f"        Build it once per node: bash code/lean/bootstrap_lean.sh\n"
            f"        (it builds the repl and stages Mathlib on node-local disk)\n"
            f"        Point it elsewhere with LEAN_REPL_BIN / MLPBENCH_LEAN_PROJECT.\n"
            f"        Or pass --no-lean, which marks the run as plumbing-only.\n"
            f"        Without this, the run completes, scores about chance, and means "
            f"nothing.")
    if boot:
        try:
            lean()
        except Exception as exc:
            sys.exit(f"[fatal] Lean REPL found but would not start: {exc}")
        # A REPL with no Mathlib project still starts; then nothing elaborates and every step
        # silently becomes Failed Formalization.
        if not elaborates("theorem mathlib_probe : (2 : ℝ) + 2 = 4 := by sorry"):
            sys.exit(f"[fatal] the Lean REPL started but Mathlib did not load.\n"
                     f"{lean_repl.describe()}\n"
                     f"        Run ./setup.sh --lean, or point MLPBENCH_LEAN_PROJECT at a built "
                     f"Mathlib project.")


def elaborates(statement: str) -> bool:
    try:
        ok, _ = lean().check_statement(statement)
        return ok
    except Exception:
        return False


DECISIONS = {
    "unproved": (UNPROVED, REFUTED),
    "refuted":  (REFUTED,),
}


def _first(per_theorem, states) -> tuple:
    return next(((th["local_id"], st["n"]) for th in per_theorem
                 for st in sorted(th["steps"], key=lambda z: z["n"])
                 if st["state"] in states), (None, None))


def aggregate(item, per_theorem, extra_notes=(),
              decision="refuted", task="fl_refute") -> dict:
    flat = [st for th in per_theorem for st in th["steps"]]
    counts = {name: sum(1 for st in flat if st["state"] == name) for name in STATES}
    supported = ("unproved",) if task == "prove" else tuple(DECISIONS)
    decisions = {}
    for name, states in DECISIONS.items():
        if name not in supported:
            decisions[name] = None
            continue
        lid, step = _first(per_theorem, states)
        decisions[name] = {"has_error": lid is not None, "target_theorem": lid,
                           "wrong_step": step,
                           "verdict": "incorrect" if lid else "correct"}
    chosen = decisions[decision]
    lid, step = chosen["target_theorem"], chosen["wrong_step"]
    return {
        "id": item["id"],
        "has_error": chosen["has_error"],
        "verdict": chosen["verdict"],
        "target_theorem": lid,
        "wrong_step": step,
        "error_class": "none",
        "reason": (("first step whose formal statement Lean showed to be FALSE"
                    if decision == "refuted" else
                    "first step whose formal statement elaborated but could not be proved")
                   if lid else
                   ("no step was refuted" if decision == "refuted"
                    else "no step reached 'Failure of Proof'")),
        "task": task,
        "decision": decision,
        "decisions": decisions,
        "decisions_undefined": [k for k, v in decisions.items() if v is None],
        "decisions_undefined_reason":
            f"task={task}: the prompt did not request the artefact those rules count",
        "state_counts": counts,
        "n_steps": len(flat),
        "coverage": (len(flat) - counts[NVR]) / len(flat) if flat else 0.0,
        "per_theorem": per_theorem,
        "formalize_prompt": "upstream",
        "adapter_notes": [
            "upstream divide_steps bypassed: authored step numbering is what gold indexes (R1)",
            "per theorem, first-in-listed-order (R1)",
            "Safe has no 'refuted' state; the decision rule here is OURS, not theirs",
            "error_class unobtainable; STRICT is 0 without a separate pass",
            "system prompt = the paper's shared F.3 prompt (fl_refute and prove get the same "
            "text); user prompt = F.3's, minus the REFUTATION block for prove",
            "formalize_prompt=upstream (Safe's own prompts/formalize.txt)",
            *extra_notes,
        ],
    }


# The R2 exemption list of the paper's appendix (Table `tab:proof-verification-exemptions`).
FL_EXEMPTIONS = """These are NOT errors and must not be reported, provided the stated condition holds:
- a **loose bound**: the bound remains valid;
- **notation, or source compression**: the intended mathematics is unambiguous or valid under a
  standard convention; brevity alone is not an error;
- an **omitted boundary case**: the case is vacuous, trivially true, or excluded by the hypotheses;
- **omitted regularity details**: routine conventions;
- **implicit justification**: "standard" or "clearly" alone is not an error; flag only when the
  intended result can be identified and shown not to apply;
- an **unrepeated assumption**: the assumption appears elsewhere in the supplied context."""


TASKS = ("fl_refute", "prove")
TASK_DECISION = {"fl_refute": "refuted", "prove": "unproved"}


# ------------------------------------------------------------------ the paper's prompt (F.3)
# `fl_refute` and `prove` render Appendix F.3 of the paper, which is the spec:
# one SHARED system prompt with no refutation-specific content, and a user prompt that for FL
# Refute holds the PROOF and REFUTATION contracts, the checker and the bundle; FL Prove is the
# same user prompt with the REFUTATION block and its contract removed. The templates below are F.3 with its
# `{placeholders}`; `F3_FILL` is what fills them. The paper's box
# wraps lines visually and drops blank lines, so only the words are the contract.
F3_SYSTEM = """# What counts as the error

Judge the proof at the level of its STEPS. Report a step as flawed if it is invalid **as
written**: a false (in)equality, a wrong constant or index, an inference that does not follow, a
named result applied where its hypotheses fail, or an illegal quantifier move. That the theorem's
stated conclusion happens to survive the slip is **not a defence**: a proof that reaches a true
conclusion through a false line is still a flawed proof, and a later step silently using the
correct value does not repair the wrong one. Trace any defect to its earliest origin step.

{exemptions}

# Error-type checklist

Judge for yourself whether each theorem's proof is correct; do NOT assume any particular one is
flawed, and do NOT assume any error exists at all. Test every step of every theorem against EACH
failure mode below. Work forward without skipping: process theorems in their listed order and
steps in increasing local step-number order. Complete the ENTIRE bundle, even after verifying a
fault. The error selected for reporting is the EARLIEST qualifying step across the whole bundle
in that order, not one error per theorem; the evaluator selects it from your artefacts after
checking them. For the structural types (assumption / quantifier), the error lives at the step
that first ASSUMES or asserts the bad thing, NOT the later step where it visibly breaks; always
trace any defect back to its origin.

{error_type_checklist}

# The bundle

Several theorem statements and proofs from the SAME paper are shown below as [T1], [T2], ... .
They MAY contain a genuine mathematical error -- or they may ALL be correct. You are not told
which. Step numbers are local to each theorem's own proof, never a global count across the
bundle.

{upstream_formalization_rules}

{method}# What to return

One block per step, in order: theorems in the order they are listed, steps by their number.
Do not skip a step because it looks obviously fine -- that judgement is exactly what this task
takes away from you.

{no_stop}

    [[[ theorem=<local_id> step=<n> ]]]
    STATEMENT
    ```lean4
    theorem test : <the whole proposition> := by sorry
    ```
    PROOF
    ```lean4
    theorem test : <the same proposition> := by
      <tactics that close it>
    ```

{no_triage_rationale}

A step whose content is a definition or a naming -- "let m = D(F, eps, d)", "write phi(x) for
(x-a)_+" -- still has a formal core: the defining equation. Write it.

{definition_step_example}

A step has nothing to formalise only if it states no equation, inequality, bound, or assertion
about anything this proof has introduced -- a pointer outside this proof, or an announcement of
what comes next. Difficulty is not a reason. Emit it as

    [[[ theorem=<local_id> step=<n> ]]]
    NO_CLAIM

and expect to need that rarely; more than one or two in a proof is triage. Every NO_CLAIM is
counted and reported.

## STATEMENT -- required, and its SHAPE is required

Formalize the step faithfully, so that the proposition carries the full mathematical content of
the step as written, not a weakened, trivial, or tautological version of it.

Write the step as ONE Lean 4 theorem, proof left as `sorry`, in exactly this form:

    theorem test : <proposition> := by sorry

If the proof fixes an object, bind it once and constrain it -- do not universally quantify it
and then require agreement.

Everything the step quantifies over goes INSIDE the proposition, as leading binders:

{binder_example}

Do NOT put binders to the left of the colon (`theorem test (B : Real) ... : ...`). The shape
matters because the harness lifts the proposition out of it verbatim when it needs to negate
it.

Do not close any goal with `plausible`, `native_decide`, `admit` or anything else that does
not build a real proof term: every proof and refutation is re-checked with `#print axioms`, and
anything resting on `sorryAx` or a non-standard axiom is discarded. (`by plausible` on a false
statement is silently accepted by a naive check -- it is not accepted here.)

Nothing between blocks -- no prose, no summary, no verdict. Do NOT say which step is wrong."""

# The Method section F.3 shows in the system prompt.
F3_METHOD = """# Method: step-level formalisation with Lean verification

You verify proofs by translating steps into Lean 4 and letting LEAN decide. Work however you
like inside this session: draft a statement, check it, fix it, try a proof, check again.

WHAT YOU RETURN IS ARTEFACTS, NOT A VERDICT. Every statement and proof you hand back is
re-checked against a fresh Lean environment before anything is scored, and your account of
what the checker said is not read. So there is nothing to be gained by reporting a proof you
did not get past the checker, and nothing lost by returning a statement you could not prove.

"""

F3_NO_STOP = """Use the checker available in this session to validate your artefacts. Do not stop after
finding an error, even one accepted by the checker. Work to the end of every theorem in the
bundle and return one block per step. After re-checking, the evaluator selects the EARLIEST
qualifying error across the entire bundle, ordered first by the theorems' listed order and then
by local step number."""

# CoT (`direct` in the paper's ablation): one generation, no checker. Only the two system-prompt
# pieces that describe the checker change; the user prompt drops F3_USER_CHECKER. Everything
# else is F.3 verbatim.
F3_METHOD_COT = """# Method: step-level formalisation with Lean verification

You verify proofs by translating steps into Lean 4 and letting LEAN decide. This is a single
generation: you have no checker and no second turn, so write each statement and proof as you
would submit it.

WHAT YOU RETURN IS ARTEFACTS, NOT A VERDICT. Every statement and proof you hand back is
re-checked against a fresh Lean environment before anything is scored, and your account of
what Lean would say is not read. So there is nothing to be gained by reporting a proof you
are not confident of, and nothing lost by returning a statement you could not prove.

"""

F3_NO_STOP_COT = """Do not stop when you think you have found something: this is a single generation with no
checker, so you cannot know that you have. Work to the end of every theorem in the bundle and
return one block per step. After re-checking, the evaluator selects the EARLIEST qualifying
error across the entire bundle, ordered first by the theorems' listed order and then by local
step number."""

F3_USER_PROOF = """## PROOF -- optional

The same theorem with a real proof. Omit this section if you could not prove it: an unproved
statement is a legitimate outcome and a fabricated proof is not. Never write `sorry` here."""

# The REFUTATION block and its contract: what FL Prove drops.
F3_USER_REFUTATION = """## REFUTATION -- optional, and this is how a step gets reported as WRONG

Failing to prove a statement is not disproof. It may be false, or it may be true and merely out
of reach, or the paper may have granted something the formalisation could not carry. A step is
reported as an error ONLY when Lean accepts a proof that its statement is FALSE.

You do not write that goal. The harness takes the proposition from your own STATEMENT, verbatim,
and assembles

    theorem refutation : Not ( <your proposition> ) := by
      <your REFUTATION tactics>

so the thing being refuted is exactly the thing you formalised, and there is no way to refute an
easier proposition instead. Your tactics start on the goal `Not (<proposition>)`.

Add the refutation as a third block of the step, after PROOF:

    REFUTATION
    ```lean4
    <tactics starting on `Not (<proposition>)`, usually beginning with `intro h` and then deriving `False`; include `intro h` yourself if used>
    ```

For a `forall`-shaped proposition the refutation is a counterexample: introduce the hypothesis,
instantiate it at concrete values, and derive a contradiction, for example:

{counterexample_example}"""

F3_USER_CHECKER = """You have a checker in the working directory:

{checker_usage}"""

F3_FILL = {
    "exemptions": FL_EXEMPTIONS,
    # The nine failure modes, without the checklist's own preamble.
    "error_type_checklist": ERROR_TAXONOMY[
        ERROR_TAXONOMY.index("1. invalid_inequality_or_equation"):].strip(),
    "no_triage_rationale": """This is the whole difference between this task and the one that lets you triage. Measured on
this benchmark, when a model was free to choose which steps to formalise it formalised 14% of
them, and the step carrying the actual error was among the ones it skipped 44% of the time. Its
judgement about where to look was wrong more often than Lean was wrong about what it looked at.
So the judgement is not yours to make here: formalise all of them and let Lean sort them out.""",
    "definition_step_example": """    theorem test : forall (a : Real) (phi : Real -> Real),
        (forall x : Real, phi x = max (x - a) 0) -> phi a = 0 := by sorry""",
    "binder_example": """    theorem test : forall (B p eps : Real), 0 < B -> 0 < eps -> eps <= B * Real.sqrt p ->
        2*B*Real.sqrt p/eps + 1 <= 3*B*Real.sqrt p/eps := by sorry""",
    "counterexample_example": """    intro h
    have := h 1 1 1 (by norm_num) (by norm_num) (by norm_num)
    norm_num at this""",
    "checker_usage": """    write a JSON array of Lean 4 source strings to stmts.json, then run
        python3 lean_check_cli.py stmts.json
    It imports Mathlib once (about ten seconds) and then checks every entry, printing one JSON
    object per line: {"i": <index>, "ok": <bool>, "sorries": <int>, "errors": [...]}.
    `ok` means it elaborated with no errors and no `sorry` -- a COMPLETE proof. A statement
    left as `:= by sorry` correctly reports ok=false, sorries=1; that is not an error.
    BATCH your statements -- one invocation with twenty entries costs far less than twenty
    invocations, because each invocation reloads Mathlib.""",
}
assert F3_FILL["error_type_checklist"].startswith("1. ") and "\n9. " in F3_FILL["error_type_checklist"]


def _fill(template: str, values: dict[str, str]) -> str:
    for k, v in values.items():
        template = template.replace("{" + k + "}", v)
    return template


MODES = ("agent", "cot")


def f3_system_prompt(mode: str = "agent") -> str:
    """F.3's shared system prompt: one text for FL Refute and FL Prove, nothing about refuting
    beyond the shared `#print axioms` paragraph that F.3 itself puts here. `cot` swaps the two
    paragraphs that describe the checker for their single-generation wording."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    cot = mode == "cot"
    return _fill(F3_SYSTEM, {**F3_FILL,
                             "upstream_formalization_rules": rules_head().strip(),
                             "method": F3_METHOD_COT if cot else F3_METHOD,
                             "no_stop": F3_NO_STOP_COT if cot else F3_NO_STOP})


def f3_user_prompt(item: dict, task: str, mode: str = "agent") -> str:
    """F.3's FL Refute user prompt; for `prove`, the REFUTATION block and its contract removed;
    for `cot`, the checker paragraph removed. The bundle comes last, after the contracts, as in
    F.3."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, not {task!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    parts = [F3_USER_PROOF]
    if task == "fl_refute":
        parts.append(_fill(F3_USER_REFUTATION, F3_FILL))
    if mode == "agent":
        parts.append(_fill(F3_USER_CHECKER, F3_FILL))
    return "\n\n".join(parts) + "\n\n" + bundle_text(item)


for _piece in (F3_METHOD_COT, F3_NO_STOP_COT):   # a CoT prompt must not point at a tool
    assert "checker in the working directory" not in _piece and "lean_check_cli" not in _piece


_AGENT_TIMEOUT = int(os.environ.get("FL_AGENT_TIMEOUT", "3600"))


def run_item_session(item, backend="codex", model=None, effort=None, do_lean=True,
                     session_dir=None, timeout=_AGENT_TIMEOUT,
                     task="fl_refute", mode="agent", entry=None, replay=None) -> dict:
    """ONE model session for the whole bundle; then every returned statement, proof and
    refutation is re-checked here against the harness's own REPL.

    agent  a tool-enabled, multi-turn session with the Lean checker in its workspace.
    cot    one tool-free generation (`core.chat.complete`, needs `entry`): the model cannot
           check anything before answering. Everything after the call is identical.

    `replay=(rc, final_text)` skips the model and re-checks a saved session's output."""
    sp = f"{session_dir.rstrip('/')}/{item['id']}" if session_dir else None
    decision = TASK_DECISION[task]
    cot_meta: dict = {}
    if replay is not None:
        rc, raw = replay
    elif mode == "cot":
        from ..core.chat import complete

        if entry is None:
            raise ValueError("mode='cot' needs the resolved model entry")
        res = complete(entry, f3_system_prompt("cot"), f3_user_prompt(item, task, "cot"),
                       timeout=timeout, raw_dir=Path(session_dir) if session_dir else None,
                       stem=str(item["id"]))
        rc = res.returncode if res.returncode is not None else -1
        raw = res.text
        cot_meta = {"n_tool_calls": len(res.tool_calls), "tokens": res.usage.as_dict()}
    elif mode == "agent":
        seed = {"lean_check_cli.py": CHECKER.read_text(encoding="utf-8"),
                "lean_repl.py": (CHECKER.parent / "lean_repl.py").read_text(encoding="utf-8")}
        rc, raw = complete_agentic(f3_user_prompt(item, task), system=f3_system_prompt(),
                                   backend=backend, model=model, effort=effort, timeout=timeout,
                                   seed_files=seed,
                                   session_path=(f"{sp}.session.jsonl" if sp else None))
    else:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    emitted, rebound = bind_theorem_ids(parse_session_batch(raw), item) if rc == 0 else ({}, [])

    per_theorem, n_proofs_offered, n_proofs_accepted = [], 0, 0
    n_refutations_offered = n_refutations_accepted = 0
    for t in item["theorems"]:
        steps = []
        for sdef in t["proof"]["steps"]:
            got = emitted.get((str(t["local_id"]), int(sdef["n"]))) or {}
            rec = {"n": sdef["n"], "state": NVR, "formal_statement": None}

            proved = refuted = False
            stmt = got.get("statement")
            if stmt:
                rec["formal_statement"] = stmt
                if got.get("proof"):
                    n_proofs_offered += 1
            elaborated = bool(stmt) and do_lean and elaborates(stmt)
            if elaborated:
                if got.get("proof") and "sorry" not in got["proof"]:
                    try:
                        proved, _, note = lean().check_certified(got["proof"])
                        if note:
                            rec["proof_rejected_because"] = note
                    except Exception:
                        proved = False
                neg = negation_of(stmt)
                if got.get("refutation") and neg and "sorry" not in got["refutation"]:
                    n_refutations_offered += 1
                    rec["refutation_goal"] = neg
                    try:
                        refuted, _, note = lean().check_certified(neg + _tactics(got["refutation"]))
                        if note:
                            rec["refutation_rejected_because"] = note
                    except Exception:
                        refuted = False
                elif got.get("refutation") and not neg:
                    rec["refutation_note"] = ("statement is not in the canonical "
                                              "`theorem name : prop := by sorry` shape, so the "
                                              "negation could not be built from it")

            if refuted:
                rec["refutes_a_quantified_function"] = quantifies_a_function(stmt)
            if proved and refuted:
                rec["state"] = MALFORMED
                rec["note"] = "both the proof AND the negation checked; formalisation is not " \
                              "self-consistent"
            elif proved:
                rec["state"] = PROVED
                rec["proof"] = got["proof"]
                n_proofs_accepted += 1
            elif refuted:
                rec["state"] = REFUTED
                rec["refutation"] = got["refutation"]
                n_refutations_accepted += 1
            elif stmt and not do_lean:
                rec["state"] = MALFORMED
                rec["note"] = "lean disabled; cannot distinguish states 2-5"
            elif stmt and not elaborated:
                rec["state"] = MALFORMED
            elif stmt:
                rec["state"] = UNPROVED
                if got.get("proof"):
                    rec["rejected_proof"] = got["proof"]
                if got.get("refutation"):
                    rec["rejected_refutation"] = got["refutation"]
            steps.append(rec)
        per_theorem.append({"local_id": t["local_id"], "steps": steps})

    known = {(str(t["local_id"]), int(x["n"]))
             for t in item["theorems"] for x in t["proof"]["steps"]}
    unmatched = sorted(f"{a}/{b}" for a, b in emitted if (a, b) not in known)
    row = aggregate(item, per_theorem, decision=decision, task=task, extra_notes=[
        ("ONE tool-enabled, multi-turn session for the whole bundle; every statement checked "
         "alone against Mathlib") if mode == "agent" else
        ("ONE tool-free generation for the whole bundle (no checker, no second turn); every "
         "statement checked alone against Mathlib"),
        "the adapter re-elaborated every returned statement and re-checked every returned "
        "proof against its own REPL; the model's account of Lean is not read",
        f"proofs offered by the model: {n_proofs_offered}, accepted by Lean: "
        f"{n_proofs_accepted}",
        f"refutations offered: {n_refutations_offered}, accepted: {n_refutations_accepted}",
        "the negated goal is built by the adapter from the model's own proposition, so a "
        "refutation cannot be of a different proposition",
    ])
    row.update(cot_meta)
    row.update({"mode": mode, "parse_ok": bool(emitted), "session_rc": rc,
                "n_formalized": len(emitted), "n_unmatched": len(unmatched),
                "unmatched": unmatched, "rebound": rebound,
                "proofs_offered": n_proofs_offered,
                "proofs_accepted": n_proofs_accepted,
                "refutations_offered": n_refutations_offered,
                "refutations_accepted": n_refutations_accepted})
    return row
