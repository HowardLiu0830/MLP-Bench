from __future__ import annotations

from typing import Any

from .leansurgery import find_toplevel
from ..benchmarks.forward_construction import render_deps

BINDER_OPEN = "([{\u2983"

__all__ = ["ANSWER_FILE", "AGENT_SYSTEM", "NAG", "joint_system", "nl_for", "nl_for_joint", "n_binders",
           "BINDER_OPEN"]


ANSWER_FILE = "answer.lean"


AGENT_SYSTEM = """You are formalizing mathematics in Lean 4 (Mathlib).

`{answer}` in your working directory is a Lean file whose target theorem has had
its statement replaced by `<<HOLE 1>>`. Your job is to write that statement.

WORKFLOW -- do these in order, do not stop early:
 1. Read `{answer}`.
 2. EDIT `{answer}`, replacing `<<HOLE 1>>` with the statement.
 3. Compile:  cd {project} && lake env lean --json $OLDPWD/{answer}
 4. If it reports an error, edit and compile again. Repeat until the only
    message is the expected `declaration uses 'sorry'` warning.
 5. Only then reply DONE, and paste the FINAL compiler output verbatim in your
    reply -- the exact text `lake env lean` printed on your last run. A reply
    without it is treated as an unverified answer.

Step 3 is not optional and not something to skip when the statement "looks
right". Writing a Lean statement that elaborates is most of this task; an
uncompiled answer is usually wrong in a way you cannot see by reading it.

Your answer is read from the FILE, not from your message. A run that never edits
`{answer}` scores zero, however good the reasoning in your reply.

* You MAY and SHOULD look things up in Mathlib. Its source is readable at
  `{mathlib}` -- use `rg` there when you are unsure of a name, a notation or a
  binder form (for example whether a sum over a finset is written `in` or the
  correct alternative), and `#check` / `#print` in a scratch file for a
  signature. Getting the spelling from Mathlib is not cheating; guessing it is
  how runs fail.
* Do NOT search for this theorem, its statement, or the benchmark it comes
  from. They are not on this filesystem and turns spent looking are wasted. The
  mathematics you are formalizing is in the natural language below and nowhere
  else.
* Keep the theorem's name, keep its proof as exactly `sorry` -- you are writing a
  STATEMENT, not a proof -- and change nothing else in the file. The definitions
  above the theorem are fixed and the statement must be phrased in terms of them.
* Say exactly what the natural language says: no more general, no more specific.
  Compiling is necessary, not sufficient."""


# A Research-level row's targets are written in ONE session (`joint`), as in `prove`: every
# target's statement is a hole, so no target's gold statement is in view while writing another.
# joint_system rewrites the singular sentences of AGENT_SYSTEM (before .format) and nothing else.
_JOINT_EDITS = (
    ("`{answer}` in your working directory is a Lean file whose target theorem has had\n"
     "its statement replaced by `<<HOLE 1>>`. Your job is to write that statement.",
     "`{answer}` in your working directory is a Lean file with {n} target theorems, each of\n"
     "which has had its statement replaced by a placeholder `<<HOLE 1>>` ... `<<HOLE {n}>>`.\n"
     "Your job is to write all {n} statements."),
    (" 2. EDIT `{answer}`, replacing `<<HOLE 1>>` with the statement.",
     " 2. EDIT `{answer}`, replacing every `<<HOLE k>>` with its theorem's statement."),
    ("message is the expected `declaration uses 'sorry'` warning.",
     "messages are the expected `declaration uses 'sorry'` warnings."),
    ("Do NOT search for this theorem, its statement, or the benchmark it comes\n  from.",
     "Do NOT search for these theorems, their statements, or the benchmark they come\n  from."),
    ("Keep the theorem's name, keep its proof as exactly `sorry`",
     "Keep each theorem's name, keep each proof as exactly `sorry`"),
    ("The definitions\n  above the theorem are fixed and the statement must be phrased in terms of them.",
     "The definitions\n  above the theorems are fixed and the statements must be phrased in terms of them."),
)


def joint_system(template: str, n: int) -> str:
    """AGENT_SYSTEM (unformatted) for a session that writes `n` statements."""
    for old, new in _JOINT_EDITS:
        if old not in template:
            raise ValueError(f"AGENT_SYSTEM no longer contains {old[:50]!r}")
        template = template.replace(old, new.replace("{n}", str(n)))
    return template


NAG = """

WHAT THE SOURCE LEAVES UNSAID IS PART OF THE STATEMENT. A paper writes `E[X]`
and means the expectation exists; it writes `f` measurable without saying so;
it says "for the minimiser" and means the minimum is attained. Lean does not
grant any of it. Before you finish, go through the statement and ask, for every
object it mentions, what the ambient mathematics is silently assuming:

  * measurability / strong measurability of every function that is integrated,
    of every set that is measured, of every random variable;
  * integrability -- `Integrable f mu` -- for every integral, expectation,
    conditional expectation or variance that appears;
  * non-emptiness and finiteness -- `Nonempty`, `Finite`, `0 < n` -- of every
    index set, sample and parameter space you quantify over;
  * that measures are probability measures, that families are independent, that
    a supremum or minimiser exists.

Write each one you need as an explicit binder. A hypothesis the source assumes
without printing is still a hypothesis, and a statement missing it is a
different theorem. Do NOT add hypotheses the mathematics does not need."""


def nl_for(row: dict[str, Any]) -> str:
    """The natural-language block of the prompt: statement, context, then notes.

    The source's proof (`nl_proof`) is not shown: the task is to formalize the statement from
    the statement and its context.
    """
    stmt = (row.get("nl_statement") or "").strip()
    context = render_deps(row.get("context"))
    if context:
        stmt = f"{stmt}\n\nCONTEXT (definitions and results the statement relies on):\n{context}"
    notes = (row.get("nl_notes") or "").strip()
    if notes:
        stmt = (f"{stmt}\n\nFORMALIZATION NOTES (where the target Lean statement departs from the "
                f"text above, or how it reads it):\n{notes}")
    return stmt


def nl_for_joint(members: list[dict[str, Any]]) -> str:
    """The NL block for a joint session: each target's statement, then the union of their
    contexts, then each target's notes. A context entry that is just another member's
    statement ("<label> (another result of the same paper)") is dropped: it is already above."""
    stmts = {(m.get("nl_statement") or "").strip() for m in members}
    parts = [f"TARGET `{m['target_name']}`:\n{(m.get('nl_statement') or '').strip()}"
             for m in members]
    seen, context = set(), []
    for m in members:
        for d in m.get("context") or ():
            if isinstance(d, dict):
                if (str(d.get("name") or "").endswith("(another result of the same paper)")
                        and (d.get("statement") or "").strip() in stmts):
                    continue
                key = (d.get("name"), d.get("statement"))
            else:
                key = d
            if key not in seen:
                seen.add(key)
                context.append(d)
    out = "\n\n".join(parts)
    rendered = render_deps(context or None)
    if rendered:
        out = f"{out}\n\nCONTEXT (definitions and results the statements rely on):\n{rendered}"
    notes = [f"`{m['target_name']}`: {(m.get('nl_notes') or '').strip()}" for m in members
             if (m.get("nl_notes") or "").strip()]
    if notes:
        out = (f"{out}\n\nFORMALIZATION NOTES (where a target Lean statement departs from the "
               f"text above, or how it reads it):\n" + "\n\n".join(notes))
    return out


def n_binders(sig: str) -> int:
    cut = find_toplevel(sig, ":", skip="=")
    scan = sig[:cut] if cut > 0 else sig
    depth = n = 0
    for ch in scan:
        if ch in BINDER_OPEN:
            if depth == 0:
                n += 1
            depth += 1
        elif ch in ")]}\u2984":
            depth = max(0, depth - 1)
    return n
