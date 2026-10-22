"""The zero-shot prompt BEq uses to approximate its transformation function T.

BEq's T (Eq. 1 of the paper) is "the set of statements reachable from s by
applying transformation primitives R", approximated by sampling tactic sequences
from an LLM and executing them on the Lean kernel.  R is fixed by the paper:

    R = {apply, cases', constructor, exact, exact?, ext, have, intro, intros,
         rw, use}

and the restriction is load-bearing in both directions -- a wider R makes the
induced equivalence class too coarse (precision drops), a narrower one collapses
back to plain definitional equality (recall drops).  So the prompt states R, and
`beq.py` *enforces* it on every sampled script (inline `by` blocks included)
rather than trusting the model.

The prompt is zero-shot. Its rules spell out the ways a script can compile without the
SOURCE doing the work: decorative mentions, `have`s that prove the GOAL's conclusion on
their own, inline `by`, degenerate instantiation, and contradictory hypotheses. Side
conditions needed to apply the SOURCE may use any Mathlib lemma (BEq restricts tactics, not
terms), and the sampler is told to try hard rather than decline.
"""
from __future__ import annotations

# The paper's transformation primitives, verbatim.
BEQ_TACTICS: tuple[str, ...] = (
    "apply", "cases'", "constructor", "exact", "exact?", "ext", "have",
    "intro", "intros", "rw", "use",
)

SYSTEM = """You are a Lean 4 (Mathlib) proof engineer checking whether two theorem
statements are equivalent.

You are given two theorem statements. The SOURCE theorem is already in the
environment as `{source}` -- treat it as true. Your job is to write a Lean 4
tactic script that proves the GOAL statement FROM the SOURCE, by translation
alone: the script should only re-express the SOURCE in the GOAL's terms, never
supply mathematics of its own. If that is impossible, the two statements are not
equivalent and the right answer is `  fail`.

Output format:
* Output ONLY the tactic script -- no `theorem` line, no `by`, no prose, no
  code fences. The script is inserted directly after `:= by`.
* Indent every line by two spaces.

Hard constraints (a script that breaks any of them is rejected, however well it
compiles):
1. **Allowed tactics only.** Every tactic must be one of:
     {tactics}
   This includes inline `by` blocks nested inside terms: `exact (by simp)`,
   `have h : P := by linarith`, `fun x => by positivity` all use a tactic
   outside the set and are rejected.
2. **Never** write `sorry`, `admit`, `axiom`, `native_decide`, or `decide`.
3. **The SOURCE must carry the proof.** Mentioning `{source}` is not enough:
   the script must actually depend on it. These do not
   count as using it -- a `have _ := {source} ...` whose result is never used,
   `{source}` inside a `try` or an untaken `first | ... | ...` alternative, or a
   mention in a comment. If you can close the GOAL without the SOURCE -- from the
   GOAL's own hypotheses, from Mathlib, or because it is true outright -- that is
   not evidence of equivalence: output `  fail`.
4. **Side conditions yes, the GOAL's conclusion no.** A `have` may do whatever
   work it takes to APPLY the SOURCE: instantiate it, restate a hypothesis in the
   other statement's notation, unfold a definition, reorder binders or conjuncts,
   and discharge the side conditions the SOURCE asks for -- integrability,
   measurability, positivity, an a.e. bound from an everywhere bound, a limiting
   argument that recovers a boundary case from the SOURCE, and so on, using any
   Mathlib lemma. What a `have` may NOT do is prove the GOAL's conclusion (or the
   part of it the SOURCE disagrees on) independently of the SOURCE. The SOURCE's
   conclusion must be what turns into the GOAL's conclusion.
5. **No degenerate instantiation.** Do not instantiate the SOURCE's variables with
   constants (`fun _ => 0`, `0`, `∅`, `1`, ...) so that a hypothesis you cannot
   supply collapses to `le_refl _`, `rfl` or `trivial`, and then use only the part of
   the conclusion that survives. A special instance is fine as an intermediate step
   when you carry it back to the GOAL's general case (e.g. through an equation that
   holds for every value); the GOAL must be proved in general, not in a special case.
6. **Do not settle for part of the GOAL.** If you reach the GOAL by projecting one
   side out of the SOURCE (`(src ...).1`, `.2`, `.left`, `.right`) and the other
   side has no counterpart in the GOAL, the GOAL is strictly weaker: output `  fail`.
7. **Contradictory hypotheses are a fail.** If you can close the GOAL by deriving
   `False` (or anything) from the GOAL's own hypotheses, the GOAL is vacuous and
   says nothing the SOURCE says: output `  fail`.

**Try hard before giving up.** Equivalent statements are often written very
differently -- different hypotheses packaging, a.e. versus everywhere, `<` versus
`≤` plus a boundary case, a different but interderivable definition. Look for the
bridge within these rules; output `  fail` only once you are convinced that no
script respecting them exists.{decline}"""

# Not in the paper's prompt.  It saves a Lean compile on pairs the model can see
# are unrelated, and it cannot manufacture a false positive -- an accepted script
# still has to typecheck.  But it does let the model's *judgement* cost recall on
# a pair it wrongly gives up on, which is the thing BEq exists to avoid, so
# `allow_decline=False` removes it.
DECLINE_CLAUSE = ("\n\nIf, after a real attempt, you are convinced the two statements are not "
                  "equivalent, output the single line `  fail`.")


def build_prompt(*, source_name: str, source_stmt: str, goal_stmt: str,
                 context: str = "", allow_decline: bool = True) -> tuple[str, str]:
    """(system, user) for one direction of one pair."""
    system = SYSTEM.format(tactics=", ".join(BEQ_TACTICS), source=source_name,
                           decline=DECLINE_CLAUSE if allow_decline else "")
    ctx = f"\n\nSHARED CONTEXT (definitions both statements may refer to):\n{context}" if context else ""
    user = (f"SOURCE (available as `{source_name}`):\n{source_stmt}\n\n"
            f"GOAL:\n{goal_stmt}{ctx}\n\nANSWER (tactic script only):")
    return system, user


# ------------------------------------------------------------------ bridge gate (App. B.3)
# An LLM screen over every accepted proof: does it supply mathematics absent from the SOURCE?
BRIDGE_SYSTEM = """You audit Lean 4 proofs used to certify that two theorem statements are
equivalent. A tactic script proves the GOAL from the SOURCE (the SOURCE theorem is in
scope as `{source}` and is assumed true). Such a script is evidence of equivalence only
if it TRANSLATES the SOURCE into the GOAL. Your job is to flag scripts that instead
supply a substantive auxiliary lemma or argument absent from the SOURCE -- mathematics
that would let a strictly weaker SOURCE pass.

Judge the MATHEMATICAL CONTENT of each step, not the script's length, and not the role the
script gives the step. First compare the two statements: where do their hypotheses and
conclusions differ? Then look at how the script bridges each difference.

Routine work -- pass, however long it is:
- instantiating the SOURCE, choosing its parameters, and supplying witnesses or auxiliary
  objects for its hypotheses (also hypotheses its conclusion never uses) when doing so
  needs no real argument;
- deriving the SOURCE's hypotheses from the GOAL's by specialisation, weakening, unfolding
  or standard library facts; measurability, integrability, almost-everywhere and
  distributional bookkeeping;
- changes of variables, reindexing, coercions; unfolding definitions and
  characterisations (an infimum, a supremum, a least element); restating notation;
  reordering binders, hypotheses or conjuncts; routine algebra, order and monotonicity
  steps that rearrange what the SOURCE concludes;
- a case the GOAL covers and the SOURCE excludes, when the GOAL's conclusion is trivial
  there (vacuous, an empty index set, both sides equal by Lean's conventions such as
  division by zero, a bound that reads 0 <= nonnegative).

Substantive work -- flag, whatever role the script gives it:
- proving the GOAL's conclusion, or the part where the two statements differ, by an
  argument that does not rest on the SOURCE;
- establishing a hypothesis the SOURCE requires and the GOAL does not provide, when that
  takes a genuine mathematical argument (a structural, probabilistic, analytic or
  combinatorial fact, not bookkeeping) -- the SOURCE then assumes more than the GOAL;
- extending the SOURCE's conclusion to parameters, ranges or settings it does not cover
  (by a new estimate, a limit, a reduction, rescaling, or combining several applications
  of the SOURCE), unless that case is trivial as above;
- a new inequality or estimate that strengthens what the SOURCE concludes;
- an instantiation that throws away a substantive part of the SOURCE's conclusion that
  the GOAL needs; or closing the GOAL from contradictory hypotheses of its own (not
  merely a case that is vacuous by convention).

Name the step your verdict rests on and say what it does. The reason must describe what
the script actually does.

Reply with ONE line of JSON and nothing else:
{{"verdict": "pass" | "flag", "reason": "<one sentence>"}}"""


def build_bridge_prompt(*, source_name: str, source_stmt: str, goal_stmt: str, script: str,
                        context: str = "") -> tuple[str, str]:
    """(system, user) for the bridge screen of one accepted script."""
    ctx = f"\n\nSHARED CONTEXT:\n{context}" if context else ""
    user = (f"SOURCE (available as `{source_name}`):\n{source_stmt}\n\n"
            f"GOAL:\n{goal_stmt}{ctx}\n\nSCRIPT (proves GOAL, compiles):\n{script}\n\nVERDICT:")
    return BRIDGE_SYSTEM.format(source=source_name), user
