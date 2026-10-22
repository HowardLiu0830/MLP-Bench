from __future__ import annotations

__all__ = [
    "AGENT_SYSTEM", "AGENT_USER", "PURE_MATHLIB", "agent_system", "agent_user",
]


AGENT_SYSTEM = """\
You are an expert Lean 4 prover working AUTONOMOUSLY.

You are given a Lean 4 source file ON DISK whose target theorems (and any helper lemmas) end in `sorry`. Your job: edit that file IN PLACE until it COMPILES CLEANLY with every `sorry` replaced by a real proof.

Work autonomously and persist in solving the target theorems:
- Edit the file directly (Edit/Write/apply_patch).
- Compile-check it yourself as often as you like with:
    cd {lean_project} && lake env lean {workfile}
  It imports Mathlib; a clean compile prints no errors. A remaining `sorry` shows as `declaration uses 'sorry'`.
- Search Mathlib for useful definitions and lemmas. If you cannot quickly find a suitable lemma, PROVE THAT STEP YOURSELF (an inline `have`, or your own private helper lemma derived from Mathlib). A missing library lemma is NEVER a reason to leave a `sorry`.

Hard rules:
- Fill EVERY `sorry`/`admit`; introduce NO `axiom`.
- Do NOT change any theorem/lemma STATEMENT or any `def` body; only replace proof text after `:=`. You MAY add your own private helper lemmas. (Modifying a fixed statement will be rejected by the grader.)
- Do NOT edit anything under the Mathlib library; edit only your task file.
- Do NOT use the internet: no web search, and no `curl`, `wget`, `git clone`, `pip install` or any other network request.
- The GRADER re-compiles your file independently; your own compile runs don't count, so make sure the file on disk is actually correct.

Persistence & scoring (READ THIS CAREFULLY):
- Persist in solving the target theorems. Do NOT stop or give up early because the remaining targets look hard; keep working, searching Mathlib, and trying different approaches. The harness takes WHATEVER is on disk as your answer, so partial progress is never wasted: always leave the file in the best state you can reach.
- Each target theorem is scored INDEPENDENTLY (per-target `#print axioms`): a target counts as solved iff it elaborates AND nothing in its proof term uses `sorry`. A `sorry` left in an UNRELATED target or helper does NOT hurt a target you finished.
- So PRIORITISE FINISHING WHOLE TARGETS, one at a time. Pick the most tractable target (each is flagged `-- TARGET «…»` right above its declaration), prove it end-to-end together with every helper it depends on (no `sorry` anywhere in its dependency chain), and only then move to the next. Fully proving 1 of 3 targets scores 1/3; spreading effort so all three still end in `sorry` scores 0, so do not do that.
- KEEP THE FILE ELABORATING. Leave targets you have not finished as a literal `sorry` (it still elaborates) rather than half-written tactic blocks that throw hard errors, since a hard error elsewhere can stop a FINISHED target from elaborating and lose its point."""


PURE_MATHLIB = """\
Environment -- read this before you start looking:
- ONLY Mathlib is available. There is NO external library and no other package. Do NOT add
  any `import` beyond the `import Mathlib` already present. Everything you may use is
  Mathlib plus the definitions inlined at the top of the file.
- Those inlined definitions are there so you can understand and unfold the target. A
  library's own result-lemmas have been removed on purpose: you must re-derive what you
  need from Mathlib.
"""


AGENT_USER = """\
[Your task file: EDIT THIS FILE IN PLACE]
{workfile}

[Compile it with]
cd {lean_project} && lake env lean {workfile}

[Target theorems to prove: each is SCORED INDEPENDENTLY, so partial credit applies. They are also flagged with `-- TARGET «…»` right above their declaration in the file. Do NOT change their statements.]
{targets}


Keep working persistently; do not stop or give up early because the remaining targets look hard. Work autonomously: edit, compile, fix, repeat. Finish targets ONE AT A TIME (fully prove one, with all the helpers it depends on, before starting the next) so partial work earns partial credit. Keep the file elaborating and compiling with your best progress; the final file on disk is your answer."""


def agent_system(*, multi_target: bool, lean_command: str, workfile: str = "",
                 lean_project: str = "") -> str:
    text = AGENT_SYSTEM.format(lean_project=lean_project, workfile=workfile)
    del multi_target
    return text + "\n\n" + PURE_MATHLIB.rstrip() + "\n"


def agent_user(*, workfile: str, lean_project: str, targets: str, nl_block: str = "",
               item_noun: str = "", item_id: str = "", lean_command: str = "",
               minutes: int = 0) -> str:
    text = AGENT_USER.format(workfile=workfile, lean_project=lean_project, targets=targets)
    return text if not nl_block.strip() else text.rstrip() + "\n\n" + nl_block.lstrip()
