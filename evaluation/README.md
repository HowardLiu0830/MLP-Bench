# Evaluation

How answers are scored. Each script reads a run directory and writes `scores.jsonl` (one row
per item) and `summary.json` into it, so scoring can be re-run without re-running the model.

| script | what it does |
|---|---|
| `auto_formalization/eval.sh <run-dir>` | typecheck, BEq against the gold, validity; two graders |
| `lean_proving/eval.sh <run-dir>` | per-target proof check with `#print axioms` |
| `fl_verification/eval.sh <run-dir>` | strict precision / recall / F1 against the gold step |

Each script is one call to `mlpbench eval --task <task> --run <run-dir>`; `mlpbench eval
--help` lists the options.

## What the scores mean

**Auto-formalization** (120 targets). The predicted statement must typecheck. Then **BEq**
(Liu et al., ICLR 2025) tries to derive the prediction from the gold and back, each direction
with a restricted tactic set; both directions -> `equivalent`, one -> `pred_stronger` /
`pred_weaker`. Every accepted derivation must pass three soundness gates: **strip** (still
works with the source deleted -> rejected), **vacuous** (still works when the source concludes
only `True` -> rejected) and **bridge** (an LLM flags a substantive lemma of its own ->
rejected). **valid** = compiles and a refutation attempt by the grader fails. Two graders,
`gpt-5.6-sol` and `claude-opus-5`, as in the paper; a pair is BEq if either proves it and
refuted if either refutes it (`--double-check none` for one grader, `--no-gates`,
`--no-refute`).

**Lean proving** (120 targets over 80 files). A target is proved if its declarations compile
and `#print axioms` shows only the standard axioms (no `sorryAx`, no `native_decide`), and
neither its statement nor the file's definitions were changed.

**FL verification** (120 items, 60 correct / 60 flawed). Strict: a flawed item is a TP only if
flagged at the gold theorem **and** step; a flag elsewhere counts as FP + FN; a correct item
flagged is an FP. A dead or unparsed answer is wrong, not "no error". Rows are re-decided from
their per-step Lean states. Headline: precision / recall / F1, plus raw accuracy;
`acc_cov` credits a correct item only if every step was checked in Lean. `--rebind` re-checks
old rows from their saved sessions without calling the model.

## Integrity counters

Every summary reports `cot_rows_with_tools` (must be 0), `agent_rows_without_lean` (an agent
row that never ran Lean), and `agent_rows_egress_suspect` (agent rows whose tool calls probed
the sandbox's API door; inspect or exclude them). A subset run is scored on that subset;
report full-benchmark numbers against the full denominators.
