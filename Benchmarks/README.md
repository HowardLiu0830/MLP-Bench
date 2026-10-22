# Benchmarks

One JSONL file per task, one JSON object per line.

| file | task | size |
|---|---|---|
| `forward-construction.jsonl` | Lean proving | 80 rows, 120 targets |
| `backward-verification.jsonl` | FL verification | 120 items, 60 correct / 60 flawed |
| `auto-formalization.jsonl` | auto-formalization | 121 items, 120 targets |

The 120 theorems of the first and third file come from 20 JMLR papers (`Research-level`) and 60
graduate-level results from lecture notes and a textbook (`Graduate-level`); scores are per
target. The verification items come from ICLR submissions and their public reviews, 26 more
JMLR papers, and the same lecture notes and textbook.

## `forward-construction.jsonl` (Lean proving)

Each row is a Lean file whose target theorems are left as `sorry`. The task is to prove them.

| field | meaning |
|---|---|
| `id`, `source` | row identity; `source` is `Research-level` / `Graduate-level` |
| `lean` | the Lean file given to the model (imports Mathlib only) |
| `targets` | `[{label, decls, statement, ...}]`, one entry per target theorem: its label, Lean declaration names and natural-language statement |
| `nl_proof` | natural-language proof of the row |
| `context` | the definitions and results the statements and proofs rely on: `[{id, name, statement}]` |
| `solution` | gold Lean proof. **Never show it to the model** |
| `targets[].topic_domain`, `targets[].topic_subtopic` | the target's topic. Not shown to the model |

## `backward-verification.jsonl` (FL verification)

Each item has a shared `context`, 3–5 theorems with numbered natural-language proofs, and a
target. The model decides whether the target's proof is valid; if not, it names the **first
wrong step**.

| field | meaning |
|---|---|
| `id` | item id. **Never show it**: it gives away the label |
| `context` | the shared setting printed above the theorems, one string |
| `theorems` | `[{local_id, statement, proof: {steps: [{n, claim}]}}]`, in order (`T1..Tn`) |
| `target_local_id` | the theorem under test |
| `label` | gold: `correct` / `flawed` |
| `error_step` | gold: the first wrong step of the target (null when correct) |
| `error_class6` | gold: admissible error classes, most likely first |
| `error_explanation` | gold: what is wrong at `error_step` (null when correct) |
| `topic_domain`, `topic_subtopic` | the item's topic. **Never show it**: it predicts the label |


## `auto-formalization.jsonl` (auto-formalization)

One item per target declaration of `forward-construction.jsonl`. The target's Lean statement is
masked as `<<HOLE 1>>`; the model fills it in from the natural-language statement.

| field | meaning |
|---|---|
| `item_id` | `<paper_id>::<declaration>::A-stmt` |
| `paper_id`, `target_name`, `source` | source row, declaration, `Research-level` / `Graduate-level` |
| `lean` | the original file |
| `holes` | `[{start, end, gold}]`: `lean[start:end]` is the masked statement. **`gold` is the answer; never show it** |
| `nl_statement` | the target's natural-language statement |
| `context` | the target's context, `[{name, statement}]` |
| `nl_notes` | formalization notes on how the Lean statement reads the source |
| `target_names` | all target declarations of the row |
| `topic_domain`, `topic_subtopic` | the target's topic. Not shown to the model |

## Licence

CC BY 4.0 ([`LICENSE`](LICENSE)). Third-party excerpts are attributed per source in
[`../ATTRIBUTION.json`](../ATTRIBUTION.json).
