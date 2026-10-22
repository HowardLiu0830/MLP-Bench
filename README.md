# MLP-Bench

A benchmark for machine-learning theory proofs: 360 problems in three tasks,
**auto-formalization**, **Lean proving**, and **FL (formal-language) proof verification**.
The 120 theorems of the first two tasks come from 20 JMLR papers (Research-level) and 60
graduate-level results from lecture notes and a textbook (Graduate-level). The 120
verification problems come from ICLR submissions and their public reviews (39 flawed), 26 more
JMLR papers and the same lecture notes and textbook (34 correct, 21 with an injected error).

## Quick start

```bash
./setup.sh --lean                                          # once
running/cot/run.sh      autoform claude-sonnet-5 --limit 3 --seed 0
running/agentic/run.sh  prove    gpt-5.6-sol     --limit 3 --seed 0
evaluation/lean_proving/eval.sh runs/prove/agent__gpt-5.6-sol__<timestamp>
```

## Tasks and data

| task | data file | unit | size |
|---|---|---|---|
| Auto-formalization | `Benchmarks/auto-formalization.jsonl` | target theorem | 121 items, **120 targets** |
| Lean proving (forward) | `Benchmarks/forward-construction.jsonl` | target theorem | 80 rows, **120 targets** |
| FL verification (backward) | `Benchmarks/backward-verification.jsonl` | proof bundle | 120 items, 60 correct / 60 flawed |

Field schemas and caveats: [`Benchmarks/README.md`](Benchmarks/README.md).

## Repository layout

```
Benchmarks/      the data (three task files)          -> Benchmarks/README.md
running/         how a model produces answers         -> running/README.md
evaluation/      how answers are scored               -> evaluation/README.md
src/mlpbench/    all the logic: one module per task, core/ for backends, sandbox and Lean
lean/            the pinned Lean + Mathlib project
setup.sh         environment setup
```

The scripts in `running/` and `evaluation/` are thin wrappers around the package's CLI,
`mlpbench` (`mlpbench --help`, `mlpbench models`, `mlpbench doctor`). A run writes
`runs/<task>/<mode>__<model>__<time>/`; an evaluator scores that directory, so either step can
be re-run alone. All three tasks run in CoT (one tool-free call), agentic (a CLI agent with
Lean, in a sandbox) and open-weights (vLLM) modes.

## Setup

Requirements: [`uv`](https://docs.astral.sh/uv/); for the Lean tasks
[`elan`](https://github.com/leanprover/elan); for agentic runs `bwrap`, `/usr/bin/python3`
and the `claude` and/or `codex` CLI, logged in:

```bash
./setup.sh                # .venv from uv.lock; fetches Safe's prompt (--no-safe skips it)
./setup.sh --lean         # + Mathlib and the Lean REPL (~5.4 GB)
./setup.sh --proxy        # + LiteLLM, for open-weights agents
curl -fsSL https://claude.ai/install.sh | bash        # Claude Code -> ~/.local/bin/claude
curl -fsSL https://chatgpt.com/codex/install.sh | sh  # Codex CLI   -> ~/.local/bin/codex
claude                                                # log in once
codex login
cp .env.example .env      # API keys, only for OpenRouter / gated HF weights
```

Use the native installers, not the npm `codex` (the sandbox lets only the exec'd CLI reach the
API). The paper's runs used Claude Code 2.1.241 and Codex CLI 0.145.0. Put the Lean build on
local disk, not NFS (`MLPBENCH_LEAN_PROJECT` points elsewhere). FL verification's system
prompt carries [Safe](https://github.com/liuchengwucn/Safe)'s `prompts/formalize.txt`, which
is not vendored: `setup.sh` fetches it at the paper's commit (`MLPBENCH_SAFE_UPSTREAM` points
elsewhere).

## License

Code is licensed under MIT ([`LICENSE`](LICENSE)); the benchmark data under CC BY 4.0
([`Benchmarks/LICENSE`](Benchmarks/LICENSE)). The data contains excerpts of published papers,
attributed per source in [`ATTRIBUTION.json`](ATTRIBUTION.json).

## Acknowledgements

The FL verification pipeline follows [Safe](https://github.com/liuchengwucn/Safe) (Liu et al.,
ACL 2025): its system prompt carries Safe's own formalisation rules.
