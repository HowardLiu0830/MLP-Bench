# Running

How a model produces answers. A run does not score; [`../evaluation/`](../evaluation/) does.

| script | what it does |
|---|---|
| `cot/run.sh <task> <model>` | one tool-free call per item |
| `agentic/run.sh <task> <model>` | a `claude` / `codex` CLI agent with Lean, in a sandbox |
| `open_source/serve_vllm.sh <hf-id>` | starts a local vLLM server (`GPUS=`, `VLLM_BIN=`); prints its PID |
| `open_source/run.sh <task> <hf-id>` | CoT against that server (= `--model vllm:<hf-id>`) |
| `open_source/serve_proxy.sh` | LiteLLM proxy in front of vLLM (or OpenRouter), for agentic open-weights runs |

`<task>` is `autoform`, `prove` or `verify`; `mlpbench models` lists the model aliases. Each
script is one call to `mlpbench run --task <task> --mode {cot,agent} --model <model>`.

## Options

| option | meaning |
|---|---|
| `--limit N --seed S` | a random, stratified subset of N items (always together: the files are ordered) |
| `--source Research-level\|Graduate-level` | one half of the benchmark (aliases `jmlr`, `lecture`) |
| `--ids`, `--workers`, `--effort`, `--resume` | item selection, parallelism, reasoning effort, continue a run |
| `--timeout` | agent wall clock per item; default 3600 / 2400 / 3600 s for autoform / prove / verify; CoT has no limit |
| `--sibling-targets joint\|hidden\|shown` | autoform, Research-level rows (3-4 targets per paper): `joint` (agent default) one session per row with every target blanked; `hidden` (CoT default) one call per target, the others removed; `shown` keeps the others' gold statements in view |
| `--nl` / `--no-nl` | prove: the agent gets the Lean file and the targets' NL statements; `--nl` adds the NL proof, `--no-nl` gives the Lean alone |
| `--fl-task fl_refute\|prove` | verify: FL Refute (default) or FL Prove (paper App. F.3) |
| `--max-actions N`, `--max-cost USD` | prove agent: stop after N actions (default 150) or USD of list-price cost (default 4) and grade what is on disk; the agent is not told |

## What each mode does per task

| task | CoT | agentic |
|---|---|---|
| `autoform` | fill the hole: `{"fills": {"1": <statement>}}` | write the statement into `answer.lean` and compile it |
| `prove` | the whole completed file in one reply | complete `<id>.lean`; the file on disk when it stops is graded |
| `verify` | formalise every step in one reply, with no checker (the paper's `direct`) | formalise every step and try to prove or refute it, with a Lean checker |

In `verify`, the harness re-checks every returned statement, proof and refutation in its own
Lean REPL in both modes; the model's account of Lean is never read.

## Open-weights models

```bash
GPUS=0 VLLM_BIN=/path/to/vllm open_source/serve_vllm.sh Qwen/Qwen3.5-9B &
open_source/run.sh <task> Qwen/Qwen3.5-9B                  # CoT
open_source/serve_proxy.sh &                               # needs ./setup.sh --proxy
agentic/run.sh <task> local:Qwen/Qwen3.5-9B                # agentic: codex CLI via the proxy
```

An agentic open-weights run is the codex CLI (the default agent framework; only Claude models
run in Claude Code) talking to the LiteLLM proxy's `/v1/responses`, which forwards to vLLM.
`prove` needs `--max-cost 0` for these models, which have no list price; the action cap still
applies. Without a GPU, `PROXY_CONFIG=litellm.openrouter.yaml open_source/serve_proxy.sh`
serves `local:<vendor/model>` (e.g. `local:deepseek/deepseek-v4-flash`) from OpenRouter
through the same path.

vLLM is not a dependency: install it separately. OpenRouter instead of a local server:
`--model openrouter:<vendor/model>` with `OPENROUTER_API_KEY`. Answer budget:
`MLPBENCH_MAX_TOKENS` (default 32k).

## The run directory

```
runs/<task>/<mode>__<model>__<timestamp>/
  config.json     what was run: task, mode, model, CLI versions, item ids, options
  answers.jsonl   one row per item, appended as items finish (--resume skips them)
  sessions/       per item: the exact prompt, the raw transcript, a meta record
```

## Sandbox

Agents run in a bwrap jail that sees only the system, the CLI, Lean and a scratch workspace --
never this repo or the gold. The jail is offline: only the CLI's own model-API calls leave
it, through an allowlisting gateway; every shell the agent starts runs without even that. CoT
calls run in the same jail with every tool disabled. `mlpbench doctor` checks the jail;
details in `src/mlpbench/core/sandbox/`.
