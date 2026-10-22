#!/usr/bin/env bash
# Open-weights model, CoT, against a vLLM server started with ./serve_vllm.sh.
#
#   GPUS=0 ./serve_vllm.sh Qwen/Qwen3.5-9B &          # once; note the PID it prints
#   ./run.sh <task> <hf-model-id> [options]
#   ./run.sh prove Qwen/Qwen3.5-9B --limit 3 --seed 0
#
# Agentic open-weights runs go through the codex CLI and a LiteLLM proxy instead:
#   ../agentic/run.sh <task> local:<hf-model-id>     (needs `./setup.sh --proxy` and the proxy up)
set -euo pipefail
source "$(dirname "$0")/../_common.sh"
usage_task "${1:-}"
TASK="$1"; HF="${2:?hf model id required, e.g. Qwen/Qwen3.5-9B}"; shift 2
exec "$MLPBENCH" run --task "$TASK" --mode cot --model "vllm:$HF" "$@"
