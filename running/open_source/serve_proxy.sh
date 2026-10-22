#!/usr/bin/env bash
# The LiteLLM proxy that lets the codex CLI drive a vLLM-served model (agentic open-weights).
#
#   GPUS=0 MAXLEN=131072 ./serve_vllm.sh Qwen/Qwen3.5-9B &     # 1. the model
#   ./serve_proxy.sh &                                          # 2. this proxy
#   ../agentic/run.sh prove local:Qwen/Qwen3.5-9B --limit 3 --seed 0 --max-cost 0
#
# No GPU: PROXY_CONFIG=litellm.openrouter.yaml serves `local:<vendor/model>` from OpenRouter
# instead (OPENROUTER_API_KEY in the environment), through the same client path.
#
# Needs `./setup.sh --proxy` (installs litellm into .venv). Listens on 127.0.0.1:<4000+uid%1000>,
# which is also the client's default (MLPBENCH_PROXY_URL overrides). Kill the PID it prints.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LITELLM="${LITELLM_BIN:-$ROOT/.venv/bin/litellm}"
[ -x "$LITELLM" ] || { echo "litellm not found at $LITELLM -- run ./setup.sh --proxy" >&2; exit 1; }
PORT="${PROXY_PORT:-$((4000 + $(id -u) % 1000))}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:$((8000 + $(id -u) % 1000))/v1}"
echo "LiteLLM proxy on http://127.0.0.1:$PORT -> $VLLM_BASE_URL  (pid $$)"
exec "$LITELLM" --config "${PROXY_CONFIG:-$HERE/litellm.vllm.yaml}" --host 127.0.0.1 --port "$PORT"
