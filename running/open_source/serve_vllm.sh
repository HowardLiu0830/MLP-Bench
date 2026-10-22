#!/usr/bin/env bash
# Serve an open-weights model with vLLM's OpenAI-compatible API, for the `vllm:<hf-id>` models.
#
#   GPUS=9 ./serve_vllm.sh Qwen/Qwen3.5-9B          # 1 GPU, foreground
#   GPUS=4,5 TP=2 ./serve_vllm.sh google/gemma-4-26B-A4B-it
#
# Then run with  --model vllm:Qwen/Qwen3.5-9B  (the URL defaults to the port below; set
# MLPBENCH_VLLM_URL if you serve elsewhere). Stop it with Ctrl-C, or `kill` the PID it prints:
# a server started with nohup/setsid holds its GPU until someone kills it.
#
# Environment:
#   VLLM_BIN    the vllm executable (default: `vllm` on PATH). vLLM is not a dependency of this
#               repo -- install it in its own environment, it pins its own torch/CUDA.
#   GPUS TP PORT MAXLEN GPU_UTIL TOOL_PARSER REASONING_PARSER EXTRA_ARGS   (see below)
set -euo pipefail
MODEL="${1:?usage: GPUS=<ids> ./serve_vllm.sh <hf-model-id>}"
VLLM_BIN="${VLLM_BIN:-vllm}"
command -v "$VLLM_BIN" >/dev/null 2>&1 || [ -x "$VLLM_BIN" ] || {
  echo "vllm not found (VLLM_BIN=$VLLM_BIN). Install vLLM in its own venv and point VLLM_BIN at it." >&2
  exit 1; }

GPUS="${GPUS:?set GPUS, e.g. GPUS=0}"
TP="${TP:-1}"
# Derive the port from the UID so two users on one box do not collide on 8000.
PORT="${PORT:-$((8000 + $(id -u) % 1000))}"
MAXLEN="${MAXLEN:-65536}"

export CUDA_VISIBLE_DEVICES="$GPUS"
# FlashInfer JIT-compiles its sampler at startup and needs a CUDA toolkit; without one it
# aborts. Torch-native sampling needs no compiler. FLASHINFER_SAMPLER=1 opts back in.
[ "${FLASHINFER_SAMPLER:-0}" = "1" ] || export VLLM_USE_FLASHINFER_SAMPLER=0
# The vllm venv's bin/ (ninja etc.) must be on PATH for the engine subprocess.
export PATH="$(dirname "$(command -v "$VLLM_BIN" || echo "$VLLM_BIN")"):$PATH"

# --reasoning-parser moves <think>...</think> into `reasoning_content`, so the answer field
# holds only the answer. Qwen3.x: qwen3 (default). Gemma: REASONING_PARSER= (empty).
REASON_ARGS=()
[ -n "${REASONING_PARSER-qwen3}" ] && REASON_ARGS=(--reasoning-parser "${REASONING_PARSER:-qwen3}")
# Tool calling is only needed for the agentic route (codex CLI -> LiteLLM -> vLLM).
# Qwen3.x needs qwen3_coder (hermes silently leaves calls as text); Gemma needs gemma4.
TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-qwen3_coder}")

echo "serving $MODEL on GPU(s) $GPUS, http://127.0.0.1:$PORT/v1  (pid $$)"
exec "$VLLM_BIN" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAXLEN" \
  --served-model-name "$MODEL" \
  "${TOOL_ARGS[@]}" "${REASON_ARGS[@]}" \
  ${EXTRA_ARGS:-} \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}"
