#!/usr/bin/env bash
# CoT: one tool-free call per item.
#
#   ./run.sh <task> <model> [options]
#   ./run.sh autoform claude-sonnet-5 --limit 3 --seed 0
#   ./run.sh prove    gpt-5.6-sol --effort medium
#
# Any other `mlpbench run` option passes through (--limit/--seed, --workers, --timeout,
# --source, --ids, --out, --resume, --nl/--no-nl). Then score with evaluation/<task>/eval.sh.
set -euo pipefail
source "$(dirname "$0")/../_common.sh"
usage_task "${1:-}"
TASK="$1"; MODEL="${2:?model alias required (mlpbench models)}"; shift 2
exec "$MLPBENCH" run --task "$TASK" --mode cot --model "$MODEL" "$@"
