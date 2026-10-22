#!/usr/bin/env bash
# Score a autoform run directory: writes scores.jsonl and summary.json into it.
#
#   ./eval.sh <run-dir> [mlpbench eval options]
set -euo pipefail
source "$(dirname "$0")/../../running/_common.sh"
RUN="${1:?usage: $0 <run-dir> [options]}"; shift
exec "$MLPBENCH" eval --task autoform --run "$RUN" "$@"
