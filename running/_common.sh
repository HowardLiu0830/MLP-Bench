# Sourced by the run scripts: find the mlpbench binary (./.venv from setup.sh, else PATH).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -x "$ROOT/.venv/bin/mlpbench" ]; then
  MLPBENCH="$ROOT/.venv/bin/mlpbench"
elif command -v mlpbench >/dev/null 2>&1; then
  MLPBENCH="$(command -v mlpbench)"
else
  echo "mlpbench not found: run ./setup.sh first." >&2
  exit 1
fi
usage_task() {
  case "${1:-}" in autoform|prove|verify) ;; *)
    echo "usage: $0 {autoform|prove|verify} <model> [mlpbench run options...]" >&2; exit 2;;
  esac
}
