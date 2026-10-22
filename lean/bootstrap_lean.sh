#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The built project lives OUTSIDE the repo (see mlpbench/core/paths.py:lean_project).
PROJECT=${MLPBENCH_LEAN_PROJECT:-${XDG_DATA_HOME:-$HOME/.local/share}/mlpbench/lean_project}
DURABLE=${LEAN_REPL_BIN:-$HOME/.local/share/mlpbench/bin/repl}
LOCAL=${LEAN_LOCAL_ROOT:-/dev/shm/$(id -un)-lean}
FORCE=${1:-}

export PATH=$HOME/.elan/bin:$PATH
mkdir -p "$PROJECT"
for f in lean-toolchain lakefile.lean lake-manifest.json; do
  if [ ! -f "$PROJECT/$f" ]; then
    cp "$HERE/$f" "$PROJECT/$f"
  elif ! cmp -s "$HERE/$f" "$PROJECT/$f"; then
    echo "$PROJECT/$f differs from lean/$f; refusing to build against a different pin." >&2
    echo "Delete $PROJECT, or point MLPBENCH_LEAN_PROJECT elsewhere." >&2
    exit 1
  fi
done
echo "project: $PROJECT"
TOOLCHAIN_NAME=$(cat "$PROJECT/lean-toolchain")
if [ ! -d "$PROJECT/.lake/packages/mathlib" ]; then
  echo "fetching Mathlib into $PROJECT (lake exe cache get)"
  ( cd "$PROJECT" && lake exe cache get )
fi
echo "toolchain: $TOOLCHAIN_NAME"

if [ "$FORCE" = "--force" ] || [ ! -f "$DURABLE" ]; then
  TAG="v${TOOLCHAIN_NAME#leanprover/lean4:v}"
  echo "building repl at $TAG"
  rm -rf /tmp/build_repl
  git clone --quiet https://github.com/leanprover-community/repl /tmp/build_repl
  git -C /tmp/build_repl checkout --quiet "$TAG"
  ( cd /tmp/build_repl && lake build )
  mkdir -p "$(dirname "$DURABLE")"
  cp /tmp/build_repl/.lake/build/bin/repl "$DURABLE"
  chmod +x "$DURABLE"
fi
mkdir -p "$LOCAL"
[ -f "$LOCAL/repl" ] || { cp "$DURABLE" "$LOCAL/repl"; chmod +x "$LOCAL/repl"; }
echo "repl: $LOCAL/repl"

SRC=$PROJECT/.lake/packages
DST=$LOCAL/packages
UNITS=$LOCAL/.stage_units
: > "$UNITS"
for p in Cli batteries Qq aesop proofwidgets importGraph LeanSearchClient plausible mathlib; do
  R="$SRC/$p/.lake/build/lib/lean"
  [ -d "$R" ] || continue
  D="$DST/$p/.lake/build/lib/lean"
  mkdir -p "$D"
  for e in "$R"/*; do
    b=$(basename "$e")
    if [ -d "$e" ] && [ "$p" = "mathlib" ]; then
      for s in "$e"/*; do echo "$s|$D/$b/$(basename "$s")" >> "$UNITS"; done
    else
      echo "$e|$D/$b" >> "$UNITS"
    fi
  done
done
echo "staging $(wc -l < "$UNITS") units, 24 at a time"
xargs -a "$UNITS" -P 24 -I{} bash -c 'IFS="|" read -r s d <<< "{}"; mkdir -p "$(dirname "$d")"; rsync -a --size-only "$s" "$(dirname "$d")/"'
rm -f "$UNITS"

echo "smoke test: lake env lean"
T=$(mktemp --suffix=.lean)
trap 'rm -f "$T"' EXIT
cat > "$T" <<'LEAN'
import Mathlib
theorem smoke (x : ℝ) (h : 0 < x) : 0 < x + 1 := by linarith
LEAN
( cd "$PROJECT" && lake env lean "$T" )
echo "LEAN OK"
