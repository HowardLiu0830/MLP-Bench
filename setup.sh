#!/usr/bin/env bash
# Set up the MLP-Bench environment.
#
#   ./setup.sh                 uv sync: ./.venv from uv.lock, this repo editable; fetch Safe
#   ./setup.sh --lean          also fetch Mathlib and build the Lean REPL (~5.4 GB)
#   ./setup.sh --no-safe       skip fetching Safe (only FL verification needs it)
#   ./setup.sh --proxy         also install LiteLLM (open-source / OpenRouter models)
#   ./setup.sh --python 3.11   pick the Python version for the venv
set -euo pipefail
cd "$(dirname "$0")"

WANT_LEAN=0 WANT_SAFE=1 EXTRAS=() PYVER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --lean)   WANT_LEAN=1 ;;
    --no-safe) WANT_SAFE=0 ;;
    --proxy)  EXTRAS+=(proxy) ;;
    --python) PYVER="${2:?--python needs a version}"; shift ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

say() { printf '\n== %s\n' "$*"; }

say "uv"
if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'MSG'
uv is not on PATH. Install it, then re-run this script:

    curl -LsSf https://astral.sh/uv/install.sh | sh

This script will not install it for you: it would be piping a remote script into a shell
on your machine, which is your call to make and not a setup step to hide.
MSG
  exit 1
fi
uv --version

if [ "$WANT_LEAN" = 1 ]; then
  say "elan"
  export PATH="$HOME/.elan/bin:$PATH"
  if ! command -v elan >/dev/null 2>&1; then
    cat >&2 <<'MSG'
--lean needs elan (the Lean toolchain manager), and it is not on PATH or in ~/.elan/bin.
Install it, then re-run this script:

    curl https://elan.lean-lang.org/elan-init.sh -sSf | sh

elan then installs the exact Lean version pinned in lean/lean-toolchain on its own.
MSG
    exit 1
  fi
  elan --version
fi

say "environment (uv sync)"
# uv.lock pins every dependency; --frozen refuses to run if it is stale rather than
# silently re-resolving. After changing pyproject.toml, run `uv lock` and commit uv.lock.
SYNC=(sync --frozen)
for e in ${EXTRAS[@]+"${EXTRAS[@]}"}; do SYNC+=(--extra "$e"); done
[ -n "$PYVER" ] && SYNC+=(--python "$PYVER")
uv "${SYNC[@]}"
PY="$PWD/.venv/bin/python"
"$PY" -c "import mlpbench; print('mlpbench', mlpbench.__version__, 'installed')"

say "external binaries"
for b in bwrap claude codex elan lake; do
  if command -v "$b" >/dev/null 2>&1; then
    printf '  %-8s %s\n' "$b" "$(command -v "$b")"
  else
    printf '  %-8s MISSING\n' "$b"
  fi
done
cat <<'MSG'

  bwrap   required for agentic runs (the agent is sandboxed, never run without it).
  claude  required to evaluate a Claude model agentically; log in with `claude` once.
  codex   required to evaluate a GPT model agentically; log in with `codex` once.
  elan    Lean toolchain manager; needed for anything that compiles Lean.
  lake    comes with elan; --lean uses it to fetch Mathlib.
MSG

if [ "$WANT_SAFE" = 1 ]; then
  # FL verification fills its system prompt with Safe's own prompts/formalize.txt, as in the
  # paper. Safe ships no licence, so it is fetched here rather than vendored. Pinned to the
  # commit the paper's runs used; the file's hash is checked.
  say "Safe (FL verification's formalisation rules)"
  SAFE_REPO="https://github.com/liuchengwucn/Safe"
  SAFE_COMMIT="b9b800aaae3ea4b16c87d3636ce2a0973486dd1f"
  SAFE_SHA256="507d77bb4f3ba9d3cef4d427003ce3ab8267d8cbda5cf212737b15afad02f2d9"
  SAFE_DIR="$("$PY" -c "from mlpbench.core.paths import safe_upstream; print(safe_upstream())")"
  if [ ! -d "$SAFE_DIR/.git" ]; then
    mkdir -p "$(dirname "$SAFE_DIR")"
    git clone --quiet "$SAFE_REPO" "$SAFE_DIR"
  fi
  git -C "$SAFE_DIR" fetch --quiet origin "$SAFE_COMMIT" 2>/dev/null || true
  git -C "$SAFE_DIR" -c advice.detachedHead=false checkout --quiet "$SAFE_COMMIT"
  got="$(sha256sum "$SAFE_DIR/prompts/formalize.txt" | cut -d' ' -f1)"
  if [ "$got" != "$SAFE_SHA256" ]; then
    echo "Safe's prompts/formalize.txt at $SAFE_DIR does not match the pinned hash" >&2
    exit 1
  fi
  echo "  $SAFE_DIR @ ${SAFE_COMMIT:0:7} (formalize.txt verified)"
fi

if [ "$WANT_LEAN" = 1 ]; then
  say "Lean (this takes a while)"
  bash lean/bootstrap_lean.sh
fi

say "done"
echo "Activate with:  source .venv/bin/activate"
