#!/usr/bin/env python3
USAGE = """Batch Lean checker, seeded into an agent session's working directory.

    python3 lean_check_cli.py stmts.json

`stmts.json` is a JSON array of Lean 4 source strings. Mathlib is imported once into a
persistent REPL (about 10 s from node-local storage) and every entry is checked against that
one environment, so twenty statements in one invocation cost roughly twenty times 0.2 s rather
than twenty times the import. BATCH accordingly.

One JSON object per line on stdout:

    {"i": 0, "ok": true,  "sorries": 0, "errors": []}
    {"i": 1, "ok": false, "sorries": 1, "errors": ["...:3:2: error: ..."]}

`ok` means it elaborated with no errors AND no `sorry` -- a complete proof. A statement left as
`:= by sorry` therefore reports ok=false with sorries=1, which is the correct answer to "is
this proved", not a failure of the tool.

This is a WORKING TOOL for the session, not the adjudicator. Whatever the session concludes
from it, the adapter re-checks every statement and proof against its own REPL and assigns the
four states from that. See llm.complete_agentic."""
import json
import sys
from pathlib import Path

try:
    from .lean_repl import LeanRepl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lean_repl import LeanRepl


def main() -> int:
    if len(sys.argv) != 2:
        print(USAGE.strip(), file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    try:
        items = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"could not read {src}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(items, list):
        print("expected a JSON ARRAY of Lean source strings", file=sys.stderr)
        return 2

    lean = LeanRepl()
    print(f"# Mathlib ready in {lean.load_seconds}s, {len(items)} statements", file=sys.stderr)
    for i, code in enumerate(items):
        if not isinstance(code, str):
            print(json.dumps({"i": i, "ok": False, "sorries": 0,
                              "errors": ["entry is not a string"]}), flush=True)
            continue
        try:
            ok, msgs = lean.check(code)
            texts = [m if isinstance(m, str) else json.dumps(m) for m in (msgs or [])]
            print(json.dumps({"i": i, "ok": bool(ok),
                              "sorries": sum(1 for t in texts if "sorry" in t.lower()),
                              "errors": texts[:8]}), flush=True)
        except Exception as exc:
            print(json.dumps({"i": i, "ok": False, "sorries": 0,
                              "errors": [f"checker error: {exc}"]}), flush=True)
    lean.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
