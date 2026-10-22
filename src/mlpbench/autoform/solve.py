from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ..core import backends as B
from ..core.cpu import pin_process, workers as auto_workers
from ..core.jsonio import Appender, load_jsonl, write_json_atomic
from ..core.paths import lean_project as default_lean_project
from ..core.paths import runs_root, workspace_root
from ..core.registry import ModelEntry, describe, resolve
from . import solver_prompts as P
from .items import JointItem, MaskedItem
from .items import statement_span
from .leansurgery import Target

__all__ = ["solve_one", "main"]

MODES = ("agent",)


def _stem(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", item_id)[:60]


def _extract_statement(text: str, item: MaskedItem) -> str | None:
    """The target's statement from the agent's FILE: the text between its name and `:=`.

    This is what the hole stands for, so it is what gets spliced back. (An earlier port took
    everything from `theorem <name>` to the end of the file, which spliced a whole second
    declaration into the hole and made every agentic answer fail to parse.)
    """
    try:
        t = Target.build(text, item.target_name, item.target_names or [item.target_name])
    except Exception:
        return None
    # Cut exactly where the item builder cut the hole, or the splice duplicates text.
    try:
        start, end = statement_span(t.decl)
    except ValueError:
        return None
    stmt = t.decl.text[start:end]
    return None if "<<HOLE" in stmt else stmt     # left blank (a joint session's other holes)


def solve_one(item: MaskedItem | JointItem, entry: ModelEntry, *, mode: str, timeout: int,
              nag: bool = False, no_nl: bool = False, lean_project: Path | None = None,
              raw_dir: Path | None = None, scratch_root: Path | None = None,
              allow_unsandboxed: bool = False, sibling_targets: str = "hidden") -> dict[str, Any]:
    """`sibling_targets` (a one-target item only; a `JointItem` is "joint"): "shown" leaves the
    row's other targets, gold statements included, in the file; "hidden" drops them."""
    project = Path(lean_project) if lean_project else default_lean_project()
    root = Path(scratch_root or (workspace_root() / "autoform_solve"))
    root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="af_", dir=str(root)))
    joint = isinstance(item, JointItem)
    if not joint and sibling_targets not in ("shown", "hidden"):
        raise ValueError("sibling_targets must be 'shown' or 'hidden' for a one-target item")
    members = item.members if joint else [item]
    if no_nl:
        nl = ""
    elif joint:
        nl = P.nl_for_joint([{"target_name": m.target_name, "nl_statement": m.nl_statement,
                              "context": m.context, "nl_notes": m.nl_notes} for m in members])
    else:
        nl = P.nl_for({"nl_statement": item.nl_statement,
                       "context": item.context, "nl_notes": item.nl_notes})
    started = time.time()

    row: dict[str, Any] = {
        "item_id": item.item_id, "paper_id": item.paper_id,
        "target_name": ",".join(item.target_names) if joint else item.target_name,
        "joint": joint, "model": entry.alias, "mode": mode,
        "effort": entry.effort, "no_nl": no_nl,
        "sibling_targets": "joint" if joint else sibling_targets,
        "nag": nag if mode != "direct" else None,
        "fills": {}, "file_changed": None, "n_tool_calls": None, "n_lean_calls": None,
        "n_turns": None, "timed_out": None, "returncode": None,
        "draft_binders": None, "final_binders": None, "eroded": None,
        "tokens": {}, "cost_usd": None, "error": None,
    }

    try:
        answer = scratch / P.ANSWER_FILE
        # One target: the row's other targets stay in the file ("shown") or are dropped
        # ("hidden"); a Graduate-level row has no others, so the two agree there.
        answer.write_text(item.masked_file if joint or sibling_targets == "shown"
                          else item.masked_file_alone, encoding="utf-8")
        template = P.joint_system(P.AGENT_SYSTEM, len(members)) if joint else P.AGENT_SYSTEM
        system = template.format(
            answer=P.ANSWER_FILE, project=project,
            mathlib=project / ".lake/packages/mathlib/Mathlib") + (P.NAG if nag else "")
        if joint:
            names = ", ".join(f"`{m.target_name}` (<<HOLE {k}>>)"
                              for k, m in enumerate(members, start=1))
            user = (f"TARGET THEOREMS: {names}\n\nNATURAL LANGUAGE:\n{nl}\n\n"
                    f"Write the {len(members)} statements into `{P.ANSWER_FILE}`, then compile "
                    f"it from `{project}` and fix it until the only messages are the expected "
                    f"`declaration uses 'sorry'` warnings.")
        else:
            user = (f"TARGET THEOREM: `{item.target_name}`\n\nNATURAL LANGUAGE:\n{nl}\n\n"
                    f"Write the statement into `{P.ANSWER_FILE}`, then compile it from "
                    f"`{project}` and fix it until the only message is the expected "
                    f"`declaration uses 'sorry'` warning.")

        before = answer.read_text(encoding="utf-8")
        stem = _stem(item.item_id)
        run = B.run_agent(entry, system, user, workspace=scratch, timeout=timeout,
                          raw_dir=raw_dir, stem=f"{stem}.agent", lean_project=project,
                          allow_unsandboxed=allow_unsandboxed)
        n_turns = 1

        if not run.n_lean_calls:
            held, done = (("statements", "messages are the expected `declaration uses 'sorry'` "
                           "warnings") if joint else
                          ("statement", "message is the expected `declaration uses 'sorry'` warning"))
            follow = (
                f"`{P.ANSWER_FILE}` currently holds your {held}. You have NOT compiled it.\n"
                f"Run:  cd {project} && lake env lean --json $OLDPWD/{P.ANSWER_FILE}\n"
                f"Fix every error it reports and recompile, until the only {done}. Edit the "
                f"file on disk; the harness reads the file, not your reply."
            )
            run2 = B.run_agent(entry, system, follow, workspace=scratch, timeout=timeout,
                               raw_dir=raw_dir, stem=f"{stem}.agent.turn2",
                               lean_project=project, allow_unsandboxed=allow_unsandboxed)
            n_turns = 2
            run.tool_calls += run2.tool_calls
            run.egress_suspect_reasons += run2.egress_suspect_reasons
            run.returncode = run2.returncode
            run.timed_out = run.timed_out or run2.timed_out

        after = answer.read_text(encoding="utf-8")
        stmts = {m.item_id: _extract_statement(after, m) for m in members}
        statement = stmts[item.item_id] if not joint else None
        if joint:
            row["fills_by_item"] = stmts
        row.update({
            "fills": ({str(k): v for k, v in enumerate(stmts.values(), start=1) if v} if joint
                      else ({"1": statement} if statement else {})),
            "file_changed": after != before,
            "n_tool_calls": len(run.tool_calls), "n_lean_calls": run.n_lean_calls,
            "n_turns": n_turns, "timed_out": run.timed_out, "returncode": run.returncode,
            "egress_suspect": run.egress_suspect,
            "egress_suspect_reasons": run.egress_suspect_reasons[:20],
            "tokens": run.usage.as_dict(), "cost_usd": run.usage.cost_usd,
        })
        if statement:
            row["final_binders"] = P.n_binders(statement)
        if row["draft_binders"] is not None and row["final_binders"] is not None:
            row["eroded"] = row["final_binders"] < row["draft_binders"]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    row["wall_clock_s"] = round(time.time() - started, 1)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlpbench autoform-solve",
                                 description='Write the Lean statement from the informal one, in agent mode.')
    ap.add_argument("--items", type=Path, required=True,
                    help="items jsonl from `mlpbench autoform-items`")
    ap.add_argument("--model", required=True, help="a registry alias")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--mode", choices=MODES, default="agent",
                    help="agent is the only mode this release ships")
    ap.add_argument("--nag", action="store_true",
                    help="agent/guard only: append the explicit-hypotheses paragraph. "
                         "The ablation that separates `cannot see the implicit assumption` "
                         "from `did not bother to write it down`.")
    ap.add_argument("--no-nl", action="store_true",
                    help="withhold the natural language entirely")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="seconds per item; 1200 is the budget the paper reports")
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lean-project", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep-sessions", action="store_true",
                    help="persist each session's stream, which is what a later audit reads")
    a = ap.parse_args(argv)

    entry = resolve(a.model, effort=a.effort)
    items = [MaskedItem.from_json(r) for r in load_jsonl(a.items)]
    if a.limit:
        if a.seed is None:
            ap.error("--limit needs --seed: the item file follows the benchmark's order, "
                     "which is one source then the other.")
        import random

        rng = random.Random(a.seed)
        picked = items[:]
        rng.shuffle(picked)
        items = picked[:a.limit]

    out = Path(a.out) if a.out else (
        runs_root() / "autoform" / entry.alias.replace("/", "-")
        / f"{a.mode}{'-nonl' if a.no_nl else ''}{'-nag' if a.nag else ''}"
          f"__{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    answers = out / "answers.jsonl"
    raw_dir = (out / "sessions") if a.keep_sessions else None

    mask = pin_process()
    write_json_atomic(out / "meta.json", {
        "items": str(a.items), "n_items": len(items), "mode": a.mode,
        "model": describe(entry), "no_nl": a.no_nl, "nag": a.nag,
        "timeout": a.timeout, "limit": a.limit, "seed": a.seed,
        "cpu_mask": f"{min(mask)}-{max(mask)}", "argv": sys.argv,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    app = Appender(answers, id_key="item_id", resume=a.resume)
    before = len(items)
    items = [it for it in items if it.item_id not in app.done]
    if a.resume and len(items) < before:
        print(f"[af] --resume: {before - len(items)} already done, {len(items)} to go")
    print(f"[af] {len(items)} items to solve, mode={a.mode}, model={entry.alias}")

    n_workers = auto_workers(a.workers)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(solve_one, it, entry, mode=a.mode, timeout=a.timeout,
                             nag=a.nag, no_nl=a.no_nl, lean_project=a.lean_project,
                             raw_dir=raw_dir): it for it in items}
        for fut in cf.as_completed(futures):
            it = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"item_id": it.item_id, "paper_id": it.paper_id,
                       "target_name": it.target_name, "mode": a.mode,
                       "error": f"{type(exc).__name__}: {exc}", "fills": {}}
            app.write(row)
            done += 1
            lean = row.get("n_lean_calls")
            print(f"[{done}/{len(items)}] {row['item_id'][:48]:48s} "
                  f"answer={'yes' if row.get('fills') else 'NO ':3s} "
                  f"lean={'-' if lean is None else lean}"
                  + (f"  turns={row['n_turns']}" if row.get("n_turns") else "")
                  + (f"  ERROR {row['error'][:60]}" if row.get("error") else ""), flush=True)

    rows = load_jsonl(answers)
    never_compiled = [r["item_id"] for r in rows
                      if r.get("n_lean_calls") == 0]
    if never_compiled:
        print(f"[af] {len(never_compiled)} agent row(s) never invoked Lean even after the "
              f"second turn; they are not agentic results: {never_compiled[:5]}")
    print(f"[af] wrote {answers}")
    print(f"[af] next: `mlpbench af-predict --items {a.items} --answers {answers}`. "
          f"This release ships no equivalence grader: the predictions file is one row per "
          f"(declaration, sample) with the whole predicted Lean file, which is the shape a "
          f"grader reads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
