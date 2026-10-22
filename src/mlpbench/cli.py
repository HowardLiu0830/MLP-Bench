"""`mlpbench` -- run a model on a task, then score the run.

    mlpbench run  --task {autoform,prove,verify} --mode {cot,agent} --model ALIAS [...]
    mlpbench eval --task {autoform,prove,verify} --run RUN_DIR [...]
    mlpbench models | doctor | autoform-items

Models: a registry alias (`mlpbench models`), or an open-weights model named inline:
`vllm:<hf-id>` (local vLLM, CoT), `openrouter:<id>` (CoT), `local:<hf-id>` (agentic, via
the LiteLLM proxy).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmarks.forward_construction import SOURCE_CHOICES
from .core.chat import COT_TIMEOUT

TASKS = ("autoform", "prove", "verify")


def _common_run_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--mode", choices=("cot", "agent"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    ap.add_argument("--seed", type=int, default=None, help="required with --limit")
    ap.add_argument("--ids", default=None, help="comma-separated item ids")
    ap.add_argument("--source", choices=SOURCE_CHOICES, default=None,
                    help="Research-level (alias jmlr) or Graduate-level (alias lecture)")
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--timeout", type=int, default=None,
                    help="seconds per item; agent 3600/2400/3600 (autoform/prove/verify), "
                         "cot effectively none")
    nl = ap.add_mutually_exclusive_group()
    nl.add_argument("--nl", dest="nl", action="store_true", default=None,
                    help="prove: also give the informal proof (by default the informal "
                         "statements only)")
    nl.add_argument("--no-nl", dest="nl", action="store_false",
                    help="withhold the natural language")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-actions", type=int, default=None,
                    help="prove, agent mode: stop the agent after N actions (one per tool "
                         "call or shown message, same for claude and codex) and grade what is "
                         "on disk; 0 = no "
                         "cap. Default 150. The agent is never told about the cap.")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="prove, agent mode: stop the agent once its list-price cost reaches "
                         "USD (metered live, see core/pricing.py) and grade what is on disk; "
                         "0 = no cap. Default 4 (SWE-bench uses 3). The agent is never told.")
    ap.add_argument("--fl-task", default="fl_refute", choices=("fl_refute", "prove"),
                    help="verify: FL Refute (default) or FL Prove, the paper's Appendix F.3 "
                         "prompts")
    ap.add_argument("--sibling-targets", default=None, choices=("shown", "joint", "hidden"),
                    help="autoform: what a Research-level target's solver sees of the other "
                         "targets of its row (3-4 per paper). joint (default in agent mode): "
                         "one session per row with every target blanked. hidden (default in "
                         "CoT mode): one item per target, the other targets dropped from the "
                         "file. shown: one item per target, the whole file in view, the other "
                         "targets' gold Lean statements included")


def withhold_nl(task: str, nl: bool | None) -> bool:
    """`nl` is True for --nl, False for --no-nl, None for neither. Withheld only on --no-nl."""
    return nl is False


def prove_nl_mode(nl: bool | None) -> str:
    """prove: the informal statements by default, plus the informal proof on --nl, nothing on
    --no-nl (`prove.rows.NL_MODES`)."""
    return "row" if nl is True else "none" if nl is False else "statement"


def _run(a: argparse.Namespace) -> int:
    from .core.registry import resolve
    from .runner import RunSpec, run_dir_for, run_items

    if a.limit and a.seed is None:
        raise SystemExit("--limit needs --seed: the benchmark files are ordered, so a prefix "
                         "is one stratum, not a sample.")
    if a.sibling_targets is not None and a.task != "autoform":
        raise SystemExit("--sibling-targets is for --task autoform")
    if a.sibling_targets is None:   # joint needs a session; CoT hides the siblings instead
        a.sibling_targets = "joint" if a.mode == "agent" else "hidden"
    if a.sibling_targets == "joint" and a.mode != "agent":
        raise SystemExit("--sibling-targets joint is for --mode agent (CoT is one call per "
                         "target; use hidden to drop the other targets)")
    entry = resolve(a.model, effort=a.effort)
    a.no_nl = withhold_nl(a.task, a.nl)
    nl_extra = ({"nl": prove_nl_mode(a.nl)} if a.task == "prove" else {"no_nl": a.no_nl})
    ids = [s for s in (a.ids or "").split(",") if s] or None
    out =run_dir_for(a.task, a.mode, entry, a.out)
    spec = RunSpec(task=a.task, mode=a.mode, entry=entry, out=out, workers=a.workers,
                   resume=a.resume, extra={"limit": a.limit, "seed": a.seed,
                                           "source": a.source, **nl_extra,
                                           **({"fl_task": a.fl_task}
                                              if a.task == "verify" else {}),
                                           **({"timeout": a.timeout or 3600}
                                              if a.task == "autoform" and a.mode == "agent"
                                              else {}),
                                           **({"sibling_targets": a.sibling_targets}
                                              if a.task == "autoform" else {})})

    if a.task == "autoform":
        from .autoform import task as T

        items = T.load_items(source=a.source, limit=a.limit, seed=a.seed, ids=ids)
        if a.sibling_targets == "joint":     # a Research-level row's targets in one session
            from .autoform.items import joint_units
            items = joint_units(items)
        # 3600 s per agent session (a Research-level joint session writes 3-4 targets)
        timeout = a.timeout or (3600 if a.mode == "agent" else COT_TIMEOUT)
        run_items(spec, items,
                  lambda it, ses: T.solve(it, entry, a.mode, ses, timeout=timeout,
                                          no_nl=a.no_nl, sibling_targets=a.sibling_targets),
                  item_id=lambda it: it.item_id)
    elif a.task == "prove":
        from .prove import task as T

        rows = T.load_items(source=a.source, limit=a.limit, seed=a.seed, ids=ids)
        timeout = a.timeout or (2400 if a.mode == "agent" else COT_TIMEOUT)
        cap = T.DEFAULT_MAX_ACTIONS if a.max_actions is None else a.max_actions
        spec.extra["max_actions"] = cap if a.mode == "agent" else None
        usd = T.DEFAULT_MAX_COST if a.max_cost is None else a.max_cost
        if a.mode == "agent" and usd:
            from .core.pricing import price_for
            if price_for(entry.model_id) is None:
                raise SystemExit(f"no list price for {entry.model_id} in core/pricing.py; "
                                 "add one or pass --max-cost 0")
        spec.extra["max_cost"] = usd if a.mode == "agent" else None
        run_items(spec, rows,
                  lambda r, ses: T.solve(r, entry, a.mode, ses, timeout=timeout,
                                         nl=prove_nl_mode(a.nl), max_actions=cap,
                                         max_cost=usd),
                  item_id=lambda r: r.id)
    else:
        from .verify import task as T

        from .verify.adapter import require_upstream

        require_upstream()              # fail before any item runs, not once per item
        items = T.load_items(limit=a.limit, seed=a.seed, ids=ids)
        timeout = a.timeout or (3600 if a.mode == "agent" else COT_TIMEOUT)
        run_items(spec, items,
                  lambda it, ses: T.solve(it, entry, a.mode, ses, timeout=timeout,
                                          fl_task=a.fl_task),
                  item_id=lambda it: it["id"])
    print(f"\nnext:  mlpbench eval --task {a.task} --run {out}")
    return 0


def _eval(a: argparse.Namespace) -> int:
    from .runner import load_run

    cfg, answers = load_run(a.run)
    task = a.task or cfg.get("task")
    if task == "autoform":
        from .autoform.beq import BEqConfig
        from .autoform.task import evaluate
        from .core.registry import resolve

        sampler = None if a.grader_model == "none" else resolve(a.grader_model,
                                                                effort=a.grader_effort)
        bcfg = BEqConfig(sampler=sampler, agentic=not a.grader_tool_free, n=a.grader_n,
                         sampler_timeout=a.grader_timeout, work_dir=a.run / "work",
                         raw_dir=(a.run / "grader_sessions") if a.keep_grader_sessions else None,
                         strip_gate=a.gates, vacuous_gate=a.gates, bridge_gate=a.gates,
                         bridge_model=(resolve(a.bridge_model, effort=a.grader_effort)
                                       if a.gates and sampler is not None else None))
        second = None
        if a.double_check != "none" and sampler is not None:     # paper: two graders
            if a.double_check == sampler.alias:
                raise SystemExit("--double-check: the second grader must differ from the first")
            from dataclasses import replace
            second = replace(bcfg, sampler=resolve(a.double_check, effort=a.grader_effort))
        summary = evaluate(a.run, answers, cfg=bcfg, jobs=a.jobs, refute=a.refute,
                           resume=a.resume, second=second)
        keys = ("n_items", "n_targets", "beq", "beq_rate", "valid", "valid_rate", "typecheck",
                "typecheck_rate", "relation", "gate_rejected_directions", "refutation",
                "agent_rows_without_lean", "agent_rows_egress_suspect", "cot_rows_with_tools",
                "per_grader")
    elif task == "prove":
        from .prove.task import evaluate

        summary = evaluate(a.run, answers, jobs=a.jobs)
        keys = ("n_rows", "n_targets", "targets_proved", "target_rate", "rows_compiled",
                "kernel_unsound", "statement_altered", "agent_rows_without_lean",
                "agent_rows_egress_suspect", "cot_rows_with_tools")
    elif task == "verify":
        from .verify.task import evaluate, rebind

        if a.rebind:
            answers, n = rebind(a.run, answers, backend=(cfg.get("model") or {}).get("backend")
                                or "claude")
            old = a.run / "answers.prebind.jsonl"
            if n:
                from .core.jsonio import write_jsonl_atomic
                if not old.exists():
                    (a.run / "answers.jsonl").replace(old)
                write_jsonl_atomic(a.run / "answers.jsonl", answers)
            print(f"[rebind] {n} row(s) re-checked" + (f"; the old answers are in {old}" if n else ""))
        summary = evaluate(a.run, answers, malf_rule=a.malf_rule,
                           parse_penalty=not a.no_parse_penalty)
        keys = ("precision", "recall", "F1", "accuracy", "headline_basis",
                "n", "n_flawed", "n_correct", "TP", "FP", "FN", "TN", "TN_cov", "acc_cov",
                "no_answer", "mislocated", "false_alarms", "agent_rows_egress_suspect",
                "cot_rows_with_tools", "rows_rebound", "rows_literal_binding", "rows_rejoined")
    else:
        raise SystemExit(f"unknown task {task!r}; pass --task")
    print(f"\n== {task} :: {a.run}")
    for k in keys:
        if k in summary:
            v = summary[k]
            print(f"  {k:28s} {v:.3f}" if isinstance(v, float) else f"  {k:28s} {v}")
    print(f"\nwrote {a.run / 'scores.jsonl'} and {a.run / 'summary.json'}")
    return 0


def _models(_a) -> int:
    from .core.registry import table

    for row in table():
        print(f"  {row['alias']:24s} {row['backend']:7s} {row['route']:10s} {row['model_id']}")
    print("\n  vllm:<hf-id>         openai  vllm       (local vLLM server; CoT)")
    print("  openrouter:<id>      openai  openrouter (OpenRouter HTTP; CoT)")
    print("  local:<hf-id>        codex   local      (codex CLI -> LiteLLM -> vLLM; agentic)")
    return 0


def _doctor(_a) -> int:
    import shutil

    from .core.paths import data_root, lean_project, workspace_root
    from .core.sandbox import selftest

    print(f"  data       {data_root()}")
    lp = lean_project()
    built = (lp / ".lake" / "packages" / "mathlib").is_dir()
    print(f"  lean       {lp}  ({'built' if built else 'NOT BUILT -- ./setup.sh --lean'})")
    print(f"  workspace  {workspace_root()}")
    for b in ("bwrap", "python3", "claude", "codex", "lake"):
        print(f"  {b:10s} {shutil.which(b) or 'MISSING'}")
    from .core.backends.codex import native_binary_problem
    from .core.config import settings

    cx = settings().codex_bin
    shim = native_binary_problem(cx) if cx else None
    if shim:
        print(f"  codex      UNUSABLE IN THE SANDBOX: {shim}")
    st = selftest()
    print(f"  sandbox    {'ok' if st.ok else 'FAILED'} {st.errors or ''}")
    return 0 if (st.ok and built and not shim) else 1


def _items(a) -> int:
    from .autoform.task import build_items
    from .core.jsonio import write_jsonl_atomic
    from .core.paths import auto_formalization_jsonl

    items = build_items()
    out = a.out or auto_formalization_jsonl()
    write_jsonl_atomic(out, [it.to_json() for it in items])
    print(f"wrote {len(items)} items to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlpbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a model on a task")
    _common_run_args(r)
    r.set_defaults(fn=_run)

    e = sub.add_parser("eval", help="score a run directory")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--task", choices=TASKS, default=None, help="default: from config.json")
    e.add_argument("--jobs", type=int, default=4, help="parallel Lean checks")
    g = e.add_argument_group("autoform (BEq grader)")
    g.add_argument("--grader-model", default="gpt-5.6-sol",
                   help="the BEq tactic sampler, also the refutation agent for validity; "
                        "`none` = exact? only and no refutation (and no second grader)")
    g.add_argument("--grader-effort", default="medium")
    g.add_argument("--double-check", nargs="?", const="claude-opus-5", default="claude-opus-5",
                   metavar="MODEL",
                   help="the second grader, on by default as in the paper (claude-opus-5): "
                        "every pair is also graded by MODEL with the same settings; BEq if "
                        "either grader proves it, refuted if either refutes it; each grader's "
                        "own scores in summary.per_grader. `none` = the first grader alone")
    g.add_argument("--grader-n", type=int, default=4, help="samples per direction")
    g.add_argument("--grader-timeout", type=int, default=900,
                   help="seconds per grader session (each BEq sample and each refutation)")
    g.add_argument("--grader-tool-free", action="store_true",
                   help="sample and refute without Lean access, one call each (the default "
                        "grader is agentic)")
    g.add_argument("--no-gates", dest="gates", action="store_false", default=True,
                   help="skip the BEq soundness gates (strip / vacuous / bridge), which are on "
                        "by default as in the paper (App. B.3)")
    g.add_argument("--bridge-model", default="gpt-6.1-sol",
                   help="the LLM behind the bridge gate's screen of every accepted script")
    g.add_argument("--no-refute", dest="refute", action="store_false", default=True,
                   help="skip the refutation attempt behind the validity rate (on by default; "
                        "one grader session per compiling prediction). "
                        "Without it `valid` equals `typecheck`, an upper bound")
    g.add_argument("--keep-grader-sessions", action="store_true")
    g.add_argument("--resume", action="store_true",
                   help="reuse the pairs an interrupted grading already wrote to "
                        "scores.partial.jsonl (refused if the grader settings differ)")
    v = e.add_argument_group("verify")
    v.add_argument("--malf-rule", choices=("region", "slot", "item", "none"), default="region",
                   help="agent rows: how a step that failed to formalise counts")
    v.add_argument("--no-parse-penalty", action="store_true",
                   help="let a dead/unparsed row fall through to 'no error found' (a free TN on "
                        "correct items). Off by default")
    v.add_argument("--rebind", action="store_true",
                   help="first re-check, from the saved sessions and with no model call, the rows "
                        "read wrongly at run time: theorems labelled `theorem=1` instead of "
                        "`theorem=T1` (matched literally before 2026-10-18, which dropped every "
                        "step), and claude sessions whose answer was in an earlier `result` turn "
                        "(only the last was read before 2026-10-20); the old answers are kept as "
                        "answers.prebind.jsonl. Needs MLPBENCH_LEAN_PROJECT")
    e.set_defaults(fn=_eval)

    sub.add_parser("models", help="list model aliases").set_defaults(fn=_models)
    sub.add_parser("doctor", help="check binaries, Lean and the sandbox").set_defaults(fn=_doctor)
    it = sub.add_parser("autoform-items", help="rebuild Benchmarks/auto-formalization.jsonl")
    it.add_argument("--out", type=Path, default=None)
    it.set_defaults(fn=_items)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
