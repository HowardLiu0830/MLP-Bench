"""Lean proving (forward construction): replace every `sorry` with a real proof.

80 rows, 120 scored targets (a target label can own more than one declaration). Score per
target, never per row.

Modes:
  cot    one tool-free call returning `{"proof_body": "<the whole completed file>"}`;
         one retry with an explicit no-tools note if no JSON came back
  agent  the model edits `<id>.lean` in a sandbox and compiles it itself; the FILE on disk
         is the answer

Both modes show the targets' informal statements by default (`--nl` adds the informal proof,
`--no-nl` withholds all natural language). The agent gets them as NL_STATEMENT.md in its
workspace, next to the Lean file, which the prompt points to. CoT keeps them inline.

The run stores each completed file under `sessions/<id>.final.lean`. Evaluation compiles it
against the pinned Mathlib and decides each target by `#print axioms` (no `sorryAx`, no
non-standard axiom), after rejecting any altered definition or target statement.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ..benchmarks import forward_construction as FC
from ..core import backends as B
from ..core.chat import complete
from ..core.jsonio import extract_json, strip_code_fence, write_json_atomic, write_jsonl_atomic
from ..core.leantext import strip_lean_comments
from ..core.paths import lean_project as default_lean_project
from ..core.paths import workspace_root
from ..core.registry import ModelEntry
from . import prompts as P
from .rows import NL_MODES, build_problem, statements_nl
from .scoring import score_file

__all__ = ["load_items", "solve", "evaluate", "COT_SYSTEM", "cot_user"]

# The tool-free contract of the paper's original one-shot prompt, with the wording about an
# extra library removed: every file here imports Mathlib only.
COT_SYSTEM = """You are an expert Lean 4 and Mathlib proof engineer.

You are given a complete Lean 4 source file in a SINGLE, NON-INTERACTIVE turn.
Some declarations (theorems, lemmas, corollaries) have their proof left as
`sorry`. Produce the COMPLETE file with every `sorry` replaced by a correct
Lean 4 proof -- in this one response, from reasoning alone. You have NO tools:
you cannot read any file, run `lake env lean`, search the web, or take a
second turn. Do all reasoning internally, then output the final file directly.

Hard rules:
- Return a single JSON object only. Required key: `proof_body` -- the ENTIRE
  completed Lean file as one string. Optional key: `rationale` (brief).
- Fill EVERY `sorry`/`admit` in the file (target theorems AND any helper lemmas).
- Do NOT change any statement: keep every `import`, `open`, `namespace`,
  `variable`, `section`, `def` (INCLUDING its body), and every theorem/lemma
  SIGNATURE (name, binders, and type up to `:=`) EXACTLY as given. You may
  only replace the proof text that comes after a theorem's `:=`.
- Do NOT add new `axiom`s. You MAY add your own private helper lemmas.
- Do not use markdown code fences inside `proof_body`.
- The file is SELF-CONTAINED and imports ONLY Mathlib. Do not add imports. All
  needed definitions are inlined in the file; the library's result-lemmas were
  intentionally removed -- re-derive what you need from Mathlib alone.
- ALWAYS return a best-effort completed file in `proof_body`. Never refuse and
  never return an empty or partial `proof_body`: if a step is hard, still write
  your best full proof attempt. A wrong attempt is far more useful than none.
"""

COT_USER = """[{noun}]
{row_id}

[Target theorems to prove -- these are what will be scored]
{targets}

{nl_section}[Lean file to complete -- replace every `sorry`, keep all statements/definitions unchanged]
{lean}

This is your ONLY turn -- no tools, no follow-up. Return the completed file now as JSON:
{{"rationale": "...", "proof_body": "<the entire completed Lean file>"}}"""

# With every tool removed, models sometimes write
# fabricated tool calls instead of the JSON. One retry that states the constraint.
NO_TOOL_RETRY = """Your previous response did not contain the required JSON object.

You have NO tools in this session: there is no Bash, no file reading, no shell, no
web access, and no second turn. Any tool call you write is inert text -- it will not
run and you will never receive a tool result. Do not describe, plan, or emit tool
calls in any syntax.

Answer directly from your own knowledge of Lean 4 and Mathlib, in THIS response.
Output ONLY the JSON object, with no prose before or after it:
{{"rationale": "...", "proof_body": "<the entire completed Lean file>"}}

[Previous response was rejected because: {prev}]

The task again:

"""

_PROOF_KEYS = ("proof_body", "completed_file", "file", "lean_file", "proof", "body")


def load_items(*, source: str | None = None, limit: int = 0, seed: int | None = None,
               ids: list[str] | None = None) -> list[FC.Row]:
    return FC.load(source=source, ids=ids, limit=limit or None, seed=seed)


def cot_user(row: FC.Row, nl: str = "statement") -> str:
    """`nl` as in `rows.NL_MODES`: none, statement (default), or row (statements and proof)."""
    if nl not in NL_MODES:
        raise ValueError(f"--nl must be one of {NL_MODES}, not {nl!r}")
    targets = "\n".join(
        f"- {t.label}: {' AND '.join(t.decls)}" + (" (one target; BOTH must be proved)"
                                                   if len(t.decls) > 1 else "")
        for t in row.targets)
    section = ""
    if nl != "none":
        section = f"[Informal statements of the targets]\n{statements_nl(row).strip()}\n\n"
    if nl == "row":
        section += f"[Informal proof / context]\n{row.nl_proof.strip() or '(none)'}\n\n"
    return COT_USER.format(noun="Paper" if row.source == FC.RESEARCH else "Theorem", row_id=row.id,
                           targets=targets, nl_section=section, lean=row.lean)


def extract_proof(text: str) -> tuple[str | None, str | None]:
    """The completed file from a CoT reply, or (None, why)."""
    try:
        payload = extract_json(text)
    except ValueError as exc:
        return None, f"no JSON object: {str(exc)[:200]}"
    for k in _PROOF_KEYS:
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            body = strip_code_fence(v)
            if re.search(r"\b(sorry|admit)\b", strip_lean_comments(body)):
                return body, "the file still contains sorry/admit"
            return body, None
    return None, "the JSON object has no `proof_body`"


def _save(sessions: Path, row_id: str, text: str) -> str:
    sessions.mkdir(parents=True, exist_ok=True)
    p = sessions / f"{row_id}.final.lean"
    p.write_text(text, encoding="utf-8")
    return p.name


def _solve_cot(row: FC.Row, entry: ModelEntry, sessions: Path, *, timeout: int,
               nl: str) -> dict[str, Any]:
    user = cot_user(row, nl)
    res = complete(entry, COT_SYSTEM, user, timeout=timeout, raw_dir=sessions, stem=row.id)
    body, why = extract_proof(res.text)
    attempts, n_tools = 1, len(res.tool_calls)
    if body is None and not res.timed_out:
        res = complete(entry, COT_SYSTEM, NO_TOOL_RETRY.format(prev=why) + user,
                       timeout=timeout, raw_dir=sessions, stem=f"{row.id}.retry")
        body, why = extract_proof(res.text)
        attempts, n_tools = 2, n_tools + len(res.tool_calls)
    return {
        "id": row.id, "source": row.source,
        "completed_file": _save(sessions, row.id, body) if body else None,
        "parse_error": why, "attempts": attempts, "n_tool_calls": n_tools,
        "returncode": res.returncode, "timed_out": res.timed_out,
        "tokens": res.usage.as_dict(), "nl_mode": nl,
        "error": None if res.returncode == 0 else (res.stderr or "")[-500:],
        "status": "answered" if body else f"NO ANSWER ({why})",
    }


# tau-bench's default is 30 agent steps (`max_num_steps`); a Lean proof needs more. Like
# tau-bench, the cap is the harness's alone: the prompt never mentions it.
DEFAULT_MAX_ACTIONS = 150
# Metered live at list price (core/pricing.py); the agent is not told, and the file on disk is
# graded. SWE-bench / mini-swe-agent use $3 (`cost_limit: 3.`).
DEFAULT_MAX_COST = 4.0


def _solve_agent(row: FC.Row, entry: ModelEntry, sessions: Path, *, timeout: int,
                 nl: str, max_actions: int | None = DEFAULT_MAX_ACTIONS,
                 max_cost: float | None = DEFAULT_MAX_COST) -> dict[str, Any]:
    project = default_lean_project()
    root = workspace_root() / "prove"
    root.mkdir(parents=True, exist_ok=True)
    ws = Path(tempfile.mkdtemp(prefix=f"{row.id}.", dir=str(root)))
    try:
        problem = build_problem(row, workspace=ws, nl=nl,
                                lean_project=project)
        before = problem.workfile.read_text(encoding="utf-8")
        if problem.nl_filename:     # beside <id>.agent.prompt.txt: the session as the agent saw it
            sessions.mkdir(parents=True, exist_ok=True)
            (sessions / f"{row.id}.{problem.nl_filename}").write_text(problem.nl_text,
                                                                       encoding="utf-8")
        system = P.agent_system(multi_target=problem.multi_target,
                                lean_command=problem.lean_command,
                                workfile=str(problem.workfile), lean_project=str(project))
        user = P.agent_user(workfile=str(problem.workfile), lean_project=str(project),
                            targets=problem.target_list(), nl_block=problem.nl_block())
        t0 = time.time()
        run = B.run_agent(entry, system, user, workspace=ws, timeout=timeout,
                          raw_dir=sessions, stem=f"{row.id}.agent", lean_project=project,
                          max_actions=max_actions or None, max_cost=max_cost or None)
        after = problem.workfile.read_text(encoding="utf-8")
        return {
            "id": row.id, "source": row.source,
            "completed_file": _save(sessions, row.id, after),
            "file_changed": after != before, "timed_out": run.timed_out,
            "returncode": run.returncode, "n_tool_calls": len(run.tool_calls),
            "n_lean_calls": run.n_lean_calls, "sandboxed": run.sandboxed,
            "egress_suspect": run.egress_suspect,
            "egress_suspect_reasons": run.egress_suspect_reasons[:20],
            "tokens": run.usage.as_dict(), "agent_s": round(time.time() - t0, 1),
            "nl_mode": problem.nl_mode,
            "n_actions": run.n_actions, "action_cap": run.action_cap,
            "action_cap_hit": run.action_cap_hit,
            "cost_usd": run.cost_usd, "cost_cap": run.cost_cap,
            "cost_cap_hit": run.cost_cap_hit,
            "status": ((f"edited lean={run.n_lean_calls}" if after != before
                        else "FILE UNCHANGED")
                       + (f" [cap {run.action_cap} hit]" if run.action_cap_hit else "")
                       + (f" [${run.cost_cap:g} hit]" if run.cost_cap_hit else "")),
        }
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def solve(row: FC.Row, entry: ModelEntry, mode: str, sessions: Path, *, timeout: int,
          nl: str = "statement", max_actions: int | None = DEFAULT_MAX_ACTIONS,
          max_cost: float | None = DEFAULT_MAX_COST) -> dict[str, Any]:
    if mode == "cot":
        return _solve_cot(row, entry, sessions, timeout=timeout, nl=nl)
    return _solve_agent(row, entry, sessions, timeout=timeout, nl=nl,
                        max_actions=max_actions, max_cost=max_cost)


def evaluate(run_dir: Path, answers: list[dict[str, Any]], *, jobs: int = 4,
             lean_timeout: int = 900) -> dict[str, Any]:
    """Grade each row's final file."""
    import concurrent.futures as cf

    rows = {r.id: r for r in FC.load()}
    scratch = run_dir / "work"

    def one(ans: dict[str, Any]) -> dict[str, Any]:
        row = rows[ans["id"]]
        name = ans.get("completed_file")
        path = run_dir / "sessions" / name if name else None
        if not path or not path.is_file():
            return {"id": row.id, "source": row.source, "checked": False,
                    "error": "no completed file", "n_targets": row.n_targets,
                    "n_targets_solved": 0, "success": False, "kernel_unsound": False,
                    "targets": [{"label": t.label, "per_target_solved": False,
                                 "per_target_reason": "no answer"} for t in row.targets]}
        sc = score_file(row, path.read_text(encoding="utf-8"), timeout=lean_timeout,
                        scratch_dir=scratch)
        return sc.as_dict()

    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        scored = list(ex.map(one, [a for a in answers if a["id"] in rows]))
    for s in scored:
        s.pop("lean", None)          # raw compiler logs are large; the verdicts are kept
    write_jsonl_atomic(run_dir / "scores.jsonl", scored)

    n_targets = sum(s["n_targets"] for s in scored)
    proved = sum(s["n_targets_solved"] for s in scored)
    reasons = [t.get("per_target_reason", "") for s in scored for t in s["targets"]]
    summary = {
        "task": "prove", "n_rows": len(scored), "n_targets": n_targets,
        "targets_proved": proved, "target_rate": proved / n_targets if n_targets else 0.0,
        "rows_compiled": sum(1 for s in scored if s.get("success")),
        "kernel_unsound": sum(1 for s in scored if s.get("kernel_unsound")),
        "statement_altered": sum(1 for r in reasons if r.startswith("REJECTED")),
        "agent_rows_without_lean": sum(1 for a in answers if a.get("n_lean_calls") == 0),
        "agent_rows_egress_suspect": [a["id"] for a in answers if a.get("egress_suspect")],
        "cot_rows_with_tools": sum(1 for a in answers if a.get("mode") == "cot"
                                   and (a.get("n_tool_calls") or 0) > 0),
        "by_source": {},
        "note": "the full-benchmark denominator is 120 targets (FC.N_TARGETS), not 80 rows",
    }
    for s in scored:
        b = summary["by_source"].setdefault(FC.canonical_source(s["source"]) or s["source"],
                                           {"targets": 0, "proved": 0})
        b["targets"] += s["n_targets"]
        b["proved"] += s["n_targets_solved"]
    write_json_atomic(run_dir / "summary.json", summary)
    return summary
