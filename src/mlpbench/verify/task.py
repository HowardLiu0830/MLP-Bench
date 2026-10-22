"""FL verification (backward): is the target proof correct, and if not, where does it first go wrong?

120 items (60 correct / 60 flawed, the file SORTED by label -- never take a prefix). The
model sees only `context` and each theorem's `local_id`, statement and numbered step claims
(`adapter.bundle_text`); every other field is gold or predicts it.

The FL pipeline, in two modes that differ only in whether the model may iterate:
  agent  a CLI agent with a Lean checker formalises every step and tries to prove or refute it;
  cot    the same prompt without the checker, one tool-free generation (the paper's `direct`).
Either way the harness re-checks every artefact in its own REPL and reports the FIRST flagged
step. It localises but does not classify (error_class is always "none").

Scoring: `verify/scoring.py`, strict, as defined in the paper.
Rows keep their per-step Lean states so the scorer can re-decide them (vacuity gate,
malformed-step rule); a dead or unparsed row is a failure, not a free "no error". See evaluation/README.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.jsonio import load_jsonl, subsample, write_json_atomic, write_jsonl_atomic
from ..core.paths import backward_verification_jsonl
from ..core.registry import ModelEntry

__all__ = ["load_items", "solve", "evaluate", "bundle", "rebind"]


def _whitelist_render(item: dict) -> str:
    parts = [f"# Question\n{(item.get('context') or '').strip()}"]
    for t in item["theorems"]:
        steps = "\n".join(f"Step {s['n']}: {s['claim']}" for s in t["proof"]["steps"])
        parts.append(f"# Theorem {t['local_id']}\n{t['statement']}\n\n"
                     f"## Steps of its proof\n{steps}")
    return "\n\n".join(parts)


def bundle(item: dict) -> str:
    """The rendered item, byte-checked against an independent four-field whitelist."""
    from .adapter import bundle_text

    text = bundle_text(item)
    if text != _whitelist_render(item):
        raise SystemExit(f"LEAK: bundle_text no longer matches the whitelist on {item.get('id')}")
    return text


def load_items(*, limit: int = 0, seed: int | None = None,
               ids: list[str] | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl(backward_verification_jsonl())
    if ids:
        want = set(ids)
        rows = [r for r in rows if r["id"] in want]
    # Sorted by label: a prefix is all-correct. Stratify so a smoke run sees both halves.
    return subsample(rows, limit=limit or None, seed=seed, stratify_on="label")


def _gold(item: dict) -> dict[str, Any]:
    # Snapshotted onto the row: scoring reads the snapshot, never today's bench file.
    return {"label": item.get("label"), "target_local_id": item.get("target_local_id"),
            "error_step": item.get("error_step"),
            "error_class6": item.get("error_class6") or [],
            "local_ids": [t["local_id"] for t in item["theorems"]]}


def _stem(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))[:120]


def _prepare(item: dict, sessions: Path):
    from . import adapter

    bundle(item)                    # the prompt shows exactly the four whitelisted fields
    adapter.require_upstream()
    adapter.preflight_lean(True)
    sessions.mkdir(parents=True, exist_ok=True)
    return adapter


def _solve_cot(item: dict, entry: ModelEntry, sessions: Path, *, timeout: int,
               task: str = "fl_refute") -> dict[str, Any]:
    adapter = _prepare(item, sessions)
    row = adapter.run_item_session(item, backend=entry.backend, model=entry.alias,
                                   effort=entry.effort, do_lean=True,
                                   session_dir=str(sessions), timeout=timeout, task=task,
                                   mode="cot", entry=entry)
    out = _row(item, row, task)
    out.update({"n_tool_calls": row.get("n_tool_calls"), "tokens": row.get("tokens")})
    return out


def _solve_agent(item: dict, entry: ModelEntry, sessions: Path, *, timeout: int,
                 task: str = "fl_refute") -> dict[str, Any]:
    adapter = _prepare(item, sessions)
    from .llm import egress_audit

    egress_audit(reset=True)
    row = adapter.run_item_session(item, backend=entry.backend, model=entry.alias,
                                   effort=entry.effort, do_lean=True,
                                   session_dir=str(sessions), timeout=timeout, task=task)
    suspect = egress_audit(reset=True)
    out = _row(item, row, task)
    out.update({"egress_suspect": bool(suspect), "egress_suspect_reasons": suspect[:20]})
    return out


def _row(item: dict, row: dict[str, Any], task: str) -> dict[str, Any]:
    """The answer row both modes write: gold snapshot, prediction, per-step Lean states."""
    pred ={k: row.get(k) for k in ("has_error", "verdict", "target_theorem", "wrong_step")}
    pred["error_class"] = row.get("error_class") or "none"
    per_theorem = [{"local_id": th["local_id"],
                    "steps": [{"n": st["n"], "state": st["state"],
                               "formal_statement": st.get("formal_statement")}
                              for st in th["steps"]]}
                   for th in row.get("per_theorem") or []]
    return {"id": item["id"], "gold": _gold(item), "pred": pred,
            "parse_error": None if row.get("parse_ok", True) else "session output unparsed",
            "parse_ok": row.get("parse_ok"), "rc": row.get("session_rc"),
            "coverage": row.get("coverage"), "fl_task": row.get("task", task),
            "per_theorem": per_theorem,
            "state_counts": row.get("state_counts"), "formalize_prompt": "upstream",
            "fl_row": {k: v for k, v in row.items() if k not in ("per_theorem",)},
            "status": (f"has_error={pred['has_error']} at {pred['target_theorem']}/"
                       f"{pred['wrong_step']} cov={row.get('coverage')}")}


def solve(item: dict, entry: ModelEntry, mode: str, sessions: Path, *, timeout: int,
          fl_task: str = "fl_refute") -> dict[str, Any]:
    if mode == "cot":
        return _solve_cot(item, entry, sessions, timeout=timeout, task=fl_task)
    if mode == "agent":
        return _solve_agent(item, entry, sessions, timeout=timeout, task=fl_task)
    raise ValueError(f"verify runs in agent or cot mode, not {mode!r}")


def _saved_output(sessions: Path, item_id: str, backend: str) -> tuple[int, str] | None:
    """(rc, final text) of a finished session, rebuilt from its saved stream the way the
    backend built it at run time; None when the session left no stream."""
    from ..core.streams import (claude_final_text, parse_claude_stream, parse_codex_stream,
                                text_from_codex_events)
    stem = str(item_id).split(".")[0]
    stream, meta = sessions / f"{stem}.stream.jsonl", sessions / f"{stem}.meta.json"
    if not stream.is_file() or not meta.is_file():
        return None
    out = stream.read_text(encoding="utf-8", errors="replace")
    if backend == "codex":
        text = text_from_codex_events(parse_codex_stream(out).events)
    else:
        text = claude_final_text(parse_claude_stream(out))
    rc = json.loads(meta.read_text(encoding="utf-8")).get("returncode")
    return (rc if rc is not None else -1), str(text)


def _answer_was_dropped(sessions: Path, item_id: str, backend: str) -> bool:
    """True when the session ended in more than one `result` turn and the last one alone, which
    is all the claude backend kept before 2026-10-20, parses to a different answer than all of
    them together (`streams.claude_final_text`)."""
    if backend == "codex":
        return False
    from ..core.streams import parse_claude_stream
    from . import adapter
    stream = sessions / f"{str(item_id).split('.')[0]}.stream.jsonl"
    if not stream.is_file():
        return False
    traj = parse_claude_stream(stream.read_text(encoding="utf-8", errors="replace"))
    texts = [r["result"] for r in traj.results if isinstance(r.get("result"), str) and r["result"]]
    if len(texts) < 2:
        return False
    return adapter.parse_session_batch(texts[-1]) != adapter.parse_session_batch("\n".join(texts))


def rebind(run_dir: Path, answers: list[dict[str, Any]], *, backend: str,
           log=print) -> tuple[list[dict[str, Any]], int]:
    """Re-check, from the saved session output and with no model call, every row that was read
    wrongly at run time, and return the new answers and the number of rows re-checked:

    - bare theorem numbers (`theorem=1`): before `adapter.bind_theorem_ids` (2026-10-18) those
      steps were dropped;
    - an answer in an earlier `result` turn: before `streams.claude_final_text` (2026-10-20) only
      the last turn was read, and a session that the CLI woke again after a background task
      timed out lost its whole answer (marked `rejoined` on the row).

    Other rows are returned untouched, as is a row's gold snapshot and its egress audit."""
    from . import adapter

    items = {it["id"]: it for it in load_items()}
    sessions = run_dir / "sessions"
    todo = []
    for i, a in enumerate(answers):
        if a["id"] not in items:
            continue
        saved = _saved_output(sessions, a["id"], backend)
        if saved is None or saved[0] != 0:
            continue
        why = []
        if "rebound" not in (a.get("fl_row") or {}):     # not yet bound (run before 2026-10-18)
            _, rebound = adapter.bind_theorem_ids(adapter.parse_session_batch(saved[1]),
                                                  items[a["id"]])
            if rebound:
                why.append(f"{len(rebound)} bare label(s)")
        if not a.get("rejoined") and _answer_was_dropped(sessions, a["id"], backend):
            why.append("answer in an earlier result turn")
        if why:
            todo.append((i, saved, why))
    if not todo:
        return answers, 0
    adapter.preflight_lean(True)
    out = list(answers)
    for k, (i, saved, why) in enumerate(todo, 1):
        a = answers[i]
        task = a.get("fl_task") or "fl_refute"
        row = adapter.run_item_session(items[a["id"]], do_lean=True, task=task,
                                       mode=a.get("mode") or "agent", replay=saved)
        new = _row(items[a["id"]], row, task)
        new.update({k2: a[k2] for k2 in ("gold", "egress_suspect", "egress_suspect_reasons",
                                         "n_tool_calls", "tokens", "wall_clock_s", "mode",
                                         "model", "rejoined")
                    if k2 in a})
        if any(w.startswith("answer in") for w in why):
            new["rejoined"] = True
        out[i] = new
        log(f"[rebind] {k}/{len(todo)} {a['id']:40s} {'; '.join(why)}  "
            f"cov {a.get('coverage')} -> {new['coverage']}  {a.get('status')} -> {new['status']}")
    return out, len(todo)


# ------------------------------------------------------------------ scoring
def evaluate(run_dir: Path, answers: list[dict[str, Any]], *, malf_rule: str = "region", parse_penalty: bool = True) -> dict[str, Any]:
    from .scoring import score_rows

    missing = [a["id"] for a in answers if a.get("per_theorem") is None]
    if missing:
        raise SystemExit(
            f"{len(missing)} row(s) carry no per-step Lean states (run before the strict "
            f"scorer existed?), e.g. {missing[:3]}. Re-run them; the strict rules decide from "
            f"those states, not from the pipeline's summary.")
    summary, per_item = score_rows(answers, malf_rule=malf_rule, parse_penalty=parse_penalty)
    summary = {"task": "verify", **summary,
               "agent_rows_egress_suspect": [a["id"] for a in answers if a.get("egress_suspect")],
               "cot_rows_with_tools": sum(1 for a in answers if a.get("mode") == "cot"
                                          and (a.get("n_tool_calls") or 0) > 0),
               # bare `theorem=1` labels bound to T1 (adapter.bind_theorem_ids); rows from
               # before 2026-10-18 were matched literally until `eval --rebind` re-checks them
               "rows_rebound": sum(1 for a in answers if (a.get("fl_row") or {}).get("rebound")),
               "rows_literal_binding": sum(1 for a in answers
                                           if "rebound" not in (a.get("fl_row") or {})),
               # answer read from an earlier `result` turn (streams.claude_final_text, 2026-10-20)
               "rows_rejoined": sum(1 for a in answers if a.get("rejoined"))}
    write_jsonl_atomic(run_dir / "scores.jsonl", per_item)
    write_json_atomic(run_dir / "summary.json", summary)
    return summary
