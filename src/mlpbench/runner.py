"""The loop every task shares: pick items, solve them in parallel, append answers.

A run directory is self-describing:

    runs/<task>/<mode>__<model>__<timestamp>/
        config.json      what was run (task, mode, model, limit/seed, item ids, versions)
        answers.jsonl    one row per item, keyed by `id`; appended as items finish
        sessions/        per-item prompt, raw stream and meta, for auditing

`mlpbench eval` reads only that directory (plus the benchmark), so scoring can be re-run
without re-running the model.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .core.cpu import pin_process, workers as auto_workers
from .core.jsonio import Appender, load_jsonl, write_json_atomic
from .core.paths import runs_root
from .core.registry import ModelEntry, describe

__all__ = ["RunSpec", "run_dir_for", "run_items", "load_run"]

MODES = ("cot", "agent")


@dataclass
class RunSpec:
    task: str
    mode: str
    entry: ModelEntry
    out: Path
    workers: int | str = "auto"
    resume: bool = False
    extra: dict[str, Any] | None = None


def run_dir_for(task: str, mode: str, entry: ModelEntry, out: Path | None) -> Path:
    if out:
        return Path(out)
    model = entry.alias.replace("/", "-").replace(":", "-")
    return runs_root() / task / f"{mode}__{model}__{time.strftime('%Y%m%d-%H%M%S')}"


# Settings that change what the model is shown; a resumed run must not mix two of them.
_INPUT_KEYS = ("nl", "no_nl", "fl_task", "sibling_targets")
# What a run without a key used, so resuming it with the default still works.
_INPUT_DEFAULTS = {"sibling_targets": "shown"}


def _check_resume_inputs(out: Path, spec: RunSpec) -> None:
    cfg = out / "config.json"
    if not (spec.resume and cfg.exists()):
        return
    old = json.loads(cfg.read_text(encoding="utf-8"))
    new = spec.extra or {}
    bad = [f"{k}: {old.get(k, _INPUT_DEFAULTS.get(k))!r} -> {new[k]!r}" for k in _INPUT_KEYS
           if k in new and old.get(k, _INPUT_DEFAULTS.get(k)) != new[k]]
    if bad:
        raise SystemExit(f"--resume would mix input settings in {out} ({'; '.join(bad)}); "
                         "start a new run directory")


def run_items(spec: RunSpec, items: Sequence[Any], solve: Callable[[Any, Path], dict[str, Any]],
              *, item_id: Callable[[Any], str]) -> Path:
    """Solve every item not already answered; returns the run directory."""
    if spec.mode == "agent" and spec.entry.is_http:
        raise SystemExit(
            f"{spec.entry.alias} is an HTTP model; it has no CLI agent. Use --mode cot, or the "
            f"agentic open-weights route `local:<hf-id>` (codex CLI via the LiteLLM proxy).")
    out = spec.out
    out.mkdir(parents=True, exist_ok=True)
    sessions = out / "sessions"
    mask = pin_process()
    _check_resume_inputs(out, spec)
    write_json_atomic(out / "config.json", {
        "task": spec.task, "mode": spec.mode, "model": describe(spec.entry),
        "n_items": len(items), "item_ids": [item_id(it) for it in items],
        "cpu_mask": f"{min(mask)}-{max(mask)}" if mask else None,
        "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(spec.extra or {}),
    })
    app = Appender(out / "answers.jsonl", id_key="id", resume=spec.resume)
    todo = [it for it in items if item_id(it) not in app.done]
    if len(todo) < len(items):
        print(f"[{spec.task}] resume: {len(items) - len(todo)} already answered")
    n_workers = auto_workers(spec.workers) if spec.mode == "agent" else (
        int(spec.workers) if str(spec.workers).isdigit() else 4)
    print(f"[{spec.task}] {len(todo)} items, mode={spec.mode}, model={spec.entry.alias}, "
          f"workers={n_workers}, out={out}", flush=True)

    def guarded(it: Any) -> dict[str, Any]:
        t0 = time.time()
        try:
            row = solve(it, sessions)
        except Exception as exc:  # one bad item must not cost the run
            row = {"error": f"{type(exc).__name__}: {exc}"}
        row.setdefault("id", item_id(it))
        row.setdefault("model", spec.entry.alias)
        row.setdefault("mode", spec.mode)
        row.setdefault("wall_clock_s", round(time.time() - t0, 1))
        return row

    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
        futs = {ex.submit(guarded, it): it for it in todo}
        for fut in cf.as_completed(futs):
            row = fut.result()
            app.write(row)
            done += 1
            tag = f"ERROR {row['error'][:70]}" if row.get("error") else row.get("status", "")
            print(f"[{spec.task}] {done}/{len(todo)} {row['id'][:56]:56s} {tag}", flush=True)
    print(f"[{spec.task}] wrote {out / 'answers.jsonl'}")
    return out


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = Path(run_dir)
    cfg_path, ans = run_dir / "config.json", run_dir / "answers.jsonl"
    if not ans.is_file():
        raise SystemExit(f"{run_dir} has no answers.jsonl -- is it a run directory?")
    import json

    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    return cfg, load_jsonl(ans)
