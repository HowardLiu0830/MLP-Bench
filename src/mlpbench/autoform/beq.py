"""BEq -- Bidirectional Extended Definitional Equivalence (Liu et al., ICLR 2025).

    s_P <-U s_Q   iff  s_P is reachable from s_Q using only the tactics in R
    s_P ~_B s_Q   iff  s_P <-U s_Q  and  s_Q <-U s_P

Each direction: assume the source with `sorry`, try `exact?` (it must name the source),
and if that fails sample n tactic scripts from a model. A script counts only if every
step is headed by a tactic in R, it compiles with no error and no `sorry`, and it names
the source. Ported from the paper's original BEq script; the sampler goes through
this package's backends (`core.chat` for one-shot, `core.backends.run_agent` for agentic).

Defaults: both directions, the sampler with Lean access (agentic), n = 4, and the three
soundness gates ON (strip, vacuous-source, bridge; see `_gates`). Each one inspects the
accepted script itself:

  strip      remove the source's declaration and recompile the unchanged script; if it
             still closes the goal it never used the source -> rejected.
  vacuous    keep the source's binders and hypotheses but make it conclude only `True`, and
             recompile the unchanged script; if it still closes the goal it never used what
             the source says (a nominal `have := src ...`) -> rejected.
  bridge     an LLM screen (`bridge_model`) inspects every accepted script for substantive
             mathematics absent from the source; flagged -> rejected.

Every accepted direction -- `exact?` included -- passes through the gates before it counts,
so `beq`, `beq_exact_only`, `relation` and every rate built on them are post-gate. Each
direction records which gates ran (`gates_enabled`), what each gate said per candidate
(`gates`), which candidates a gate rejected (`gate_rejected`), and whether anything
compiled before gating (`ok_before_gates`, a diagnostic, never a score).

`BEqConfig(strip_gate=False, ...)` or `mlpbench eval --no-gates` turns the gates off; ungated
BEq is not comparable with gated BEq. In lazy mode `exact?` is gated before the sampler is
skipped, so an `exact?` that a gate rejects does not stop the sampler from trying.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import backends as B
from ..core.chat import complete
from ..core.jsonio import trim_log
from ..core.paths import lean_project as default_lean_project
from ..core.paths import workspace_root
from ..core.registry import ModelEntry
from . import leansurgery as A
from .beq_prompts import (BEQ_TACTICS, build_bridge_prompt,
                          build_prompt)

__all__ = ["BEqConfig", "beq_pair", "check_direction", "script_respects_R", "clean_script",
           "gates_enabled"]

BANNED = re.compile(r"\b(sorry|admit|axiom|native_decide|decide)\b")


@dataclass
class BEqConfig:
    sampler: ModelEntry | None = None   # None: `exact?` only
    agentic: bool = True                # give the sampler the exact file plus Lean
    n: int = 4
    sampler_timeout: int = 900
    jobs: int = 4
    lean_project: Path | None = None
    timeout: int = 1200                 # per Lean invocation
    heartbeats: int = 400000
    work_dir: Path | None = None
    lazy: bool = True                   # sample only after `exact?` fails (the paper's order)
    allow_decline: bool = True
    both_directions: bool = True        # compute d2 even when d1 fails -> 4-way relation
    enforce_tactics: bool = True
    strip_gate: bool = True             # Appendix B.3
    vacuous_gate: bool = True
    bridge_gate: bool = True
    bridge_model: ModelEntry | None = None  # the bridge screen's LLM; None -> the sampler
    bridge_timeout: int = 600
    raw_dir: Path | None = None

    @property
    def project(self) -> Path:
        return Path(self.lean_project) if self.lean_project else default_lean_project()


_HEAD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_'?!]*")
# A `by` that opens a tactic block mid-line: `exact (by simp)`, `have h : P := by linarith`.
_INLINE_BY_RE = re.compile(r"(?<![\w.'])by(?![\w'])\s*(\S.*)?$")


def _step_heads(step: str) -> list[str | None]:
    """The tactic heading `step`, then the head of every `by` block opened inside it
    (None where a block opens on something that is not a tactic name)."""
    heads: list[str | None] = []
    rest = step
    while True:
        m = _HEAD_RE.match(rest)
        heads.append(m.group(0) if m else None)
        nested = _INLINE_BY_RE.search(rest[m.end():] if m else rest)
        if not nested:
            return heads
        rest = (nested.group(1) or "").lstrip("({·•").strip()
        if not rest:            # `... := by` at end of line: the block is the next lines
            return heads


def _logical_steps(script: str) -> list[str]:
    """Join a term's continuation lines onto the step they belong to.

    A line indented deeper than the content column of the step above it continues that
    step's term (`apply foo\n    (x := a)`), unless the step above ends in `by`, which
    opens a nested tactic block whose lines are steps of their own."""
    steps: list[str] = []
    col = -1                                  # content column of the current step
    opens_block = True
    for line in script.splitlines():
        code = line.split("--", 1)[0].rstrip()
        if not code.strip():
            continue
        indent = len(code) - len(code.lstrip())
        if steps and not opens_block and indent > col:
            steps[-1] += " " + code.strip()
        else:
            bullet = re.match(r"\s*([·•]\s*)?", code)
            col = bullet.end() if bullet else indent
            steps.append(code.strip())
        opens_block = bool(re.search(r"(?<![\w.'])by$", steps[-1]))
    return steps


def script_respects_R(script: str) -> tuple[bool, str]:
    """Is every tactic of `script` -- inline `by` blocks included -- a primitive in R?"""
    if not script.strip():
        return False, "empty script"
    if BANNED.search(script):
        return False, "banned tactic"
    for logical in _logical_steps(script):
        for raw in re.split(r"<;>|(?<![:<>]);", logical):
            step = re.sub(r"^[·•\-\*]\s*", "", raw.strip())
            step = step.strip("{} \t")
            if not step:
                continue
            for i, head in enumerate(_step_heads(step)):
                if head is None:
                    what = "unparsable step" if i == 0 else "unparsable inline `by` block"
                    return False, f"{what}: {raw.strip()[:60]}"
                if head not in BEQ_TACTICS:
                    where = "tactic outside R" if i == 0 else "inline `by` tactic outside R"
                    return False, f"{where}: {head}"
    return True, ""


AGENTIC_FILE = "check.lean"

AGENTIC_NOTE = """

A Lean workspace is available and `{f}` already holds EXACTLY the file the grader
will compile: the source theorem `{source}` in scope, and the check declaration
`theorem beq_chk_llm0 ... := by` with its tactic block replaced by `sorry`.

Replace that `sorry`, then compile:

    cd {proj} && lake env lean --json $OLDPWD/{f}

Use the compiler as much as you need: compile early and often, read the errors,
inspect the goal state, try another bridge when one fails. Several genuinely
different attempts within the rules are expected before you conclude anything;
do not reply `  fail` on reading the statements alone. Change nothing except the
tactic block.

Once `beq_chk_llm0` compiles with no error and no `sorry`, check that the SOURCE
carries the proof: copy `{f}` to `nosrc.lean`, delete the declaration of `{source}`
from the copy (its whole `namespace ... end` block if it lives in one), and compile
the copy. If `beq_chk_llm0` still compiles there, your script does not use the
SOURCE -- rework it, or reply `  fail` if it cannot be made to.

When you are done, reply with the tactic block ALONE: the REPLY is what is graded
and recompiled, not the file.
"""


def clean_script(text: str) -> str:
    """Strip the wrappers models add back around a bare tactic block."""
    text = (text or "").strip()
    fence = re.search(r"```(?:lean4?|lean)?\s*\n(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    text = re.sub(r"^\s*theorem[^\n]*:=\s*by\s*\n", "", text)
    text = re.sub(r"^\s*by\s*\n", "", text)
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(ln if ln.startswith("  ") else "  " + ln.lstrip() for ln in lines)


def _sample_once(cfg: BEqConfig, system: str, user: str, seed: str | None,
                 stem: str) -> tuple[str, dict[str, Any]]:
    entry = cfg.sampler
    assert entry is not None
    if cfg.agentic and seed is not None and entry.backend in ("claude", "codex"):
        root = workspace_root() / "beq"
        root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="beq_ag_", dir=str(root)))
        try:
            (scratch / AGENTIC_FILE).write_text(seed, encoding="utf-8")
            run = B.run_agent(entry, system, user, workspace=scratch,
                              timeout=cfg.sampler_timeout, raw_dir=cfg.raw_dir, stem=stem,
                              lean_project=cfg.project)
            return run.final_text or "", {"n_lean_calls": run.n_lean_calls,
                                          "n_commands": run.n_shell_calls,
                                          "agent_timed_out": run.timed_out,
                                          "egress_suspect": run.egress_suspect}
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    res = complete(entry, system, user, timeout=cfg.sampler_timeout, raw_dir=cfg.raw_dir,
                   stem=stem)
    return res.text or "", {"n_lean_calls": 0, "n_commands": 0,
                            "agent_timed_out": res.timed_out}


def sample_scripts(cfg: BEqConfig, *, source_name: str, source_stmt: str, goal_stmt: str,
                   context: str, workspace_seed: str | None, stem: str) -> list[dict[str, Any]]:
    """n independent samples of T, each labelled with whether it survived R."""
    if cfg.sampler is None or cfg.n <= 0:
        return []
    system, user = build_prompt(source_name=source_name, source_stmt=source_stmt,
                                goal_stmt=goal_stmt, context=context,
                                allow_decline=cfg.allow_decline)
    agentic = bool(cfg.agentic and workspace_seed and cfg.sampler.backend in ("claude", "codex"))
    if agentic:
        user += AGENTIC_NOTE.format(f=AGENTIC_FILE, proj=cfg.project, source=source_name)

    def one(i: int) -> dict[str, Any]:
        try:
            raw, meta = _sample_once(cfg, system, user, workspace_seed if agentic else None,
                                     f"{stem}.s{i}")
        except Exception as exc:  # noqa: BLE001 - a dead sample must not kill the pair
            return {"raw": "", "script": "", "valid": False, "reason": f"llm error: {exc}"}
        if not raw.strip():
            return {"raw": "", "script": "", "valid": False, "reason": "empty reply", **meta}
        script = clean_script(raw)
        if script.strip() == "fail":
            return {"raw": trim_log(raw, 2000), "script": "", "valid": False,
                    "reason": "model declined", **meta}
        ok, reason = (True, "") if not cfg.enforce_tactics else script_respects_R(script)
        return {"raw": trim_log(raw, 2000), "script": script, "valid": bool(ok and script),
                "reason": reason, **meta}

    with ThreadPoolExecutor(max_workers=max(1, min(cfg.jobs, cfg.n))) as pool:
        return list(pool.map(one, range(cfg.n)))


def statement_text(t: A.Target) -> str:
    return (t.decl.sig if t.decl else "").strip()


def context_text(t: A.Target, limit: int = 4000) -> str:
    keep = [c.text for c in A.parse_chunks(t.body)
            if c.kind in ("def", "abbrev", "structure", "class", "instance", "inductive",
                          "variable", "open")]
    return trim_log("".join(keep), limit)


def _with_heartbeats(text: str, heartbeats: int) -> str:
    if not heartbeats:
        return text
    ie = text.find("\n\n")
    return text[: ie + 1] + f"set_option maxHeartbeats {heartbeats}\n" + text[ie + 1:]


def _verdicts(run: A.LeanRun, cands: list[A.Candidate], src_full: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    short = src_full.split(".")[-1]
    for cand in cands:
        require = src_full if cand.key == "exact" else None
        v = A.candidate_verdict(run.messages, cand, require_name=require)
        if cand.key != "exact" and v["ok"]:
            # "at least one successful proof that USES s_Q"
            v["used_source"] = (src_full in cand.script) or bool(
                re.search(r"(?<![\w.])" + re.escape(short) + r"(?![\w.])", cand.script))
            v["ok"] = v["ok"] and v["used_source"]
        if run.timed_out:
            v["ok"] = False
            v["errors"] = ["lean timeout"]
        v["script"] = cand.script
        out[cand.key] = v
    return out


def lean_ran(run: A.LeanRun) -> bool:
    """Did Lean elaborate the file? A crash (no space, OOM, a killed process) exits non-zero
    with no error message, which `candidate_verdict` would read as a clean compile -- and in a
    gate a clean compile is a rejection."""
    return not run.timed_out and (run.returncode == 0 or any(
        m.get("severity") == "error" for m in run.messages))


def compile_without_source(goal: A.Target, source: A.Target, src_full: str, cfg: BEqConfig,
                           batch: list[A.Candidate], stem: str, *,
                           print_axioms: bool = False) -> A.LeanRun:
    """The direction file with the source's declaration removed. `print_axioms` adds a
    `#print axioms` after each candidate (read back by `proves_without_source`)."""
    text = _with_heartbeats(A.build_direction_file(goal, source, batch), cfg.heartbeats)
    text, _ = drop_source_decl(text, src_full)
    if print_axioms:
        for c in batch:
            i = text.index(f"theorem beq_chk_{c.key} ")
            j = text.index(c.script, i) + len(c.script)
            text = text[:j] + f"\n\n#print axioms beq_chk_{c.key}\n" + text[j:]
    A._fix_spans(text, batch)
    return A.run_lean(text, lean_project=cfg.project, timeout=cfg.timeout,
                      work_dir=cfg.work_dir, stem=stem)


def proves_without_source(goal: A.Target, source: A.Target, src_full: str, cfg: BEqConfig,
                          cand: A.Candidate, stem: str) -> tuple[bool | None, str]:
    """Does `cand` close the goal with the source's declaration removed? Compiled ALONE: a
    failed declaration stays in the environment as a sorry'd copy of the goal, so in a batch
    `exact?` can close one candidate with another. A proof that depends on `sorryAx` (a
    sorry'd sibling or lemma in the benchmark's context) does not count. None: Lean did not
    run."""
    run = compile_without_source(goal, source, src_full, cfg, [cand], stem, print_axioms=True)
    if not lean_ran(run):
        return None, trim_log(run.raw, 300)
    ax = " ".join(m.get("data") or "" for m in run.messages
                  if "depends on axioms" in (m.get("data") or ""))
    if "sorryAx" in ax:
        return False, "depends on sorryAx"
    return bool(A.candidate_verdict(run.messages, cand)["ok"]), ""


def _exact_term(suggestion: str) -> str | None:
    """The tactic `exact?` reported ("Try this: [apply] exact ..."), or None if absent or cut."""
    s = suggestion.split("Try this:", 1)[-1].strip()
    s = re.sub(r"^\[\w+\]\s*", "", s)
    return s if s.startswith(("exact ", "refine ")) and "\n" not in s and "…" not in s else None


def proves_with_vacuous_source(goal: A.Target, source: A.Target, cfg: BEqConfig,
                               cand: A.Candidate, stem: str) -> tuple[bool | None, str]:
    """Does `cand` still close the goal when the source keeps its binders and hypotheses but
    concludes only `True`? Compiled ALONE (see `proves_without_source`). None: not
    applicable, or Lean did not run."""
    vac = source.with_true_conclusion()
    if vac is None:
        return None, "the source's signature has no conclusion to replace"
    text = _with_heartbeats(A.build_direction_file(goal, vac, [cand]), cfg.heartbeats)
    A._fix_spans(text, [cand])
    run = A.run_lean(text, lean_project=cfg.project, timeout=cfg.timeout,
                     work_dir=cfg.work_dir, stem=stem)
    if not lean_ran(run):
        return None, trim_log(run.raw, 300)
    return bool(A.candidate_verdict(run.messages, cand)["ok"]), ""


def check_direction(goal: A.Target, source: A.Target, cfg: BEqConfig, *,
                    stem: str = "beq") -> dict[str, Any]:
    """One unidirectional implication: `exact?` first, then sampled scripts."""
    # Must match the name build_direction_file emits: a shared preamble puts the source
    # at top level as `beq_src`, otherwise it lives in the BEqSrc namespace.
    src_full = (A.SRC_LOCAL if A.shares_preamble(goal, source)
                else A.SRC_NS + "." + source.full_name())

    def compile_batch(batch: list[A.Candidate]) -> A.LeanRun:
        text = _with_heartbeats(A.build_direction_file(goal, source, batch), cfg.heartbeats)
        A._fix_spans(text, batch)
        return A.run_lean(text, lean_project=cfg.project, timeout=cfg.timeout,
                          work_dir=cfg.work_dir, stem=stem)

    last_batch: list[A.Candidate] = []

    def judge(batch: list[A.Candidate]) -> tuple[dict, A.LeanRun]:
        nonlocal last_batch
        last_batch = batch
        run = compile_batch(batch)
        verdicts = _verdicts(run, batch, src_full)
        # A script that fails to PARSE swallows every declaration after it.
        if A.has_parse_error(run.messages) and len(batch) > 1:
            verdicts = {}
            for cand in batch:
                solo = [A.Candidate(cand.key, cand.script)]
                verdicts.update(_verdicts(compile_batch(solo), solo, src_full))
        return verdicts, run

    def sample() -> list[dict[str, Any]]:
        seed = None
        if cfg.agentic:
            seed = _with_heartbeats(
                A.build_direction_file(goal, source, [A.Candidate("llm0", "  sorry")]),
                cfg.heartbeats)
        return sample_scripts(cfg, source_name=src_full, source_stmt=statement_text(source),
                              goal_stmt=statement_text(goal),
                              context=context_text(source) + "\n" + context_text(goal),
                              workspace_seed=seed, stem=stem)

    samples: list[dict[str, Any]] = []
    exact_cand = [A.Candidate("exact", "  exact?")]
    pre_gate: dict[str, bool] = {}

    def gate(vs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        pre_gate.update({k: bool(v.get("ok")) for k, v in vs.items()})
        return _gates(vs, goal, source, src_full, cfg, stem)

    if cfg.lazy:
        verdicts, run = judge(exact_cand)
        # Gate `exact?` BEFORE deciding to skip the sampler: an `exact?` a gate rejects is
        # no evidence, and must not stand in for the search it would otherwise cut short.
        gate_log = gate(verdicts)
        if not verdicts["exact"]["ok"] and cfg.sampler is not None:
            samples = sample()
            batch = [A.Candidate(f"llm{i}", s["script"]) for i, s in enumerate(samples)
                     if s["valid"]]
            if batch:
                more, run = judge(batch)
                gate_log.update(gate(more))
                verdicts.update(more)
    else:
        samples = sample() if cfg.sampler is not None else []
        batch = exact_cand + [A.Candidate(f"llm{i}", s["script"])
                              for i, s in enumerate(samples) if s["valid"]]
        verdicts, run = judge(batch)
        gate_log = gate(verdicts)

    ok = any(v["ok"] for v in verdicts.values())
    return {
        "beq": bool(ok),                                    # post-gate
        "ok_before_gates": any(pre_gate.values()),          # diagnostic, never a score
        "gates_enabled": gates_enabled(cfg),
        "gates": gate_log,
        "gate_rejected": sorted(f"{k}:{v['gate_failed']}" for k, v in verdicts.items()
                                if v.get("gate_failed")),
        "context_errors": A.outside_span_errors(run.messages, last_batch),
        "exact_ok": bool(verdicts.get("exact", {}).get("ok")),
        "llm_ok": bool(any(v["ok"] for k, v in verdicts.items() if k != "exact")),
        "winning": sorted(k for k, v in verdicts.items() if v["ok"]),
        "n_sampled": len(samples),
        "n_valid": sum(1 for s in samples if s["valid"]),
        # Zero Lean calls across every agentic sample is a tool-free run wearing the label.
        "agent_lean_calls": sum(s.get("n_lean_calls") or 0 for s in samples),
        "agent_timed_out": sum(1 for s in samples if s.get("agent_timed_out")),
        "reject_reasons": sorted({s["reason"] for s in samples if not s["valid"] and s["reason"]}),
        "candidates": verdicts,
        "timed_out": run.timed_out,
        "source_theorem": src_full,
    }


# --------------------------------------------------------------------------- gates
def gates_enabled(cfg: BEqConfig) -> dict[str, Any]:
    bridge = cfg.bridge_model or cfg.sampler
    return {"strip": bool(cfg.strip_gate), "vacuous": bool(cfg.vacuous_gate),
            "bridge": bool(cfg.bridge_gate),
            "bridge_model": (bridge.alias if bridge and cfg.bridge_gate else None)}


PROJ_RE = re.compile(r"\)\s*\.\s*(1|2|left|right)\b")
_VERDICT_RE = re.compile(r"\{[^{}]*\"verdict\"[^{}]*\}", re.S)


def _source_pat(src_full: str) -> tuple[str, re.Pattern[str]]:
    short = src_full.split(".")[-1]
    return short, re.compile(r"(?<![\w.])" + re.escape(short) + r"(?![\w'])")


def drop_source_decl(text: str, src_full: str) -> tuple[str, int]:
    lines = text.splitlines()
    if src_full.startswith(A.SRC_NS + "."):
        start = end = None
        for i, ln in enumerate(lines):
            if re.match(rf"\s*namespace\s+{A.SRC_NS}\b", ln):
                start = i
            elif start is not None and re.match(rf"\s*end\s+{A.SRC_NS}\b", ln):
                end = i
                break
        if start is None or end is None:
            return text, 0
        return "\n".join(lines[:start] + lines[end + 1:]), end - start + 1
    short = src_full.split(".")[-1]
    start = next((i for i, ln in enumerate(lines)
                  if re.match(rf"\s*theorem\s+{re.escape(short)}\b", ln)), None)
    if start is None:
        return text, 0
    end = start
    while end < len(lines) and not re.search(r"(?<![\w.])sorry(?![\w'])", lines[end]):
        end += 1
    return "\n".join(lines[:start] + lines[end + 1:]), end - start + 1


def conjunct_flag(script: str, goal_sig: str, src_full: str) -> str:
    if "∧" not in (goal_sig or ""):
        return ""
    _, pat = _source_pat(src_full)
    hits = [ln for ln in script.splitlines() if pat.search(ln)]
    if len(hits) == 1 and PROJ_RE.search(hits[0]):
        return f"source used once, through a projection: {hits[0].strip()[:120]}"
    return ""


def bridge_screen(script: str, *, goal: A.Target, source: A.Target, src_full: str,
                  cfg: BEqConfig, stem: str, attempts: int = 2) -> dict[str, Any]:
    """The LLM screen: does an accepted script supply mathematics absent from the source?
    A screen that never returns a verdict rejects (the gates favour precision)."""
    entry = cfg.bridge_model or cfg.sampler
    system, user = build_bridge_prompt(
        source_name=src_full, source_stmt=statement_text(source),
        goal_stmt=statement_text(goal), script=script,
        context=context_text(source) + "\n" + context_text(goal))
    last = ""
    for i in range(attempts):
        try:
            res = complete(entry, system, user, timeout=cfg.bridge_timeout, raw_dir=cfg.raw_dir,
                           stem=f"{stem}.bridge{i}")
            last = res.text or ""
        except Exception as exc:  # noqa: BLE001 - retried, then counted as a rejection
            last = f"error: {exc}"
            continue
        m = _VERDICT_RE.search(last)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        verdict = str(obj.get("verdict", "")).strip().lower()
        if verdict in ("pass", "flag"):
            return {"flagged": verdict == "flag", "reason": str(obj.get("reason", ""))[:300]}
    return {"flagged": True, "reason": "no verdict from the screen: " + trim_log(last, 200),
            "error": True}


def _gates(verdicts, goal, source, src_full, cfg, stem) -> dict[str, Any]:
    """In order of cost: strip and vacuous (one compile per candidate each), then bridge (one
    LLM call per candidate). A candidate counts only if it survives all three, so the order
    changes the cost, never the verdict."""
    log: dict[str, dict[str, Any]] = {}
    for key, v in verdicts.items():
        if not v.get("ok"):
            continue
        g: dict[str, Any] = {}
        script = v.get("script") or ""
        flag = conjunct_flag(script, (goal.decl.after_name if goal.decl else "") or "", src_full)
        if flag:
            g["conjunct_warning"] = flag
        # `exact?` is checked through the term it found: re-running it would search afresh
        term = _exact_term(v.get("suggestion") or "") if key == "exact" else None
        checked = "  " + term if term else script
        if cfg.strip_gate:
            probe = A.Candidate(key + "_strip", checked)
            if drop_source_decl(A.build_direction_file(goal, source, [probe]), src_full)[1] == 0:
                g["strip"] = "not applicable"
            else:
                ok, why = proves_without_source(goal, source, src_full, cfg, probe,
                                                f"{stem}.strip")
                if ok is None:
                    g["strip"] = "inconclusive (Lean did not run: " + why + ")"
                elif ok:
                    g["strip"] = "REJECTED: compiles with the source removed"
                    v.update(ok=False, gate_failed="strip")
                else:
                    g["strip"] = "passed"
        if cfg.vacuous_gate and v["ok"]:
            probe = A.Candidate(key + "_vac", checked)
            ok, why = proves_with_vacuous_source(goal, source, cfg, probe, f"{stem}.vac")
            if ok is None:
                g["vacuous"] = "inconclusive (" + why + ")"
            elif ok:
                g["vacuous"] = "REJECTED: compiles when the source concludes only True"
                v.update(ok=False, gate_failed="vacuous")
            else:
                g["vacuous"] = "passed"
        if cfg.bridge_gate and v["ok"]:
            shown = script
            if key == "exact" and v.get("suggestion"):
                shown = script + "\n  -- exact? found: " + v["suggestion"].replace("\n", " ")
            b = bridge_screen(shown, goal=goal, source=source, src_full=src_full, cfg=cfg,
                              stem=f"{stem}.{key}")
            if b["flagged"]:
                g["bridge"] = "REJECTED: " + b["reason"]
                v.update(ok=False, gate_failed="bridge", bridge_reason=b["reason"])
            else:
                g["bridge"] = "passed: " + b["reason"]
        if g:
            log[key] = g
    return log


def beq_pair(ref: A.Target, pred: A.Target, cfg: BEqConfig, *, stem: str = "pair") -> dict[str, Any]:
    """s_ref ~_B s_pred: both unidirectional implications must hold."""
    d1 = check_direction(ref, pred, cfg, stem=stem + ".ref_from_pred")
    if not d1["beq"] and not cfg.both_directions:
        d2: dict[str, Any] = {"skipped": True, "beq": False, "exact_ok": False, "llm_ok": False,
                              "ok_before_gates": False, "gates_enabled": gates_enabled(cfg),
                              "gates": {}, "gate_rejected": [],
                              "winning": [], "n_sampled": 0, "n_valid": 0, "candidates": {}}
    else:
        d2 = check_direction(pred, ref, cfg, stem=stem + ".pred_from_ref")
    return {
        "beq": bool(d1["beq"] and d2["beq"]),
        "beq_exact_only": bool(d1["exact_ok"] and d2["exact_ok"]),
        "pred_implies_ref": bool(d1["beq"]),
        "ref_implies_pred": bool(d2["beq"]),
        "relation": ("equivalent" if d1["beq"] and d2["beq"] else
                     "pred_stronger" if d1["beq"] else
                     "pred_weaker" if d2["beq"] else "unrelated"),
        "ref_from_pred": d1,
        "pred_from_ref": d2,
    }
