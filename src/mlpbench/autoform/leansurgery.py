from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.cpu import cpu_list, lean_threads
from ..core.jsonio import trim_log
from ..core.paths import lean_project as default_lean_project
from ..core.paths import workspace_root

DEFAULT_LEAN_CMD = "lake env lean --json"


def _project(p: Path | None) -> Path:
    return Path(p) if p is not None else default_lean_project()


_TYPECHECK_CACHE: dict[str, dict[str, Any]] = {}


OPENERS = "([{⟨⦃⟮"


CLOSERS = ")]}⟩⦄⟯"


DECL_KEYWORDS = (
    "theorem", "lemma", "def", "abbrev", "instance", "structure", "class",
    "inductive", "example", "opaque", "axiom",
)


MODIFIERS = (
    "noncomputable", "private", "protected", "nonrec", "partial", "unsafe",
    "scoped", "local", "@[",
)


OTHER_COMMANDS = (
    "namespace", "end", "section", "open", "variable", "variables", "universe",
    "set_option", "attribute", "macro", "macro_rules", "notation", "syntax",
    "deriving", "mutual", "import", "run_cmd", "#",
)


def split_imports(src: str) -> tuple[list[str], str]:
    imports: list[str] = []
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("import "):
            imports.append(stripped)
            i += 1
        elif not stripped or stripped.startswith("--"):
            i += 1
        else:
            break
    return imports, "\n".join(lines[i:])


def _strip_for_scan(text: str) -> str:
    out = list(text)
    i, n = 0, len(text)
    depth_block = 0
    while i < n:
        two = text[i : i + 2]
        if depth_block == 0 and two == "--":
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if two == "/-":
            depth_block += 1
            out[i] = out[i + 1] = " "
            i += 2
            continue
        if two == "-/" and depth_block:
            depth_block -= 1
            out[i] = out[i + 1] = " "
            i += 2
            continue
        if depth_block:
            if text[i] != "\n":
                out[i] = " "
            i += 1
            continue
        if text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == "\\":
                    j += 1
                j += 1
            for k in range(i, min(j + 1, n)):
                if text[k] != "\n":
                    out[k] = " "
            i = j + 1
            continue
        i += 1
    return "".join(out)


def find_toplevel(text: str, needle: str, *, start: int = 0, skip: str = "") -> int:
    scan = _strip_for_scan(text)
    depth = 0
    i = start
    n = len(scan)
    while i < n:
        c = scan[i]
        if c in OPENERS:
            depth += 1
        elif c in CLOSERS:
            depth -= 1
        elif depth == 0 and scan.startswith(needle, i):
            nxt = scan[i + len(needle) : i + len(needle) + 1]
            if not (skip and nxt and nxt in skip):
                return i
        i += 1
    return -1


@dataclass
class Chunk:
    start: int
    end: int
    kind: str
    name: str | None
    text: str


def parse_chunks(body: str) -> list[Chunk]:
    lines = body.splitlines(keepends=True)
    starts: list[int] = []
    offsets: list[int] = []
    off = 0
    for line in lines:
        offsets.append(off)
        stripped = line.lstrip()
        is_start = False
        if line and not line[0].isspace() and stripped:
            head = stripped.split(None, 1)[0] if stripped.split() else ""
            if (
                head in DECL_KEYWORDS
                or head in OTHER_COMMANDS
                or head in MODIFIERS
                or stripped.startswith("@[")
                or stripped.startswith("/--")
                or stripped.startswith("/-")
                or head.startswith("#")
            ):
                is_start = True
        if is_start:
            starts.append(off)
        off += len(line)
    if not starts:
        return [Chunk(0, len(body), "", None, body)] if body.strip() else []
    if starts[0] != 0 and body[: starts[0]].strip():
        starts.insert(0, 0)

    chunks: list[Chunk] = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(body)
        text = body[s:e]
        kind, name = _chunk_kind_and_name(text)
        chunks.append(Chunk(s, e, kind, name, text))
    return chunks


_IDENT = r"[A-Za-z_À-ɏͰ-Ͽ℀-⅏][^\s(){}\[\],:;]*"


def _chunk_kind_and_name(text: str) -> tuple[str, str | None]:
    scan = _strip_for_scan(text)
    pos = 0
    while True:
        m = re.match(r"\s*(@\[[^\]]*\]|/--.*?-/|/-.*?-/)\s*", scan[pos:], flags=re.S)
        if m:
            pos += m.end()
            continue
        m = re.match(r"\s*(" + "|".join(re.escape(w) for w in MODIFIERS if w != "@[") + r")\s+", scan[pos:])
        if m:
            pos += m.end()
            continue
        break
    m = re.match(r"\s*(" + "|".join(DECL_KEYWORDS + OTHER_COMMANDS) + r")\b", scan[pos:])
    if not m:
        return "", None
    kind = m.group(1)
    rest = text[pos + m.end() :]
    if kind in DECL_KEYWORDS:
        nm = re.match(r"\s*(" + _IDENT + r")", rest)
        return kind, (nm.group(1) if nm else None)
    if kind in ("namespace", "end", "open"):
        nm = re.match(r"\s*([^\s]+)", rest)
        return kind, (nm.group(1) if nm else None)
    return kind, None


@dataclass
class Decl:
    name: str
    kind: str
    text: str
    sig: str
    after_name: str
    conclusion: str


SORRY_BODY_RE = re.compile(r"^\s*(?:by\s+)?sorry\s*$", re.S)


def _proof_assign(text: str) -> int:
    i, best = 0, -1
    while True:
        j = find_toplevel(text, ":=", start=i)
        if j < 0:
            break
        if SORRY_BODY_RE.match(text[j + 2 :]):
            return j
        if best < 0:
            best = j
        i = j + 2
    return best


def parse_decl(chunk: Chunk) -> Decl | None:
    if chunk.kind not in DECL_KEYWORDS or not chunk.name:
        return None
    text = chunk.text
    i = _proof_assign(text)
    sig = (text[:i] if i >= 0 else text).rstrip()
    kw = re.search(r"\b" + re.escape(chunk.kind) + r"\b", sig)
    if not kw:
        return None
    nm = re.search(r"\b" + re.escape(chunk.name) + r"\b", sig[kw.end() :])
    if not nm:
        return None
    after_name = sig[kw.end() + nm.end() :]
    j = find_toplevel(after_name, ":", skip="=")
    conclusion = after_name[j + 1 :].strip() if j >= 0 else ""
    return Decl(chunk.name, chunk.kind, text, sig, after_name, conclusion)


def local_decl_names(body: str) -> set[str]:
    names: set[str] = set()
    for c in parse_chunks(body):
        if c.kind in DECL_KEYWORDS and c.name:
            names.add(c.name)
    return names


def enclosing_namespace(body: str, offset: int) -> str:
    # A scope stack, not a name list: `end Foo` closes whatever was opened last, and a
    # SECTION's `end` must not pop a namespace component (a `section X ... end X` inside
    # `namespace N` must not drop `N` from every later name).
    stack: list[tuple[str, list[str]]] = []
    for c in parse_chunks(body):
        if c.start >= offset:
            break
        if c.kind == "namespace" and c.name:
            stack.append(("namespace", c.name.split(".")))
        elif c.kind == "section":
            stack.append(("section", []))
        elif c.kind == "end" and stack:
            stack.pop()
    return ".".join(p for _, parts in stack for p in parts)


def qualify(text: str, names: set[str], prefix: str) -> str:
    if not names:
        return text
    pat = re.compile(
        r"(?<![\w.'])(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")(?![\w.'])"
    )
    return pat.sub(lambda m: prefix + m.group(1), text)


@dataclass
class Target:
    lean: str
    target_name: str
    imports: list[str] = field(default_factory=list)
    body: str = ""
    decl: Decl | None = None
    decl_start: int = 0
    decl_end: int = 0
    siblings: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, lean: str, target_name: str, all_targets: list[str] | None = None) -> "Target":
        imports, body = split_imports(lean)
        chunks = parse_chunks(body)
        hit = next((c for c in chunks if c.kind in ("theorem", "lemma") and c.name == target_name), None)
        if hit is None:
            raise KeyError(f"target `{target_name}` not found among {[c.name for c in chunks if c.name]}")
        decl = parse_decl(hit)
        if decl is None:
            raise ValueError(f"could not split the signature of `{target_name}`")
        sibs = [t for t in (all_targets or []) if t != target_name]
        return cls(lean, target_name, imports, body, decl, hit.start, hit.end, sibs)

    def full_name(self) -> str:
        ns = enclosing_namespace(self.body, self.decl_start)
        return f"{ns}.{self.target_name}" if ns else self.target_name

    def body_without_siblings(self) -> str:
        if not self.siblings:
            return self.body
        chunks = parse_chunks(self.body)
        keep = [c for c in chunks if not (c.kind in ("theorem", "lemma") and c.name in self.siblings)]
        return "".join(c.text for c in keep)

    def with_true_conclusion(self) -> "Target | None":
        """This target with its binders and hypotheses kept and its conclusion replaced by
        `True` (the vacuous-source gate). Siblings are dropped, as in a direction file. None
        when the signature has no top-level `:`."""
        d = self.decl
        j = find_toplevel(d.after_name, ":", skip="=")
        if j < 0:
            return None
        decl = f"{d.kind} {d.name}{d.after_name[:j].rstrip()} : True := sorry\n\n"
        lean = "\n".join(self.imports) + "\n\n" + self.body_with_target_replaced(decl)
        return Target.build(lean, self.target_name, [self.target_name])

    def body_with_target_replaced(self, replacement: str) -> str:
        chunks = parse_chunks(self.body)
        out: list[str] = []
        for c in chunks:
            if c.kind in ("theorem", "lemma") and c.name in self.siblings:
                continue
            if c.start == self.decl_start:
                out.append(replacement)
                continue
            out.append(c.text)
        return "".join(out)


SRC_NS = "BEqSrc"


@dataclass
class Candidate:
    key: str
    script: str
    line_start: int = 0
    line_end: int = 0


SRC_LOCAL = "beq_src"


def shares_preamble(goal: Target, source: Target) -> bool:
    return goal.body_with_target_replaced("<T>") == source.body_with_target_replaced("<T>")


def build_direction_file(goal: Target, source: Target, candidates: list[Candidate]) -> str:
    imports = list(dict.fromkeys(goal.imports + source.imports)) or ["import Mathlib"]

    checks: list[str] = []
    for cand in candidates:
        decl = f"theorem beq_chk_{cand.key} {goal.decl.after_name.strip()} := by\n{cand.script}\n"
        checks.append(decl)

    if shares_preamble(goal, source):
        src_decl = (f"theorem {SRC_LOCAL} {source.decl.after_name.strip()} := sorry\n\n")
        body = goal.body_with_target_replaced(src_decl + "".join(d + "\n" for d in checks))
        text = "\n".join(imports) + "\n\n" + body
        if not text.endswith("\n"):
            text += "\n"
        _fix_spans(text, candidates)
        return text

    head = "\n".join(imports) + "\n\n"
    head += f"namespace {SRC_NS}\n" + source.body_without_siblings().rstrip() + f"\nend {SRC_NS}\n\n"

    prefix_lines = head.count("\n")
    replacement_parts: list[str] = []
    cursor = prefix_lines
    for cand, decl in zip(candidates, checks):
        cand.line_start = cursor + 1
        cursor += decl.count("\n")
        cand.line_end = cursor
        replacement_parts.append(decl + "\n")
        cursor += 1
    body = goal.body_with_target_replaced("".join(replacement_parts))

    text = head + body
    if not text.endswith("\n"):
        text += "\n"
    _fix_spans(text, candidates)
    return text


def _fix_spans(text: str, candidates: list[Candidate]) -> None:
    lines = text.splitlines()
    starts: dict[str, int] = {}
    for idx, line in enumerate(lines, start=1):
        m = re.match(r"theorem beq_chk_(\S+)", line)
        if m:
            starts[m.group(1)] = idx
    ordered = sorted(((starts.get(c.key, 0), c) for c in candidates), key=lambda p: p[0])
    for pos, (start, cand) in enumerate(ordered):
        cand.line_start = start
        nxt = ordered[pos + 1][0] if pos + 1 < len(ordered) else len(lines) + 1
        cand.line_end = max(start, nxt - 1)


@dataclass
class LeanRun:
    ok: bool
    returncode: int | None
    messages: list[dict[str, Any]]
    timed_out: bool = False
    raw: str = ""
    source_path: str | None = None


def run_lean(
    source: str,
    *,
    lean_project: Path | None = None,
    lean_cmd: str = DEFAULT_LEAN_CMD,
    timeout: int = 900,
    work_dir: Path | None = None,
    stem: str = "beq",
    keep: bool = False,
) -> LeanRun:
    work_dir = Path(work_dir or (workspace_root() / "autoform")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix=f"{stem}.", dir=work_dir)
    lean_file = Path(tmpdir) / "check.lean"
    lean_file.write_text(source)
    cmd = shlex.split(lean_cmd) + ["-j", str(lean_threads()), str(lean_file)]
    if shutil.which("taskset"):
        cmd = ["taskset", "-c", cpu_list()] + cmd
    env = os.environ.copy()
    env.setdefault("LEAN_NUM_THREADS", str(lean_threads()))
    try:
        proc = subprocess.run(
            cmd, cwd=str(_project(lean_project)), capture_output=True, text=True, timeout=timeout, env=env
        )
        raw = (proc.stdout or "") + (proc.stderr or "")
        rc: int | None = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        raw = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        rc, timed_out = None, True

    messages: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict) and "severity" in msg:
            messages.append(msg)

    if not keep:
        try:
            lean_file.unlink()
            Path(tmpdir).rmdir()
        except OSError:
            pass
    return LeanRun(
        ok=rc == 0,
        returncode=rc,
        messages=messages,
        timed_out=timed_out,
        raw=trim_log(raw),
        source_path=str(lean_file) if keep else None,
    )


PARSE_ERROR_RE = re.compile(r"unexpected (?:token|identifier|character)|; expected ")


def messages_in_span(messages: list[dict[str, Any]], line_start: int, line_end: int) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        pos = m.get("pos") or {}
        line = pos.get("line")
        if isinstance(line, int) and line_start <= line <= line_end:
            out.append(m)
    return out


def candidate_verdict(
    messages: list[dict[str, Any]], cand: Candidate, *, require_name: str | None = None
) -> dict[str, Any]:
    mine = messages_in_span(messages, cand.line_start, cand.line_end)
    errors = [m for m in mine if m.get("severity") == "error"]
    infos = [m for m in mine if m.get("severity") == "information"]
    sorried = [m for m in mine if "declaration uses 'sorry'" in (m.get("data") or "")]
    ok = not errors and not sorried
    used_source = True
    suggestion = ""
    if ok and require_name:
        suggestion = "\n".join((m.get("data") or "") for m in infos)
        used_source = require_name in suggestion
        ok = ok and used_source
    return {
        "key": cand.key,
        "ok": bool(ok),
        "errors": [trim_log(m.get("data") or "", 600) for m in errors[:3]],
        "suggestion": trim_log(suggestion, 600),
        "used_source": used_source,
        "line_start": cand.line_start,
        "line_end": cand.line_end,
    }


def has_parse_error(messages: list[dict[str, Any]]) -> bool:
    return any(
        m.get("severity") == "error" and PARSE_ERROR_RE.search(m.get("data") or "")
        for m in messages
    )


def typecheck_file(
    lean: str,
    *,
    lean_project: Path | None = None,
    lean_cmd: str = DEFAULT_LEAN_CMD,
    timeout: int = 900,
    work_dir: Path | None = None,
    stem: str = "typecheck",
) -> dict[str, Any]:
    import hashlib

    key = hashlib.sha1(lean.encode()).hexdigest()
    hit = _TYPECHECK_CACHE.get(key)
    if hit is not None:
        return hit
    run = run_lean(lean if lean.endswith("\n") else lean + "\n",
                   lean_project=lean_project, lean_cmd=lean_cmd, timeout=timeout,
                   work_dir=work_dir, stem=stem)
    errors = [m for m in run.messages if m.get("severity") == "error"]
    res = {
        "ok": bool(not errors and not run.timed_out),
        "n_errors": len(errors),
        "errors": [trim_log(m.get("data") or "", 400) for m in errors[:3]],
        "timed_out": run.timed_out,
    }
    _TYPECHECK_CACHE[key] = res
    return res


def outside_span_errors(messages: list[dict[str, Any]], cands: list[Candidate]) -> list[str]:
    out = []
    for m in messages:
        if m.get("severity") != "error":
            continue
        line = (m.get("pos") or {}).get("line")
        if not isinstance(line, int):
            continue
        if any(c.line_start <= line <= c.line_end for c in cands):
            continue
        out.append(trim_log(m.get("data") or "", 300))
    return out
