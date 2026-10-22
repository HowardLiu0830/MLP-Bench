from __future__ import annotations

import re

__all__ = [
    "strip_lean_comments", "normalize_ws", "extract_decl_statement", "namespace_paths",
    "build_axiom_check_file", "statement_preserved", "altered_definition",
    "DECL_RE", "SEG_RE", "REAL_SORRY_RE",
    "mark_targets",
]


REAL_SORRY_RE = re.compile(r"\b(?:sorry|admit)\b")


DECL_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:@\[[^\]]*\]\s*)?"
    r"(?:noncomputable\s+|private\s+|protected\s+|scoped\s+|local\s+)*"
    r"(?P<kind>theorem|lemma|corollary|def|abbrev|instance|example)\s+(?P<name>[^\s({\[:]+)",
    re.MULTILINE,
)


SEG_RE = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\]\s*)?"
    r"(?:noncomputable\s+|private\s+|protected\s+|scoped\s+|local\s+)*"
    r"(?P<kind>theorem|lemma|corollary|def|abbrev|instance|example|"
    r"section|namespace|end|variable|open|universe|set_option|attribute|include|omit)\b"
    r"(?:\s+(?P<name>[^\s({\[:]+))?",
    re.MULTILINE,
)


_DECL_KINDS = {"theorem", "lemma", "corollary", "def", "abbrev", "instance", "example"}


_UNPROTECTED_KINDS = {
    "open", "section", "namespace", "end", "universe", "set_option", "attribute",
    "include", "omit",
}


def strip_lean_comments(src: str) -> str:
    out: list[str] = []
    i, n = 0, len(src)
    depth = 0
    in_str = False
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if depth > 0:
            if c == "/" and nxt == "-":
                depth += 1
                i += 2
                continue
            if c == "-" and nxt == "/":
                depth -= 1
                i += 2
                continue
            i += 1
            continue
        if in_str:
            out.append(c)
            if c == "\\" and nxt:
                out.append(nxt)
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == "/" and nxt == "-":
            depth = 1
            i += 2
            continue
        if c == "-" and nxt == "-":
            j = src.find("\n", i)
            if j < 0:
                break
            i = j
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _top_level_assign(code: str) -> int:
    depth = 0
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if c in "([{⟨":
            depth += 1
        elif c in ")]}⟩":
            depth -= 1
        elif c == ":" and i + 1 < n and code[i + 1] == "=" and depth == 0:
            return i + 2
        i += 1
    return -1


def extract_decl_statement(src: str, lean_name: str) -> str:
    for m in DECL_RE.finditer(src):
        if m.group("name") == lean_name:
            start = m.start()
            depth = 0
            i = m.end()
            n = len(src)
            while i < n:
                c = src[i]
                if c in "([{":
                    depth += 1
                elif c in ")]}":
                    depth -= 1
                elif c == ":" and i + 1 < n and src[i + 1] == "=" and depth == 0:
                    return src[start:i].rstrip()
                i += 1
            return src[start : start + 600].rstrip()
    return f"(declaration `{lean_name}` not found)"


def mark_targets(src: str, targets: dict[str, str]) -> str:
    lines = src.split("\n")
    todo = dict(targets)
    out: list[str] = []
    for line in lines:
        m = DECL_RE.match(line)
        if m and m.group("name") in todo:
            label = todo.pop(m.group("name"))
            out.append(f"{m.group('indent')}-- TARGET \u00ab{label}\u00bb")
        out.append(line)
    if todo:
        raise ValueError(f"cannot mark targets not declared in the file: {sorted(todo)}")
    return "\n".join(out)

def namespace_paths(src: str) -> list[str]:
    stack: list[tuple[str, str | None]] = []
    paths: set[str] = set()
    for line in src.splitlines():
        s = line.strip()
        m = re.match(r"^namespace\s+([A-Za-z_][\w.]*)", s)
        if m:
            stack.append(("ns", m.group(1)))
            paths.add(".".join(n for k, n in stack if k == "ns" and n))
            continue
        if re.match(r"^section(\s|$)", s):
            stack.append(("sec", None))
            continue
        if re.match(r"^end(\s|$)", s):
            if stack:
                stack.pop()
    return sorted(paths)


def build_axiom_check_file(src: str, target_names: list[str]) -> str:
    if not src.endswith("\n"):
        src += "\n"
    nss = namespace_paths(src)
    out = [src, "\nsection PerTargetAxiomCheck"]
    out += [f"open {ns}" for ns in nss]
    for t in target_names:
        out.append(f"#print axioms {t}")
    out.append("end PerTargetAxiomCheck\n")
    return "\n".join(out)


def _extract_decl(src: str, lean_name: str) -> str:
    got = extract_decl_statement(src, lean_name)
    if got.startswith("(declaration") and "." in lean_name:
        got = extract_decl_statement(src, lean_name.rsplit(".", 1)[-1])
    return got


def statement_preserved(original_src: str, agent_src: str, lean_name: str) -> bool:
    orig = normalize_ws(strip_lean_comments(_extract_decl(original_src, lean_name)))
    got = normalize_ws(strip_lean_comments(_extract_decl(agent_src, lean_name)))
    if orig.startswith("(declaration") or got.startswith("(declaration"):
        return False
    return orig == got


def altered_definition(original_src: str, agent_src: str) -> str | None:
    comp_ws = normalize_ws(strip_lean_comments(agent_src))
    orig = strip_lean_comments(original_src)
    matches = list(SEG_RE.finditer(orig))
    pos = 0
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(orig)
        span = orig[start:end]
        kind = m.group("kind")
        name = m.group("name") or kind
        if kind in _UNPROTECTED_KINDS:
            continue
        if kind in _DECL_KINDS and REAL_SORRY_RE.search(span):
            sig_end = _top_level_assign(span)
            preserved = span[:sig_end] if sig_end > 0 else span[: REAL_SORRY_RE.search(span).start()]
            label = f"statement signature of `{name}`"
        else:
            preserved = span
            label = (f"definition `{name}`" if kind in _DECL_KINDS
                     else f"`{kind}{(' ' + name) if name != kind else ''}` block")
        preserved = normalize_ws(preserved)
        if not preserved:
            continue
        idx = comp_ws.find(preserved, pos)
        if idx < 0:
            idx = comp_ws.find(preserved, 0)
            if idx < 0:
                return label
        pos = idx + len(preserved)
    return None
