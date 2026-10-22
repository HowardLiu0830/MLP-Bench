from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "load_jsonl",
    "iter_jsonl",
    "write_jsonl_atomic",
    "write_json_atomic",
    "Appender",
    "subsample",
    "extract_json",
    "strip_code_fence",
    "trim_log",
]


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{p}:{line_no}: expected a JSON object")
            yield row


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl_atomic(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]],
                       *, sort_keys: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n")
    tmp.replace(p)


def write_json_atomic(path: str | os.PathLike[str], obj: Any, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    tmp.replace(p)


class Appender:

    def __init__(self, path: str | os.PathLike[str], *, id_key: str = "id", resume: bool = True):
        self.path = Path(path)
        self.id_key = id_key
        self.done: set[str] = set()
        if resume and self.path.exists():
            for row in iter_jsonl(self.path):
                key = row.get(id_key)
                if key is not None:
                    self.done.add(str(key))
        elif not resume and self.path.exists():
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def pending(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if str(r.get(self.id_key)) not in self.done]

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        key = row.get(self.id_key)
        if key is not None:
            self.done.add(str(key))

    def drop(self, ids: Iterable[str]) -> int:
        if not self.path.exists():
            return 0
        drop = {str(i) for i in ids}
        rows = [r for r in iter_jsonl(self.path) if str(r.get(self.id_key)) not in drop]
        removed = len(load_jsonl(self.path)) - len(rows)
        write_jsonl_atomic(self.path, rows)
        self.done -= drop
        return removed


def subsample(rows: Sequence[dict[str, Any]], *, limit: int | None, seed: int | None,
              stratify_on: str | None = None) -> list[dict[str, Any]]:
    if not limit or limit >= len(rows):
        return list(rows)
    if seed is None:
        strat = f" (rows are ordered by {stratify_on})" if stratify_on else ""
        raise ValueError(
            f"--limit {limit} needs --seed: this file is not shuffled{strat}, so the "
            "first rows are one stratum, not a sample."
        )
    rng = random.Random(seed)
    if not stratify_on:
        picked = rows[:]
        rng.shuffle(picked)
        return picked[:limit]

    groups: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r.get(stratify_on), []).append(r)
    for g in groups.values():
        rng.shuffle(g)

    out: list[dict[str, Any]] = []
    keys = sorted(groups, key=lambda k: (k is None, str(k)))
    quota = {k: limit * len(groups[k]) / len(rows) for k in keys}
    take = {k: int(quota[k]) for k in keys}
    for k in sorted(keys, key=lambda k: quota[k] - take[k], reverse=True):
        if sum(take.values()) >= limit:
            break
        take[k] = min(take[k] + 1, len(groups[k]))
    for k in keys:
        out.extend(groups[k][:take[k]])
    order = {id(r): i for i, r in enumerate(rows)}
    out.sort(key=lambda r: order[id(r)])
    return out


_BIG_ESCAPE = re.compile(r"\\U([0-9a-fA-F]{8})")


def _repair_escapes(s: str) -> str:

    def sub(m: re.Match[str]) -> str:
        cp = int(m.group(1), 16)
        if cp <= 0xFFFF:
            return f"\\u{cp:04x}"
        cp -= 0x10000
        return f"\\u{0xD800 + (cp >> 10):04x}\\u{0xDC00 + (cp & 0x3FF):04x}"

    return _BIG_ESCAPE.sub(sub, s)


def _object_spans(text: str) -> list[str]:
    spans: list[str] = []
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    spans.append(text[start:i + 1])
                    break
    seen: set[str] = set()
    ordered: list[str] = []
    for sp in spans:
        if sp not in seen:
            seen.add(sp)
            ordered.append(sp)
    return ordered


def strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    fence = re.fullmatch(r"```(?:[A-Za-z0-9_+-]*)?\n(.*)\n```", text, flags=re.S)
    return fence.group(1).strip() if fence else text


def extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    for candidate in (raw, _repair_escapes(raw)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    spans = _object_spans(raw)
    if not spans:
        raise ValueError(f"no JSON object found in output: {raw[:240]!r}")
    last: Exception | None = None
    for span in spans:
        for candidate in (span, _repair_escapes(span)):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last = exc
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError(f"could not parse a JSON object in output: {last}")


def trim_log(log: str, budget: int = 5000) -> str:
    if len(log) <= budget:
        return log
    head = log[:1800]
    tail = log[-(budget - len(head) - 40):]
    return f"{head.rstrip()}\n\n... [truncated] ...\n\n{tail.lstrip()}"
