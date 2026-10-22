from __future__ import annotations

import json
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from ..pricing import Price

__all__ = ["TRANSIENT", "SESSION_LIMIT", "BACKOFF", "seconds_until_reset", "run_with_retry",
           "ActionCounter", "CostMeter", "AnyStop"]

TRANSIENT = re.compile(
    r"at capacity|rate.?limit|too many requests|\b429\b|\b50[0234]\b|overloaded|"
    r"temporarily unavailable|connection reset|timed? ?out reading|EOF occurred",
    re.I,
)

BACKOFF: tuple[int, ...] = (20, 60, 150, 300)

SESSION_LIMIT = re.compile(
    r"\blimit\b.{0,40}?resets?\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)", re.I | re.S
)
LIMIT_CAP_S = 2 * 60 * 60


def seconds_until_reset(out: str, *, now: time.struct_time | None = None) -> int:
    m = SESSION_LIMIT.search(out or "")
    if not m:
        return 0
    hour, minute, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
    hour = (hour % 12) + (12 if mer == "pm" else 0)
    t = now or time.localtime()
    delta = (hour - t.tm_hour) * 3600 + (minute - t.tm_min) * 60 - t.tm_sec
    if delta <= 0:
        delta += 24 * 3600
    return min(delta + 60, LIMIT_CAP_S)


class ActionCounter:
    """Counts agent actions in a live CLI event stream, one per item, the same for both CLIs.

    An action is one tool call or one piece of text the model shows: a codex item (command,
    file change, web search, message; `reasoning` / `todo_list` / `error` are not actions),
    and for claude each `tool_use` block and each non-empty `text` block (thinking is not an
    action). codex's JSON stream has no model-message boundaries, so counting per message
    (tau-bench, SWE-bench Pro) is not possible there; counting claude per block instead puts
    both on the same footing. Like tau-bench the cap is never told to the agent: the run just
    ends and the state on disk is graded. Called on each stdout line; returns True -- kill
    now -- once the cap-th action's result has come back (claude: the next tool_result;
    codex: that item's `item.completed`), or, as a fallback, as soon as an action beyond the
    cap appears.
    """

    _NOT_ACTIONS = ("reasoning", "todo_list", "error")
    reason = "the action cap"

    def __init__(self, cap: int):
        self.cap = int(cap or 0)
        self.reset()

    def reset(self) -> None:
        """A retried call starts a fresh session, so its actions start from zero."""
        self.n, self.hit = 0, False
        self._seen: set[str] = set()
        self._last: str | None = None

    def _reach(self, m: int) -> bool:
        """The state after action m is on disk: stop if m is the cap."""
        if self.cap and m == self.cap:
            return self._stop()
        return False

    def _stop(self) -> bool:
        self.hit = True
        self.n = min(self.n, self.cap)
        return True

    def __call__(self, line: str) -> bool:
        if not self.cap or not line.startswith("{"):
            return False
        try:
            e = json.loads(line)
        except ValueError:
            return False
        t = e.get("type")
        if t == "user" and self._last and self._last.startswith("c:"):
            if self._reach(self.n):                  # the n-th claude action has its result
                return True
        if t == "assistant":
            msg = e.get("message") or {}
            for i, b in enumerate(msg.get("content") or []):
                if b.get("type") == "tool_use":
                    k = f"c:{msg.get('id')}:{b.get('id') or i}"
                elif b.get("type") == "text" and str(b.get("text", "")).strip():
                    k = f"c:{msg.get('id')}:text:{hash(b['text'])}"
                else:
                    continue
                if self._count(k):
                    return True
            return False
        key = None
        if t in ("item.started", "item.completed"):
            it = e.get("item") or {}
            if it.get("type") not in (None, *self._NOT_ACTIONS):
                key = "x:" + str(it.get("id"))
                if t == "item.completed" and key == self._last:
                    self._seen.add(key)
                    return self._reach(self.n)       # the n-th codex action completed
        return key is not None and self._count(key)

    def _count(self, key: str) -> bool:
        if key in self._seen:
            return False
        self._seen.add(key)
        self.n += 1
        self._last = key
        if self.cap and self.n > self.cap:
            return self._stop()                      # fallback: an action beyond the cap began
        return False


class CostMeter:
    """Adds up what an agent run has cost so far, at list price, and stops it at `cap` dollars.

    claude: every `stream_event` / `message_delta` (needs `--include-partial-messages`) carries
    the finished reply's full usage, thinking included. codex's `--json` stream has no usage
    until the session ends, so the meter reads the session's rollout file instead, whose
    `token_count` events give the running total after every model reply. Like the action cap,
    the agent is never told; when the total reaches the cap the run is killed and the state on
    disk is graded. `usage()` gives the token totals, which survive the kill.
    """

    reason = "the cost cap"

    def __init__(self, cap: float, price: Price, *, rollout_dir: Path | None = None):
        self.cap, self.price, self.rollout_dir = float(cap), price, rollout_dir
        self.reset()

    def reset(self) -> None:
        self.cost, self.hit = 0.0, False
        self.tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
        self._w1h: float = 1.0           # share of claude cache writes that are 1-hour
        self._codex_total: dict[str, int] = {}
        self._offsets: dict[Path, int] = {}
        if self.rollout_dir and self.rollout_dir.is_dir():   # a retry: skip the old session
            for f in self.rollout_dir.rglob("rollout-*.jsonl"):
                self._offsets[f] = f.stat().st_size

    def _add(self, *, input=0, output=0, cache_read=0, cache_write=0, cache_write_1h=0,
             prompt=None):
        self.cost += self.price.cost(input=input, output=output, cache_read=cache_read,
                                     cache_write=cache_write, cache_write_1h=cache_write_1h,
                                     prompt=prompt)
        self.tokens["input"] += input
        self.tokens["output"] += output
        self.tokens["cache_read"] += cache_read
        self.tokens["cache_creation"] += cache_write + cache_write_1h

    def _claude(self, e: dict) -> None:
        ev = e.get("event") or {}
        if ev.get("type") == "message_start":
            cc = ((ev.get("message") or {}).get("usage") or {}).get("cache_creation") or {}
            one, five = cc.get("ephemeral_1h_input_tokens"), cc.get("ephemeral_5m_input_tokens")
            if isinstance(one, int) and isinstance(five, int) and one + five:
                self._w1h = one / (one + five)
        elif ev.get("type") == "message_delta":
            u = ev.get("usage") or {}
            w = int(u.get("cache_creation_input_tokens") or 0)
            w1 = round(w * self._w1h)
            self._add(input=int(u.get("input_tokens") or 0),
                      output=int(u.get("output_tokens") or 0),
                      cache_read=int(u.get("cache_read_input_tokens") or 0),
                      cache_write=w - w1, cache_write_1h=w1)

    def _codex_poll(self) -> None:
        if not self.rollout_dir or not self.rollout_dir.is_dir():
            return
        for f in sorted(self.rollout_dir.rglob("rollout-*.jsonl")):
            off = self._offsets.get(f, 0)
            try:
                with f.open("rb") as fh:
                    fh.seek(off)
                    chunk = fh.read()
            except OSError:
                continue
            end = chunk.rfind(b"\n") + 1           # leave a half-written line for next time
            self._offsets[f] = off + end
            for raw in chunk[:end].splitlines():
                try:
                    e = json.loads(raw)
                except ValueError:
                    continue
                p = e.get("payload") or {}
                info = (p.get("info") or {}) if p.get("type") == "token_count" else {}
                tot = info.get("total_token_usage")
                if not isinstance(tot, dict) or tot == self._codex_total:
                    continue
                d = {k: int(tot.get(k) or 0) - int(self._codex_total.get(k) or 0)
                     for k in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                               "output_tokens")}
                self._codex_total = tot
                cached, written = d["cached_input_tokens"], d["cache_write_input_tokens"]
                last = info.get("last_token_usage") or {}
                self._add(input=max(d["input_tokens"] - cached - written, 0),
                          output=d["output_tokens"], cache_read=cached, cache_write=written,
                          # the long-context rate is per request; a delta may span two replies
                          prompt=int(last["input_tokens"]) if "input_tokens" in last else None)

    def _check(self) -> bool:
        if self.cap and self.cost >= self.cap:
            self.hit = True
            return True
        return False

    def tick(self) -> bool:
        """Called while the stream is quiet: codex usage lands in the rollout, not on stdout."""
        self._codex_poll()
        return self._check()

    def __call__(self, line: str) -> bool:
        if line.startswith("{") and '"stream_event"' in line:
            try:
                e = json.loads(line)
            except ValueError:
                e = {}
            if e.get("type") == "stream_event":
                self._claude(e)
        self._codex_poll()
        return self._check()

    def finish(self) -> None:
        """Read whatever the rollout gained after the last stdout line (the run's tail)."""
        self._codex_poll()


class AnyStop:
    """Feeds every line to each stop rule (so each keeps counting) and stops on the first hit."""

    def __init__(self, *rules):
        self.rules = [r for r in rules if r is not None]
        self.reason = "the action cap"

    def reset(self) -> None:
        for r in self.rules:
            if hasattr(r, "reset"):
                r.reset()

    def _pick(self, hits: list) -> bool:
        if any(hits):
            self.reason = getattr(self.rules[hits.index(True)], "reason", "the action cap")
            return True
        return False

    def tick(self) -> bool:
        return self._pick([bool(r.tick()) if hasattr(r, "tick") else False for r in self.rules])

    def __call__(self, line: str) -> bool:
        return self._pick([bool(r(line)) for r in self.rules])


def _run_streaming(cmd, *, stdin: str, cwd, env, timeout: int,
                   stop_when: Callable[[str], bool]) -> tuple[int | None, str, str, bool]:
    """Like subprocess.run, but reads stdout as it arrives and kills the process as soon as
    `stop_when(line)` is true. Returns (rc, out, err, stopped); rc is None on timeout."""
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=str(cwd) if cwd else None,
                            env=env, bufsize=1)
    lines: queue.Queue = queue.Queue()
    err_parts: list[str] = []

    def feed():
        try:
            proc.stdin.write(stdin)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def pump_out():
        for ln in proc.stdout:
            lines.put(ln)
        lines.put(None)

    def pump_err():
        err_parts.append(proc.stderr.read())

    for fn in (feed, pump_out, pump_err):
        threading.Thread(target=fn, daemon=True).start()
    out_parts: list[str] = []
    deadline = time.monotonic() + timeout
    stopped = timed_out = False
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            timed_out = True
            break
        try:
            ln = lines.get(timeout=min(left, 5.0))
        except queue.Empty:
            tick = getattr(stop_when, "tick", None)
            if tick is not None and tick():
                stopped = True
                break
            continue
        if ln is None:
            break
        out_parts.append(ln)
        if stop_when(ln):
            stopped = True
            break
    if stopped or timed_out:
        proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    time.sleep(0.2)
    out, err = "".join(out_parts), "".join(err_parts)
    if timed_out:
        return None, out, err + f"\n[backend] call timed out after {timeout}s", False
    if stopped:
        return 0, out, err + f"\n[backend] stopped at {getattr(stop_when, 'reason', 'the action cap')}", True
    return proc.returncode, out, err, False


def run_with_retry(
    cmd: Sequence[str],
    *,
    stdin: str,
    cwd: str | Path | None,
    env: dict[str, str] | None,
    timeout: int,
    on_output: Callable[[str, str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda m: sys.stderr.write(m + "\n"),
    stop_when: Callable[[str], bool] | None = None,
) -> tuple[int | None, str, str, int]:
    attempts = (0,) + BACKOFF
    rc: int | None = 1
    out = err = ""
    n = 0

    for i, pause in enumerate(attempts):
        if pause:
            log(f"[backend] transient failure, retry {i}/{len(BACKOFF)} in {pause}s")
            sleep(pause)
        n = i + 1
        if stop_when is not None:
            if hasattr(stop_when, "reset"):
                stop_when.reset()
            rc, out, err, stopped = _run_streaming(cmd, stdin=stdin, cwd=cwd, env=env,
                                                   timeout=timeout, stop_when=stop_when)
            if on_output:
                on_output(out, err)
            if rc is None or stopped or rc == 0 or not TRANSIENT.search(out + "\n" + err):
                return rc, out, err, n
            wait = seconds_until_reset(out + "\n" + err)
            if wait:
                log(f"[backend] usage cap; sleeping {wait // 60}min until it resets")
                sleep(wait)
            continue
        try:
            proc = subprocess.run(cmd, input=stdin, text=True, capture_output=True,
                                  cwd=str(cwd) if cwd else None, env=env, timeout=timeout)
            out, err, rc = proc.stdout or "", proc.stderr or "", proc.returncode
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
            err = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
            err = (err or "") + f"\n[backend] call timed out after {timeout}s"
            if on_output:
                on_output(out, err)
            return None, out, err, n

        if on_output:
            on_output(out, err)

        combined = out + "\n" + err
        if rc == 0 or not TRANSIENT.search(combined):
            break

        wait = seconds_until_reset(combined)
        if wait:
            log(f"[backend] usage cap; sleeping {wait // 60}min until it resets")
            sleep(wait)
            n += 1
            try:
                proc = subprocess.run(cmd, input=stdin, text=True, capture_output=True,
                                      cwd=str(cwd) if cwd else None, env=env, timeout=timeout)
                out, err, rc = proc.stdout or "", proc.stderr or "", proc.returncode
            except subprocess.TimeoutExpired as exc:
                out = exc.stdout if isinstance(exc.stdout, str) else ""
                err = f"\n[backend] call timed out after {timeout}s"
                rc = None
            if on_output:
                on_output(out, err)
            if rc == 0:
                break

    return rc, out, err, n
