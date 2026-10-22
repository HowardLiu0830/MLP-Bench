from __future__ import annotations

import os
import time

__all__ = [
    "lean_threads",
    "cpu_share",
    "cores_free",
    "cpu_list",
    "cpu_set",
    "workers",
    "pin_process",
]


def lean_threads() -> int:
    return max(1, int(os.environ.get("MLPBENCH_LEAN_THREADS", "2")))


def cpu_share() -> float:
    return float(os.environ.get("MLPBENCH_CPU_SHARE", "0.75"))


def _busy_cores(window: float = 1.0) -> float:

    def snap() -> tuple[int, int]:
        with open("/proc/stat", encoding="ascii") as f:
            parts = [int(x) for x in f.readline().split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        return sum(parts), idle

    t0, i0 = snap()
    time.sleep(window)
    t1, i1 = snap()
    dt, di = max(1, t1 - t0), max(0, i1 - i0)
    total = os.cpu_count() or 1
    return max(0.0, total * (1.0 - di / dt))


_budget: int | None = None


def cores_free(*, share: float | None = None, recompute: bool = False) -> int:
    global _budget
    if _budget is not None and not recompute:
        return _budget
    share = cpu_share() if share is None else share
    total = os.cpu_count() or 1
    _budget = max(1, int((total - _busy_cores()) * share))
    return _budget


def cpu_list() -> str:
    env = os.environ.get("MLPBENCH_CPU_LIST")
    if env:
        return env
    return f"0-{max(0, cores_free() - 1)}"


def _parse_cpu_list(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def cpu_set() -> set[int]:
    allowed = os.sched_getaffinity(0)
    wanted = _parse_cpu_list(cpu_list()) & allowed
    return wanted or set(allowed)


def workers(requested: int | str | None = None, *, threads_per_job: int | None = None) -> int:
    if isinstance(requested, int) and requested > 0:
        return requested
    if isinstance(requested, str) and requested.isdigit() and int(requested) > 0:
        return int(requested)
    per = threads_per_job or lean_threads()
    return max(1, cores_free() // max(1, per))


def pin_process() -> set[int]:
    want = cpu_set()
    os.sched_setaffinity(0, want)
    return set(os.sched_getaffinity(0))


