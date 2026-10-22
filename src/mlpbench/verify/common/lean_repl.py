#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path


_USER = os.environ.get("USER") or os.environ.get("LOGNAME") or str(os.getuid())
DURABLE_ROOT = Path(os.environ["MLPBENCH_LEAN_DURABLE"]) \
    if os.environ.get("MLPBENCH_LEAN_DURABLE") else None
PROJECT = Path(os.environ.get("LEAN_PROJECT") or os.environ.get("MLPBENCH_LEAN_PROJECT")
               or (Path.home() / ".local/share/mlpbench/lean_project"))
def _local_root() -> Path:
    if os.environ.get("LEAN_LOCAL_ROOT"):
        return Path(os.environ["LEAN_LOCAL_ROOT"])
    for c in (Path(f"/dev/shm/{_USER}-lean"), Path(f"/tmp/{_USER}-lean")):
        if (c / "packages").is_dir():
            return c
    return Path(f"/dev/shm/{_USER}-lean")


LOCAL_ROOT = _local_root()
LOCAL_PACKAGES = LOCAL_ROOT / "packages"
ENV_PICKLE = LOCAL_ROOT / "mathlib_env.olean"
TOOLCHAIN_NAME = (PROJECT / "lean-toolchain").read_text().strip() \
    if (PROJECT / "lean-toolchain").is_file() else "leanprover/lean4:v4.27.0-rc1"


def _first_file(*cands: Path | None) -> Path | None:
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


def _first_dir(*cands: Path | None) -> Path | None:
    for c in cands:
        if c and Path(c).is_dir():
            return Path(c)
    return None


def _elan_root() -> Path:
    return Path(os.environ.get("ELAN_HOME") or Path.home() / ".elan")


def _toolchain() -> Path | None:
    slug = TOOLCHAIN_NAME.replace("/", "--").replace(":", "---")
    return _first_dir(Path(os.environ["LEAN_TOOLCHAIN_DIR"])
                      if os.environ.get("LEAN_TOOLCHAIN_DIR") else None,
                      _elan_root() / "toolchains" / slug)


REPL_BIN = _first_file(Path(os.environ["LEAN_REPL_BIN"]) if os.environ.get("LEAN_REPL_BIN") else None,
                       LOCAL_ROOT / "repl",
                       DURABLE_ROOT / "bin/repl" if DURABLE_ROOT else None) \
    or (LOCAL_ROOT / "repl")
TOOLCHAIN = _toolchain() or _elan_root() / "toolchains" / TOOLCHAIN_NAME.replace("/", "--").replace(":", "---")
TOOLCHAIN_LIB = TOOLCHAIN / "lib/lean"


def describe() -> str:
    return "\n".join([
        f"  repl binary : {REPL_BIN}  {'OK' if REPL_BIN.is_file() else 'MISSING'}",
        f"  toolchain   : {TOOLCHAIN}  {'OK' if TOOLCHAIN.is_dir() else 'MISSING'}",
        f"  project     : {PROJECT}  {'OK' if PROJECT.is_dir() else 'MISSING'}",
        f"  packages    : {LOCAL_PACKAGES if LOCAL_PACKAGES.is_dir() else PROJECT / '.lake/packages'}"
        f"  {'node-local' if LOCAL_PACKAGES.is_dir() else 'not staged (slower first import)'}",
    ])

PKG_ORDER = ("Cli", "batteries", "Qq", "aesop", "proofwidgets",
             "importGraph", "LeanSearchClient", "plausible", "mathlib")


def lean_path(prefer_local: bool = True) -> str:
    root = LOCAL_PACKAGES if (prefer_local and LOCAL_PACKAGES.is_dir()) else \
        PROJECT / ".lake/packages"
    parts = [str(root / p / ".lake/build/lib/lean") for p in PKG_ORDER]
    parts = [p for p in parts if Path(p).is_dir()]
    parts.append(str(PROJECT / ".lake/build/lib/lean"))
    parts.append(str(TOOLCHAIN_LIB))
    return ":".join(parts)


class LeanRepl:

    def __init__(self, imports: str = "import Mathlib", use_pickle: bool = False,
                 timeout: int = 1800, prefer_local: bool = True,
                 max_heartbeats: int = 40000):
        env = os.environ.copy()
        env["LEAN_PATH"] = lean_path(prefer_local)
        env["PATH"] = f"{TOOLCHAIN / 'bin'}:{env.get('PATH', '')}"
        env["ELAN_TOOLCHAIN"] = TOOLCHAIN_NAME
        self._lock = threading.Lock()
        self._timeout = timeout
        self.proc = subprocess.Popen(
            [str(REPL_BIN)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env, bufsize=1)
        self.env_id = None
        self.load_seconds = None
        self.max_heartbeats = max_heartbeats
        self._bootstrap(imports, use_pickle)

    def _send(self, obj: dict) -> dict:
        with self._lock:
            self.proc.stdin.write(json.dumps(obj) + "\n\n")
            self.proc.stdin.flush()
            chunks = []
            while True:
                line = self.proc.stdout.readline()
                if line == "":
                    raise RuntimeError("repl closed stdout; it probably died — "
                                       f"stderr: {self._drain_stderr()[:400]}")
                if line.strip() == "" and chunks:
                    break
                chunks.append(line)
            return json.loads("".join(chunks))

    def _drain_stderr(self) -> str:
        try:
            return self.proc.stderr.read() or ""
        except Exception:
            return ""

    def _bootstrap(self, imports: str, use_pickle: bool) -> None:
        import time
        t0 = time.time()
        if use_pickle and ENV_PICKLE.exists():
            r = self._send({"unpickleEnvFrom": str(ENV_PICKLE)})
            self.env_id = r.get("env")
        if self.env_id is None:
            r = self._send({"cmd": imports, "env": None})
            self.env_id = r.get("env")
            if self.env_id is None:
                raise RuntimeError(f"could not import Mathlib: {json.dumps(r)[:500]}")
            if use_pickle:
                ENV_PICKLE.parent.mkdir(parents=True, exist_ok=True)
                self._send({"pickleTo": str(ENV_PICKLE), "env": self.env_id})
        self.load_seconds = round(time.time() - t0, 1)

    def _bounded(self, code: str) -> str:
        return (f"set_option maxHeartbeats {self.max_heartbeats} in\n{code}"
                if self.max_heartbeats else code)

    def extend(self, code: str, env=None) -> tuple[bool, list, object]:
        r = self._send({"cmd": self._bounded(code), "env": self.env_id if env is None else env})
        msgs = r.get("messages", []) or []
        errs = [m for m in msgs if m.get("severity") == "error"]
        return (not errs), msgs, r.get("env")

    def check(self, code: str) -> tuple[bool, list]:
        r = self._send({"cmd": self._bounded(code), "env": self.env_id})
        msgs = r.get("messages", []) or []
        errs = [m for m in msgs if m.get("severity") == "error"]
        return (not errs and not r.get("sorries")), msgs

    def check_statement(self, code: str, env=None) -> tuple[bool, list]:
        r = self._send({"cmd": self._bounded(code), "env": self.env_id if env is None else env})
        msgs = r.get("messages", []) or []
        errs = [m for m in msgs if m.get("severity") == "error"]
        return (not errs and len(r.get("sorries", []) or []) == 1), msgs

    OK_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
    _DECL = __import__("re").compile(r"\b(?:theorem|lemma)\s+([A-Za-z_][\w']*)")

    def check_certified(self, code: str, env=None) -> tuple[bool, list, str]:
        m = self._DECL.search(code or "")
        if not m:
            ok, msgs = self.check(code)
            return ok, msgs, "no theorem/lemma name found; axioms not verified"
        name = m.group(1)
        r = self._send({"cmd": self._bounded(code) + f"\n#print axioms {name}",
                        "env": self.env_id if env is None else env})
        msgs = r.get("messages", []) or []
        texts = [str(x.get("data", "")) for x in msgs]
        if any(x.get("severity") == "error" for x in msgs):
            first = next((str(x.get("data", "")) for x in msgs
                          if x.get("severity") == "error"), "")
            return False, msgs, "elaboration error: " + " ".join(first.split())[:300]
        if r.get("sorries") or any("uses 'sorry'" in t for t in texts):
            return False, msgs, "declaration uses sorry"
        line = next((t for t in texts if "depends on axioms" in t), None)
        if line is None:
            return False, msgs, "could not read the axiom list"
        used = {a.strip() for a in line.split("[", 1)[-1].rstrip("]").split(",") if a.strip()}
        bad = used - self.OK_AXIOMS
        if bad:
            return False, msgs, f"depends on non-standard axioms: {sorted(bad)}"
        return True, msgs, ""

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    import sys
    import time
    print("LEAN_PATH roots:")
    for p in lean_path().split(":"):
        print("   ", p)
    print(f"\npickle exists: {ENV_PICKLE.exists()}")
    t0 = time.time()
    lean = LeanRepl()
    print(f"bootstrap: {lean.load_seconds}s  env={lean.env_id}")
    for code, want in [
        ("theorem t1 (a b : ℝ) (h : a ≤ b) : a - 1 ≤ b := by linarith", True),
        ("theorem t2 (a b : ℝ) (h : a ≤ b) : b ≤ a - 1 := by linarith", False),
    ]:
        t = time.time()
        ok, msgs = lean.check(code)
        print(f"  check -> {ok} (want {want})  {time.time() - t:.2f}s")
    t = time.time()
    ok, _ = lean.check_statement("theorem t3 (a : ℝ) : a = a := by sorry")
    print(f"  check_statement(sorry) -> {ok} (want True)  {time.time() - t:.2f}s")
    lean.close()
    print(f"total {time.time() - t0:.1f}s")
    sys.exit(0)
