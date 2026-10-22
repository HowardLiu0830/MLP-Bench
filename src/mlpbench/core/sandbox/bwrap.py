from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from ...errors import SandboxUnavailable
from ..paths import lean_project as default_lean_project
from .egress import (JAIL_SOCK_DIR, OFFLINE, PROXY_PORT, TOOL_SHELLS, TOOLS_DIR, Egress, gateway,
                     bridge_python, bridge_script, tool_shell_wrapper)

__all__ = ["LEAN_PROJECT_EXPOSE", "Egress", "OFFLINE", "AGENT_STATE_HIDE_DIRS", "AGENT_STATE_HIDE_FILES",
           "CODEX_STATE_RW", "CODEX_STATE_RO", "CLAUDE_STATE_RW",
           "bwrap_bin", "require_bwrap", "agent_jail"]

LEAN_PROJECT_EXPOSE: tuple[str, ...] = (
    ".lake", ".lake/packages", "lakefile.lean", "lakefile.toml", "lake-manifest.json",
    "lean-toolchain",
)

# CODEX_STATE_RO are mounted as writable throwaway COPIES, never the originals.
# Agent state is WHITELISTED, not blacklisted. The CLIs keep every past conversation in their
# home directories (codex: thread_history_*.sqlite, state_*.sqlite, session_index.jsonl,
# scratch dirs; claude: projects/, history, ~/.claude.json's per-project prompt history), and
# those can hold benchmark items and gold fields. A blacklist misses
# whatever file the next CLI version adds, so the state directory is an empty tmpfs and only
# the files a CLI needs to authenticate are mounted into it.
CODEX_STATE_RW: tuple[str, ...] = ("auth.json",)          # token refresh writes it
CODEX_STATE_RO: tuple[str, ...] = ("version.json", "models_cache.json", "installation_id")
CLAUDE_STATE_RW: tuple[str, ...] = (".credentials.json",)  # token refresh writes it

# Kept for callers/tests that name them; everything here is now hidden by construction.
AGENT_STATE_HIDE_DIRS: tuple[str, ...] = (
    "projects", "jobs", "sessions", "file-history", "paste-cache",
    "shell-snapshots", "plans", "tasks", "todos", "backups", "downloads",
    "telemetry", "ide", "skills", "plugins", "cache", "daemon",
)
AGENT_STATE_HIDE_FILES: tuple[str, ...] = ("history.jsonl", "stats-cache.json")

_SYSTEM_DIRS: tuple[str, ...] = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt")


def bwrap_bin() -> str | None:
    return os.environ.get("BWRAP_BIN") or shutil.which("bwrap")


def require_bwrap() -> str:
    bw = bwrap_bin()
    if not bw:
        raise SandboxUnavailable(
            "bubblewrap (bwrap) is not on PATH.\n"
            "The agent conditions need it: without a mount namespace the model can read "
            "the benchmark file, and the gold proofs are a field of that file.\n"
            "Install bwrap, or pass --sandbox none to run anyway -- results from an "
            "unsandboxed run are stamped `anticheat: false` and are not benchmark results."
        )


def _resolve_binary(argv0: str) -> Path:
    p = Path(argv0)
    if not p.is_absolute():
        p = Path(shutil.which(argv0) or argv0)
    return p


def _abs(p: Path | str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (Path.cwd() / q)


def _symlink_chain(p: Path, limit: int = 16) -> list[Path]:
    seen: list[Path] = []
    cur = p
    for _ in range(limit):
        if cur in seen:
            break
        seen.append(cur)
        if not cur.is_symlink():
            break
        nxt = Path(os.readlink(cur))
        cur = nxt if nxt.is_absolute() else (cur.parent / nxt)
    try:
        real = p.resolve(strict=False)
    except (OSError, RuntimeError):
        return seen
    if real not in seen:
        seen.append(real)
    return seen


def _sanitized_copy(src: Path, kind: str) -> Path | None:
    """A throwaway copy of a CLI config with its history removed."""
    import json
    import tempfile

    from ..paths import workspace_root

    try:
        text = src.read_text(encoding="utf-8")
    except OSError:
        return None
    if kind == "verbatim":
        out = text
    elif kind == "claude.json":
        try:
            data = json.loads(text)
        except ValueError:
            return None
        # `projects` carries per-directory prompt history; the rest is onboarding/feature
        # state the CLI wants so that it does not run its first-start flow.
        for k in ("projects", "tipsHistory", "githubRepoPaths"):
            data.pop(k, None)
        out = json.dumps(data)
    else:  # codex config.toml: drop the [projects."<path>"] trust table, and the user's own
        # [shell_environment_policy]: variables it sets for the host (e.g. a LEAN_LOCAL_ROOT
        # the jail does not bind) would reach the agent's shells. The harness sets what those
        # shells need with -c (codex.NO_NETWORK).
        keep, skipping = [], False
        for line in text.splitlines():
            if line.startswith("["):
                skipping = line.startswith(("[projects", "[shell_environment_policy"))
            if not skipping:
                keep.append(line)
        out = "\n".join(keep) + "\n"
    d = workspace_root() / "_agent_state"
    d.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f"{kind}.", dir=str(d))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(out)
    return Path(name)


def release_jail(cmd: Sequence[str]) -> None:
    """Delete the throwaway config copies `agent_jail` made for `cmd`. Call it once the
    command has exited; each jail leaves ~5 copies (~60 KB each) in `_agent_state`."""
    from ..paths import workspace_root

    d = workspace_root() / "_agent_state"
    for arg in cmd:
        p = Path(arg)
        if p.parent == d:
            p.unlink(missing_ok=True)


def _agent_state_binds(home: Path, codex_sessions: Path | None = None,
                       cli: str | None = None) -> list[str]:
    """Both state dirs become empty tmpfs; `cli` ("claude" / "codex", None = both) picks whose
    credentials are mounted back in. A host-side token refresh replaces the credentials file,
    which leaves a live jail's bind of it stale, and every nested tool shell then fails to
    mask it ("Can't bind mount ... No such file"). So a jail holds only its own CLI's file."""
    binds: list[str] = []
    codex_homes = [home / ".codex"]
    ch = os.environ.get("CODEX_HOME")
    if ch and Path(ch).exists() and Path(ch) not in codex_homes:
        codex_homes.insert(0, Path(ch))
    if codex_sessions is not None:
        # The one piece of codex state that leaves the jail: this run's own rollout, which the
        # cost meter reads live (codex prints no usage until the session ends). The directory
        # is fresh per run, so it holds nothing from any other conversation.
        codex_sessions.mkdir(parents=True, exist_ok=True)

    for cfg in codex_homes:
        if not cfg.exists():
            continue
        binds += ["--tmpfs", str(cfg)]
        for f in (CODEX_STATE_RW if cli in (None, "codex") else ()):
            if (cfg / f).is_file():
                binds += ["--bind", str(cfg / f), str(cfg / f)]
        # The CLI rewrites these on start (migrations, cache refresh), so each is a WRITABLE
        # throwaway copy: a read-only mount makes codex refuse to start, and binding the
        # real file would let a sandboxed run edit the user's own config.
        for f in CODEX_STATE_RO:
            if (cfg / f).is_file():
                copy = _sanitized_copy(cfg / f, "verbatim")
                if copy:
                    binds += ["--bind", str(copy), str(cfg / f)]
        if (cfg / "config.toml").is_file():
            clean = _sanitized_copy(cfg / "config.toml", "config.toml")
            if clean:
                binds += ["--bind", str(clean), str(cfg / "config.toml")]
        if codex_sessions is not None and cfg == codex_homes[0]:
            binds += ["--bind", str(codex_sessions), str(cfg / "sessions")]

    claude_dir = home / ".claude"
    if claude_dir.exists():
        binds += ["--tmpfs", str(claude_dir)]
        for f in (CLAUDE_STATE_RW if cli in (None, "claude") else ()):
            if (claude_dir / f).is_file():
                binds += ["--bind", str(claude_dir / f), str(claude_dir / f)]

    claude_json = home / ".claude.json"
    if claude_json.is_file():
        clean = _sanitized_copy(claude_json, "claude.json")
        if clean:
            binds += ["--bind", str(clean), str(claude_json)]   # the CLI rewrites it
    return binds


def _lean_binds(home: Path, project: Path) -> tuple[list[str], list[str]]:
    binds: list[str] = []
    elan = home / ".elan"
    if elan.exists():
        binds += ["--ro-bind", str(elan), str(elan)]
    if project.exists():
        for sub in LEAN_PROJECT_EXPOSE:
            src = project / sub
            if not src.exists():
                continue
            for hop in _symlink_chain(src):
                if hop.is_symlink() or not hop.exists():
                    continue
                binds += ["--ro-bind", str(hop), str(hop)]
    env = ["--setenv", "PATH",
           f"{elan}/bin:{home}/.local/bin:/usr/local/bin:/usr/bin:/bin"]
    return binds, env


_PROXY_VARS: tuple[str, ...] = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                                "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")

# Runs first inside the jail: start the bridge (egress.BRIDGE_PY) with this shell's PID -- the
# PID the CLI will have, since the shell execs it -- wait until every port listens (read off
# /proc/net/tcp, so the wait opens no connection), then exec the real command. The bridge dies
# with the jail's PID namespace when the command exits.
_BRIDGE_SH = r"""
py=$1; script=$2; shift 2
n=$1; shift
ports=""; args=""
while [ "$n" -gt 0 ]; do
  ports="$ports $1"; args="$args $1 $2"; shift 2; n=$((n-1))
done
"$py" -I -S "$script" $$ $args &
for port in $ports; do
  hex=$(printf '%04X' "$port"); k=0
  until grep -q "0100007F:$hex 00000000:0000 0A" /proc/net/tcp; do
    k=$((k+1))
    if [ "$k" -gt 200 ]; then
      echo "mlpbench: the sandbox egress bridge on port $port did not start" >&2; exit 125
    fi
    sleep 0.05
  done
done
exec "$@"
"""


def _egress_args(egress: Egress, tag: str = "") -> tuple[list[str], list[str], list[str]]:
    """(binds, env, command prefix) that give the offline jail its allowlisted API door."""
    env: list[str] = []
    for v in _PROXY_VARS:                      # nothing from the host's proxy setup leaks in
        env += ["--unsetenv", v]
    if not egress:
        return [], env, []
    py = bridge_python()
    gw = gateway()
    script = f"{JAIL_SOCK_DIR}/bridge.py"
    binds: list[str] = ["--ro-bind", str(bridge_script()), script]
    bridges: list[str] = []
    if egress.connect:
        inside = f"{JAIL_SOCK_DIR}/proxy.sock"
        binds += ["--ro-bind", str(gw.proxy_socket(egress.connect, tag)), inside]
        bridges += [str(PROXY_PORT), inside]
        url = f"http://127.0.0.1:{PROXY_PORT}"
        for v in _PROXY_VARS[:6]:
            env += ["--setenv", v, url]
    for target in egress.forward:
        if target[1] == PROXY_PORT:
            raise SandboxUnavailable(f"cannot forward port {PROXY_PORT}: the egress proxy uses it")
        inside = f"{JAIL_SOCK_DIR}/fwd{target[1]}.sock"
        binds += ["--ro-bind", str(gw.forward_socket(target, tag)), inside]
        bridges += [str(target[1]), inside]
    # Loopback is the jail's own (the bridges live there), never the host's.
    env += ["--setenv", "NO_PROXY", "127.0.0.1,localhost,::1",
            "--setenv", "no_proxy", "127.0.0.1,localhost,::1"]
    binds += _tool_shell_binds()
    # The bridge runs on the REAL bash: /bin/sh -> bash is the tool wrapper by now.
    prefix = [f"{TOOLS_DIR}/bash", "-c", _BRIDGE_SH, "mlpbench-egress", py, script,
              str(len(bridges) // 2), *bridges]
    return binds, env, prefix


def _tool_shell_binds() -> list[str]:
    """Shells inside the jail start the agent's commands in a nested, network-less namespace.

    The CLI keeps the API door; anything it runs through a shell does not (egress.py)."""
    bw = bwrap_bin()
    if not bw or not Path("/usr/bin/bash").exists():
        raise SandboxUnavailable("cannot isolate the agent's shell: bash or bwrap is missing")
    binds = ["--ro-bind", str(Path(shutil.which(bw) or bw).resolve()), f"{TOOLS_DIR}/bwrap"]
    for sh in TOOL_SHELLS:
        if not Path(sh).exists():
            continue
        name = Path(sh).name
        binds += ["--ro-bind", str(Path(sh).resolve()), f"{TOOLS_DIR}/{name}",
                  "--ro-bind", str(tool_shell_wrapper(name)), sh]
    return binds


def agent_jail(
    cmd: Sequence[str],
    *,
    workspace: Path,
    with_lean: bool = True,
    lean_project: Path | None = None,
    extra_ro: Iterable[Path] = (),
    allow_unsandboxed: bool = False,
    codex_sessions: Path | None = None,
    egress: Egress = OFFLINE,
    cli: str | None = None,
) -> list[str]:
    """`cmd` inside a fresh mount, PID and network namespace.

    The network namespace has loopback only: no route, no DNS, so nothing the agent runs
    (shell, lake, python, pip, curl, raw sockets) reaches anything. The model CLI's own API
    traffic leaves through `egress` alone (see egress.py): an HTTPS CONNECT allowlist and
    loopback forwards, bridged in over bind-mounted unix sockets. If that bridge cannot be set
    up the command fails; it never falls back to the host network.
    """
    bw = bwrap_bin()
    if not bw:
        if not allow_unsandboxed:
            require_bwrap()
        return list(cmd)

    home = Path.home()
    binary = _resolve_binary(cmd[0])

    binds: list[str] = []
    for d in _SYSTEM_DIRS:
        if Path(d).is_symlink():
            # usrmerge (/bin -> usr/bin): a symlink, not a second bind of the same directory,
            # so a file mounted over /usr/bin/bash is also what /bin/bash resolves to.
            binds += ["--symlink", os.readlink(d), d]
        elif Path(d).exists():
            binds += ["--ro-bind", d, d]

    # State first, binary second: the codex CLI installs itself INSIDE ~/.codex
    # (packages/standalone/...), so the empty tmpfs over ~/.codex must be laid down before
    # the binary's directory is bound on top of it, not after (which hides the binary).
    binds += _agent_state_binds(home, codex_sessions, cli)

    for p in _symlink_chain(binary):
        for q in (p, p.parent):
            if q.exists():
                binds += ["--ro-bind", str(q), str(q)]

    lean_env: list[str] = []
    if with_lean:
        project = Path(lean_project) if lean_project else default_lean_project()
        lb, lean_env = _lean_binds(home, project)
        binds += lb

    # Extra read-only paths go AFTER `--dev /dev`, so one under /dev/shm (where the Lean REPL
    # and staged Mathlib usually live) survives the fresh /dev instead of being covered by it.
    late: list[str] = []
    for extra in extra_ro:
        p = _abs(extra)
        if p.exists():
            late += ["--ro-bind", str(p), str(p)]

    workspace = _abs(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # The gateway sockets live under the fresh /run, so they are mounted after it.
    net_binds, net_env, prefix = _egress_args(egress, str(workspace))

    # A private /tmp, laid down first so nothing bound later is hidden under it. The tool
    # shells nest bwrap, and bwrap 0.4 builds its new root on /tmp: without one in the jail
    # every agent command fails with "Failed to mount tmpfs" (it used to exist only when the
    # workspace happened to live under /tmp).
    return [
        bw, "--tmpfs", "/tmp", *binds,
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/run",
        *late, *net_binds,
        "--bind", str(workspace), str(workspace),
        "--setenv", "HOME", str(home), "--setenv", "TMPDIR", "/tmp",
        *lean_env, *net_env,
        "--chdir", str(workspace),
        "--unshare-pid", "--unshare-net", "--die-with-parent",
        *prefix, *cmd,
    ]
