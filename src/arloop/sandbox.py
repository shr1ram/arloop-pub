"""bwrap sandbox for agent code: an allowlist mount namespace and an
allowlist environment.

System dirs and the running interpreter's prefixes are bound read-only, the
task's public data read-only, the workspace read-write, and nothing else —
the private answer tree is ABSENT from the namespace, not merely unwritable.
The env is rebuilt from nothing because the parent's carries the LLM API key,
and the network namespace is unshared because agent code has no use for it.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

#: every thread-pool knob the agent's stack reads. LOKY_MAX_CPU_COUNT is not
#: optional: joblib sizes a PROCESS pool that OMP_NUM_THREADS cannot bound.
THREAD_CAP_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                   "LOKY_MAX_CPU_COUNT")

#: parent-env keys forwarded when set. Never credentials.
_ENV_FORWARD = ("LANG", "LC_ALL", "TZ")


class SandboxUnavailable(RuntimeError):
    """bwrap missing, or unprivileged user namespaces disabled on this host."""


def derive_thread_cap(max_workers: int) -> int:
    """Per-cell thread budget = cores / concurrent cells, floored at 1."""
    return max(1, (os.cpu_count() or 1) // max(1, max_workers))


def bwrap_path() -> str | None:
    """Absolute path to bwrap, or None."""
    # absolute, because the exec happens under the sandbox's own narrow PATH
    return os.environ.get("ARLOOP_BWRAP") or shutil.which("bwrap")


def assert_bwrap_works() -> None:
    """Fail-closed probe: bwrap must exist AND be able to build a namespace."""
    bwrap = bwrap_path()
    if bwrap is None:
        raise SandboxUnavailable(
            "bwrap not found — install bubblewrap or set $ARLOOP_BWRAP")
    if not os.access(bwrap, os.X_OK):
        raise SandboxUnavailable(f"bwrap at {bwrap} is not executable")
    try:
        probe = subprocess.run(
            [bwrap, "--unshare-all", "--ro-bind", "/", "/", "/bin/true"],
            capture_output=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise SandboxUnavailable("bwrap probe hung for 30s") from exc
    if probe.returncode != 0:
        raise SandboxUnavailable(
            "bwrap cannot create namespaces on this host "
            f"(rc={probe.returncode}: "
            f"{probe.stderr.decode(errors='replace').strip()})")


def build_env(cpu_cap: int) -> dict[str, str]:
    """The sandbox environment, built up from nothing (allowlist, not scrub)."""
    venv_bin = str(Path(sys.executable).parent)
    # PYTHONUNBUFFERED: a killed script's pipe is read after the kill, so a
    # block-buffered stdout would leave the agent an empty timeout log.
    env = {"PATH": f"{venv_bin}:/usr/bin:/bin",
           "HOME": "/tmp/home", "TMPDIR": "/tmp",
           "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": ""}
    for key in _ENV_FORWARD:
        if key in os.environ:
            env[key] = os.environ[key]
    for key in THREAD_CAP_KEYS:
        env[key] = str(cpu_cap)
    return env


def _system_binds() -> list[str]:
    """RO-bind /usr and /etc; recreate the usr-merge top-level symlinks, or
    bind them where they are real directories."""
    args = ["--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc"]
    for top in ("/bin", "/sbin", "/lib", "/lib64"):
        path = Path(top)
        if path.is_symlink():
            args += ["--symlink", os.readlink(top), top]
        elif path.is_dir():
            args += ["--ro-bind", top, top]
    return args


def _python_binds() -> list[str]:
    """RO-bind every prefix the interpreter needs, at the paths AS WRITTEN.

    A uv-managed venv symlinks bin/python out of the venv into uv's
    interpreter store, so binding the venv alone leaves the exec ENOENT.
    """
    prefixes: dict[str, None] = {}          # ordered de-dupe
    path = Path(sys.executable)
    prefixes[str(path.parents[1])] = None
    for _ in range(10):                     # cycle guard
        if not path.is_symlink():
            break
        target = Path(os.readlink(path))
        if not target.is_absolute():
            target = path.parent / target
        path = target
        prefixes[str(path.parents[1])] = None
    args: list[str] = []
    for prefix in prefixes:
        args += ["--ro-bind", prefix, prefix]
    return args


def build_argv(command: str, workspace: Path,
               ro_binds: Sequence[tuple[str, str]] = ()) -> list[str]:
    """bwrap argv running `command` under /bin/sh inside the allowlist ns."""
    ws = str(Path(workspace).resolve())
    argv = [bwrap_path() or "bwrap", "--die-with-parent", "--unshare-all"]
    argv += _system_binds()
    argv += ["--proc", "/proc", "--dev", "/dev",
             "--tmpfs", "/dev/shm", "--tmpfs", "/tmp", "--dir", "/tmp/home",
             *_python_binds(),
             "--bind", ws, ws, "--chdir", ws]
    for src, dst in ro_binds:
        argv += ["--ro-bind", str(src), str(dst)]
    argv += ["/bin/sh", "-c", command]
    return argv
