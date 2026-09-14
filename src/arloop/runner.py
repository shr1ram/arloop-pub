"""The Runner seam: how an attempt's workspace gets executed.

SubprocessRunner is the real path (bwrap-confined by default);
CallableRunner is the in-process test double. A Submission carries the FULL
log — the classifier and the memory writer both need complete error text.
"""
from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable, Protocol

from arloop.sandbox import build_argv, build_env

#: exit code reported on a per-attempt timeout (mirrors timeout(1))
TIMEOUT_EXIT = 124

#: seconds a timed-out process group gets between SIGTERM and SIGKILL — long
#: enough for the prompt-mandated faulthandler to dump the line it hung on.
TERM_GRACE_S = 3.0


@dataclass
class Submission:
    """The outcome of running one attempt's solution.py."""

    exit_code: int
    log: str                           # full combined stdout+stderr
    duration_s: float


class Runner(Protocol):
    def submit(self, workspace: Path, command: str,
               timeout_s: int) -> Submission:
        """Run `command` in `workspace` and return its Submission."""


class CallableRunner:
    """In-process runner backed by a function; for tests."""

    def __init__(self, fn: Callable[[Path, str, int], Submission]):
        self._fn = fn

    def submit(self, workspace: Path, command: str,
               timeout_s: int) -> Submission:
        """Delegate to the wrapped function."""
        return self._fn(Path(workspace), command, timeout_s)


class SubprocessRunner:
    """Local execution, bwrap-confined unless sandbox=False."""

    def __init__(self, cpu_cap: int = 1,
                 ro_binds: tuple[tuple[str, str], ...] = (),
                 sandbox: bool = True):
        self.cpu_cap = cpu_cap
        self.ro_binds = ro_binds
        self.sandbox = sandbox

    def submit(self, workspace: Path, command: str,
               timeout_s: int) -> Submission:
        """Run the attempt, killing the whole process group on timeout."""
        workspace = Path(workspace)
        t0 = monotonic()
        if self.sandbox:
            cmd = build_argv(command, workspace, self.ro_binds)
            shell = False
            env = build_env(self.cpu_cap)
        else:
            cmd, shell = command, True
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        # start_new_session so a timeout kills the whole GROUP: agent code
        # routinely spawns workers that would otherwise outlive the shell.
        # errors="replace" so one non-UTF8 byte is a classifiable log, not a
        # UnicodeDecodeError that crashes the run.
        proc = subprocess.Popen(
            cmd, shell=shell, cwd=workspace, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", env=env)
        try:
            out, err = proc.communicate(timeout=timeout_s)
            log = out + err
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
            out, err = proc.communicate()
            log = ((out or "") + (err or "")
                   + f"\nTIMEOUT: killed after {timeout_s}s\n")
            exit_code = TIMEOUT_EXIT
        return Submission(exit_code=exit_code, log=log,
                          duration_s=monotonic() - t0)

    def _kill_group(self, proc: subprocess.Popen) -> None:
        """SIGTERM the group, wait briefly, then SIGKILL whatever is left.

        TERM first so the prompt-mandated faulthandler can fire; the
        unconditional KILL still guarantees the group dies.
        """
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            proc.kill()                     # group gone; reap the leader
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
