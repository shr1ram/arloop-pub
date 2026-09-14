"""The sandbox argv/env are an allowlist, and bwrap really runs a command."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from arloop.runner import SubprocessRunner
from arloop.sandbox import (THREAD_CAP_KEYS, bwrap_path, build_argv, build_env,
                            derive_thread_cap)


def test_thread_cap_is_cores_over_workers_floored_at_one():
    cores = os.cpu_count() or 1
    assert derive_thread_cap(1) == cores
    assert derive_thread_cap(10_000) == 1
    assert derive_thread_cap(0) == cores        # a zero worker count is a 1


def test_env_is_built_from_nothing_not_scrubbed(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-also-secret")
    monkeypatch.setenv("LANG", "en_GB.UTF-8")
    env = build_env(cpu_cap=3)
    assert "VLLM_API_KEY" not in env and "OPENAI_API_KEY" not in env
    assert env["LANG"] == "en_GB.UTF-8"
    assert env["HOME"] == "/tmp/home" and env["TMPDIR"] == "/tmp"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["PATH"] == f"{Path(sys.executable).parent}:/usr/bin:/bin"


def test_every_thread_knob_carries_the_cap():
    env = build_env(cpu_cap=3)
    assert all(env[key] == "3" for key in THREAD_CAP_KEYS)


def test_argv_binds_the_workspace_rw_and_extras_ro(tmp_path):
    data = tmp_path / "public"
    data.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    argv = build_argv("python solution.py", ws, ((str(data), "/data"),))

    assert argv[0] == (bwrap_path() or "bwrap")
    assert Path(argv[0]).is_absolute() or argv[0] == "bwrap"
    assert "--die-with-parent" in argv and "--unshare-all" in argv
    assert argv[-3:] == ["/bin/sh", "-c", "python solution.py"]

    pairs = list(zip(argv, argv[1:], argv[2:]))
    assert ("--bind", str(ws.resolve()), str(ws.resolve())) in pairs
    assert ("--ro-bind", str(data), "/data") in pairs
    assert ("--ro-bind", "/usr", "/usr") in pairs
    assert ("--ro-bind", "/etc", "/etc") in pairs
    assert ("--chdir", str(ws.resolve())) in list(zip(argv, argv[1:]))
    # the interpreter's own prefix must be reachable inside the namespace
    assert str(Path(sys.executable).parents[1]) in argv


def test_argv_shares_no_network_and_masks_tmp(tmp_path):
    argv = build_argv("true", tmp_path)
    assert "--share-net" not in argv
    pairs = list(zip(argv, argv[1:]))
    assert ("--tmpfs", "/tmp") in pairs and ("--tmpfs", "/dev/shm") in pairs
    assert ("--dir", "/tmp/home") in pairs


@pytest.mark.skipif(bwrap_path() is None, reason="bwrap not installed")
def test_sandboxed_runner_really_executes(tmp_path):
    out = SubprocessRunner(cpu_cap=1, sandbox=True).submit(
        tmp_path, "/bin/true", timeout_s=60)
    assert out.exit_code == 0
    assert out.duration_s >= 0.0


@pytest.mark.skipif(bwrap_path() is None, reason="bwrap not installed")
def test_sandbox_hides_the_parent_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "sk-secret")
    out = SubprocessRunner(cpu_cap=1, sandbox=True).submit(
        tmp_path, "/bin/sh -c 'echo [$VLLM_API_KEY]'", timeout_s=60)
    assert out.exit_code == 0
    assert "sk-secret" not in out.log
