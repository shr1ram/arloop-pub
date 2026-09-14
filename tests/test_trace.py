"""Trace writer, manifest round trip, discard and iter_runs."""
from __future__ import annotations

import json

import pytest

from arloop.trace import (
    SCHEMA_VERSION, RunManifest, TraceEvent, TraceWriter, iter_runs,
    read_events, read_manifest, run_dir_for,
)


def _manifest(run_id="run-0001", task_id="ahc001", seed=0, status="running"):
    return RunManifest(run_id=run_id, task_id=task_id, seed=seed,
                       config_hash="cfg-abc", model="qwen27b",
                       temperature=0.5, status=status,
                       budget={"type": "tokens", "limit": 50_000})


def _start(root, **kw):
    m = _manifest(**kw)
    return TraceWriter(run_dir_for(root, m.task_id, m.run_id), m)


def test_manifest_written_at_start_with_running_status(tmp_path):
    w = _start(tmp_path)
    m = read_manifest(w.run_dir)
    assert m.status == "running"
    assert m.started_at and m.host and m.pid > 0
    assert (w.run_dir / "artifacts").is_dir()
    w.close()


def test_run_dir_has_no_benchmark_level(tmp_path):
    w = _start(tmp_path)
    assert w.run_dir == tmp_path / "ahc001" / "run-0001"
    w.close()


def test_events_get_monotonic_seq_and_read_back_in_flight(tmp_path):
    w = _start(tmp_path)
    e0 = w.emit("task_presented", {"goal": "g"}, attempt=0)
    e1 = w.emit("llm_call", {"tokens": {"prompt": 1, "completion": 2}},
                attempt=0)
    assert (e0.seq, e1.seq) == (0, 1)
    assert [ev.type for ev in read_events(w.run_dir)] == ["task_presented",
                                                          "llm_call"]
    w.close()


def test_close_gzips_and_finalises_manifest(tmp_path):
    w = _start(tmp_path)
    w.emit("score", {"proxy_score": 0.5}, attempt=0)
    w.close(status="ok", proxy_score=0.5, heldout_score=0.4,
            budget_spent=1234, cost={"tokens": 1234})
    assert (w.run_dir / "trace.jsonl.gz").exists()
    assert not (w.run_dir / "trace.jsonl").exists()
    m = read_manifest(w.run_dir)
    assert m.status == "ok" and m.finished_at
    assert (m.proxy_score, m.heldout_score) == (0.5, 0.4)
    assert m.budget == {"type": "tokens", "limit": 50_000, "spent": 1234}
    assert [ev.payload for ev in read_events(w.run_dir)] == [
        {"proxy_score": 0.5}]


def test_context_manager_marks_a_crash_failed(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with _start(tmp_path) as w:
            w.emit("task_presented", {"goal": "g"}, attempt=0)
            raise RuntimeError("boom")
    assert read_manifest(w.run_dir).status == "failed"


def test_emit_after_close_raises(tmp_path):
    w = _start(tmp_path)
    w.close()
    with pytest.raises(RuntimeError):
        w.emit("task_presented", {}, attempt=0)


def test_reattaching_to_a_finalised_run_refused(tmp_path):
    w = _start(tmp_path)
    w.close()
    with pytest.raises(RuntimeError, match="finalised"):
        TraceWriter(w.run_dir, _manifest())


def test_unknown_event_type_refused(tmp_path):
    w = _start(tmp_path)
    with pytest.raises(ValueError):
        w.emit("not_an_event", {}, attempt=0)
    w.close()


def test_nan_payload_refused_at_write_time(tmp_path):
    w = _start(tmp_path)
    with pytest.raises(ValueError):
        w.emit("score", {"proxy_score": float("nan")}, attempt=0)
    w.close()


def test_manifest_extra_round_trips_at_top_level(tmp_path):
    m = _manifest()
    m.extra = {"template_hash": "abc123", "arm": "final-grid-r0-w100-x0"}
    w = TraceWriter(run_dir_for(tmp_path, m.task_id, m.run_id), m)
    w.close()
    raw = json.loads((w.run_dir / "manifest.json").read_text())
    assert raw["template_hash"] == "abc123"
    back = read_manifest(w.run_dir)
    assert back.extra == {"template_hash": "abc123",
                          "arm": "final-grid-r0-w100-x0"}


def test_extra_may_not_shadow_a_field():
    m = _manifest()
    m.extra = {"status": "ok"}
    with pytest.raises(ValueError):
        m.validate()


def test_event_round_trip():
    ev = TraceEvent(ts=1.0, run_id="r", seq=3, attempt=1, type="submit",
                    payload={"command": "python solution.py"})
    assert TraceEvent.from_dict(json.loads(ev.to_json())) == ev


def test_discard_removes_the_run_dir(tmp_path):
    w = _start(tmp_path)
    w.emit("task_presented", {"goal": "g"}, attempt=0)
    w.discard()
    assert not w.run_dir.exists()
    w.discard()   # idempotent
    assert list(iter_runs(tmp_path)) == []


def test_truncated_final_line_is_tolerated(tmp_path):
    w = _start(tmp_path)
    w.emit("task_presented", {"goal": "g"}, attempt=0)
    with open(w.run_dir / "trace.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"ts": 1.0, "run_id":')
    assert [ev.type for ev in read_events(w.run_dir)] == ["task_presented"]
    w.close()


def test_iter_runs_filters_by_status_and_skips_torn_manifests(tmp_path):
    for run_id, status in (("run-a", "ok"), ("run-b", "failed")):
        w = _start(tmp_path, run_id=run_id)
        w.close(status=status)
    torn = tmp_path / "ahc002" / "run-c"
    torn.mkdir(parents=True)
    (torn / "manifest.json").write_text("{not json")

    assert {m.run_id for m, _ in iter_runs(tmp_path)} == {"run-a", "run-b"}
    ok = list(iter_runs(tmp_path, status="ok"))
    assert [m.run_id for m, _ in ok] == ["run-a"]
    assert ok[0][1] == tmp_path / "ahc001" / "run-a"


def test_reader_refuses_a_newer_schema(tmp_path):
    w = _start(tmp_path)
    w.close()
    path = w.run_dir / "manifest.json"
    raw = json.loads(path.read_text())
    raw["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="newer"):
        read_manifest(w.run_dir)
