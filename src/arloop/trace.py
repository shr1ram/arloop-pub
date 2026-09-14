"""Run manifests + append-only event logs.

One run lives at <root>/<task_id>/<run_id>/ as manifest.json (written at run
START with status running, rewritten atomically at close), trace.jsonl which
is gzipped at close, and artifacts/. The collection of manifests doubles as
the grid ledger, and the traces are the corpus the memory bank derives from.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import socket
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

EVENT_TYPES = frozenset({
    "task_presented", "memory_retrieval", "llm_call", "code_written",
    "submit", "execution_result", "submission_validation", "public_eval",
    "score", "final_outcome",
})

MANIFEST_NAME = "manifest.json"
TRACE_NAME = "trace.jsonl"
TRACE_GZ_NAME = "trace.jsonl.gz"
ARTIFACTS_DIR = "artifacts"
STATUSES = ("running", "ok", "failed")


@dataclass
class TraceEvent:
    ts: float
    run_id: str
    seq: int
    attempt: int
    type: str
    payload: dict[str, Any]

    def validate(self) -> None:
        """Raise ValueError unless the event is well formed."""
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.type!r}")
        if self.seq < 0 or self.attempt < 0:
            raise ValueError("seq/attempt must be >= 0")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dict")

    def to_json(self) -> str:
        """One JSONL line.

        allow_nan=False: the payload carries provider-controlled tool args,
        and json re-emits NaN/Infinity verbatim, producing a line only Python
        can read back. Fail at write time instead.
        """
        return json.dumps({"ts": self.ts, "run_id": self.run_id,
                           "seq": self.seq, "attempt": self.attempt,
                           "type": self.type, "payload": self.payload},
                          ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceEvent":
        """Rebuild an event, tolerating unknown envelope fields."""
        return cls(ts=d["ts"], run_id=d["run_id"], seq=d["seq"],
                   attempt=d["attempt"], type=d["type"], payload=d["payload"])


@dataclass
class RunManifest:
    """Run metadata; host/pid make a crashed run detectable and reclaimable."""
    run_id: str
    task_id: str
    seed: int
    config_hash: str
    model: str
    temperature: float
    status: str = "running"
    schema_version: int = SCHEMA_VERSION
    started_at: str = ""
    finished_at: Optional[str] = None
    host: str = ""
    pid: int = 0
    budget: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    proxy_score: Optional[float] = None
    heldout_score: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError unless the manifest is well formed."""
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.status not in STATUSES:
            raise ValueError(f"status {self.status!r} not in {STATUSES}")
        # extra merges at the TOP level, so a colliding key would silently
        # shadow a real field (extra["status"] corrupting the ledger)
        shadowed = {f.name for f in fields(self)} & set(self.extra)
        if shadowed:
            raise ValueError(f"extra must not shadow fields: {sorted(shadowed)}")

    def to_dict(self) -> dict[str, Any]:
        """Flat dict with `extra` merged at the top level."""
        d = {f.name: getattr(self, f.name)
             for f in fields(self) if f.name != "extra"}
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunManifest":
        """Rebuild a manifest; unknown keys land in `extra`."""
        known = {f.name for f in fields(cls) if f.name != "extra"}
        m = cls(**{k: v for k, v in d.items() if k in known},
                extra={k: v for k, v in d.items() if k not in known})
        m.validate()
        return m


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def run_dir_for(root: Path, task_id: str, run_id: str) -> Path:
    """Where one run's directory lives under the traces root."""
    return Path(root) / task_id / run_id


class TraceWriter:
    """Append-only JSONL during a run; gzip + final manifest at close.

    Every event is flushed, so a crash costs at most a truncated last line.
    As a context manager an exception closes the run as failed: a crash must
    never leave a permanent "running" manifest.
    """

    def __init__(self, run_dir: Path, manifest: RunManifest):
        self.run_dir = Path(run_dir)
        self.manifest = manifest
        manifest.started_at = manifest.started_at or _utc_now_iso()
        manifest.host = manifest.host or socket.gethostname()
        manifest.pid = manifest.pid or os.getpid()
        manifest.validate()

        self.run_dir.mkdir(parents=True, exist_ok=True)
        existing = self.run_dir / MANIFEST_NAME
        if existing.exists():
            try:
                prior = json.loads(existing.read_text()).get("status")
            except (json.JSONDecodeError, OSError):
                prior = None
            if prior in ("ok", "failed"):
                # re-attaching would reset a finalised run to "running",
                # clobber its gzip and duplicate seq numbers
                raise RuntimeError(
                    f"run_dir {self.run_dir} already holds a finalised run "
                    f"(status {prior}); refusing to overwrite it")
        (self.run_dir / ARTIFACTS_DIR).mkdir(exist_ok=True)
        _atomic_write_json(existing, manifest.to_dict())

        self._seq = 0
        self._fh = open(self.run_dir / TRACE_NAME, "a", encoding="utf-8")
        self._closed = False

    def emit(self, type: str, payload: dict[str, Any],
             attempt: int) -> TraceEvent:
        """Append one event and return it."""
        if self._closed:
            raise RuntimeError("TraceWriter is closed")
        event = TraceEvent(ts=time.time(), run_id=self.manifest.run_id,
                           seq=self._seq, attempt=attempt, type=type,
                           payload=payload)
        event.validate()
        self._fh.write(event.to_json() + "\n")
        self._fh.flush()
        self._seq += 1
        return event

    def artifact_path(self, name: str) -> Path:
        """Path for an artifact file inside this run's artifacts dir."""
        return self.run_dir / ARTIFACTS_DIR / name

    def close(self, status: str = "ok", *,
              proxy_score: Optional[float] = None,
              heldout_score: Optional[float] = None,
              budget_spent: Optional[float] = None,
              cost: Optional[dict[str, Any]] = None) -> None:
        """Seal the run: gzip the trace, then finalise the manifest."""
        if self._closed:
            return
        self._fh.close()
        # gzip -> manifest -> unlink, each step resumable: in any other order
        # a crash mid-close strands the run with a "running" manifest and no
        # raw trace to retry from
        raw = self.run_dir / TRACE_NAME
        if raw.exists():
            with open(raw, "rb") as src, \
                    gzip.open(self.run_dir / TRACE_GZ_NAME, "wb") as dst:
                dst.write(src.read())

        m = self.manifest
        m.status = status
        m.finished_at = _utc_now_iso()
        if proxy_score is not None:
            m.proxy_score = proxy_score
        if heldout_score is not None:
            m.heldout_score = heldout_score
        if budget_spent is not None:
            m.budget = {**m.budget, "spent": budget_spent}
        if cost is not None:
            m.cost = {**m.cost, **cost}
        m.validate()
        _atomic_write_json(self.run_dir / MANIFEST_NAME, m.to_dict())
        raw.unlink(missing_ok=True)
        self._closed = True

    def discard(self) -> None:
        """Delete this run's dir entirely: a run that aborted before making
        any attempt carries no experimental signal, and leaving the corpse
        behind bloats every ledger scan. Idempotent; safe after close()."""
        if self._closed:
            return
        try:
            self._fh.close()
        except OSError:
            pass
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self._closed = True

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(status="failed" if exc_type is not None else "ok")


def read_manifest(run_dir: Path) -> RunManifest:
    """Read one run's manifest; refuse a trace newer than this reader."""
    m = RunManifest.from_dict(
        json.loads((Path(run_dir) / MANIFEST_NAME).read_text()))
    if m.schema_version > SCHEMA_VERSION:
        raise ValueError(
            f"trace at {run_dir} has schema v{m.schema_version}, newer than "
            f"this reader (v{SCHEMA_VERSION})")
    return m


def read_events(run_dir: Path) -> Iterator[TraceEvent]:
    """Yield a run's events: the gzip for closed runs, the raw file for
    in-flight ones. A truncated final line (crash signature) is tolerated."""
    run_dir = Path(run_dir)
    gz, raw = run_dir / TRACE_GZ_NAME, run_dir / TRACE_NAME
    if gz.exists():
        fh = gzip.open(gz, "rt", encoding="utf-8")
    elif raw.exists():
        fh = open(raw, "rt", encoding="utf-8")
    else:
        raise FileNotFoundError(f"no trace file in {run_dir}")
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                break
            yield TraceEvent.from_dict(d)


def iter_runs(root: Path, status: Optional[str] = None
              ) -> Iterator[tuple[RunManifest, Path]]:
    """Iterate (manifest, run_dir) under a traces root, optionally by status."""
    root = Path(root)
    if not root.exists():
        return
    for manifest_path in sorted(root.glob("*/*/" + MANIFEST_NAME)):
        run_dir = manifest_path.parent
        try:
            m = read_manifest(run_dir)
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
            # one torn manifest (a crash mid-seal on the shared filesystem)
            # must not abort a whole ledger scan; a schema refusal is a
            # ValueError and stays loud on purpose
            log.warning("skipping unreadable manifest %s: %s",
                        manifest_path, exc)
            continue
        if status and m.status != status:
            continue
        yield m, run_dir
