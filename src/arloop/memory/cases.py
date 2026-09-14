"""The Case: the unit of memory. One kind, one granularity (whole-run
episodes), so neither is a field. Provenance — which trace events a case was
derived from — is mandatory, and the id is content-addressed over it."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Case:
    id: str
    content: str                       # what gets injected into context
    task_id: str                       # the anchor task (LOTO keys on it)
    provenance: list[dict[str, Any]]   # [{"run_id": ..., "seq_span": [lo, hi]}]
    meta: dict[str, Any] = field(default_factory=dict)  # retrieval_context, chars

    @property
    def chars(self) -> int:
        """Character length of the injected content."""
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for cases.jsonl."""
        return {"id": self.id, "content": self.content, "task_id": self.task_id,
                "provenance": self.provenance, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Case":
        """Rebuild a Case from its cases.jsonl row."""
        return cls(id=d["id"], content=d["content"], task_id=d["task_id"],
                   provenance=d["provenance"], meta=d.get("meta", {}))


def make_case(content: str, task_id: str, provenance: list[dict[str, Any]],
              meta: dict[str, Any] | None = None) -> Case:
    """Build a Case with a content-addressed id: an identical derivation gives
    an identical id, which is the anchor every retrieval audit joins on."""
    if not provenance:
        raise ValueError("a case needs provenance")
    digest = hashlib.sha256(
        json.dumps([content, provenance], sort_keys=True).encode()).hexdigest()[:12]
    meta = dict(meta or {})
    meta["chars"] = len(content)
    return Case(id=f"case-{digest}", content=content, task_id=task_id,
                provenance=provenance, meta=meta)
