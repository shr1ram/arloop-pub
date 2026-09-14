"""Bank (a derived artifact, always rebuildable from corpus x write policy)
and MemoryView, the only read surface the agent ever sees."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arloop.memory.cases import Case
from arloop.tokens import approx_tokens

log = logging.getLogger(__name__)

MANIFEST_NAME = "bank_manifest.json"
CASES_NAME = "cases.jsonl"


def bank_id_for(derivation: dict[str, Any]) -> str:
    """Hash of the derivation identity (corpus, write params, writer model)."""
    digest = hashlib.sha256(
        json.dumps(derivation, sort_keys=True).encode()).hexdigest()[:12]
    return f"bank-{digest}"


@dataclass
class Bank:
    bank_id: str
    cases: list[Case]
    manifest: dict[str, Any]
    dir: Path | None = None        # where the bank lives (embedding index cache)

    @property
    def by_id(self) -> dict[str, Case]:
        """Cases keyed by id."""
        return {c.id: c for c in self.cases}


def build_bank(bank_dir: str | Path, cases: list[Case],
               derivation: dict[str, Any]) -> Bank:
    """Write cases.jsonl, then the manifest atomically LAST — a bank without a
    manifest is unfinished, not corrupt."""
    bank_dir = Path(bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank_id = bank_id_for(derivation)

    cases_tmp = bank_dir / f"{CASES_NAME}.tmp{os.getpid()}"
    with open(cases_tmp, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
    os.replace(cases_tmp, bank_dir / CASES_NAME)

    manifest = {"bank_id": bank_id, "derivation": derivation,
                "n_cases": len(cases),
                "total_chars": sum(c.chars for c in cases)}
    tmp = bank_dir / f"{MANIFEST_NAME}.tmp{os.getpid()}"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, bank_dir / MANIFEST_NAME)
    return Bank(bank_id=bank_id, cases=cases, manifest=manifest, dir=bank_dir)


def load_bank(bank_dir: str | Path) -> Bank:
    """Read a bank written by build_bank."""
    bank_dir = Path(bank_dir)
    manifest = json.loads((bank_dir / MANIFEST_NAME).read_text())
    cases = [Case.from_dict(json.loads(line))
             for line in (bank_dir / CASES_NAME).read_text().splitlines() if line]
    return Bank(bank_id=manifest["bank_id"], cases=cases, manifest=manifest,
                dir=bank_dir)


class MemoryView:
    """The agent's whole read surface: query(text) -> [(Case, score)], ranked
    by the retriever and packed under a token budget."""

    def __init__(self, bank: Bank, retriever, budget_tokens: int,
                 exclude_task_id: str | None = None):
        self._bank = bank
        self._retriever = retriever
        self._budget = budget_tokens
        self._exclude_task_id = exclude_task_id

    def query(self, text: str) -> list[tuple[Case, float]]:
        """Rank the whole bank, drop the running task's own cases, then greedily
        pack by rank under the read budget.

        Dropping them is leave-one-task-out: a task must never retrieve
        experience derived from itself, or memory would be handing it its own
        answer key.
        """
        ranked = self._retriever.query(text, self._bank, len(self._bank.cases))
        if self._exclude_task_id is not None:
            # Applied AFTER ranking: rank positions stay bank-wide and scores
            # are not renormalised — excluded cases simply vanish.
            ranked = [(c, s) for c, s in ranked
                      if c.task_id != self._exclude_task_id]
        return self._pack(ranked)

    def _pack(self, ranked: list[tuple[Case, float]]) -> list[tuple[Case, float]]:
        """Greedy by rank, skip-not-stop; one deliberate over-budget exception."""
        packed: list[tuple[Case, float]] = []
        remaining = self._budget
        seen: set[str] = set()
        for case, score in ranked:
            if case.content in seen:
                continue        # identical bytes twice is pure waste
            size = approx_tokens(case.content)
            if size > remaining:
                continue        # skip, don't stop: skipping omits, never reorders
            packed.append((case, score))
            seen.add(case.content)
            remaining -= size

        if ranked and approx_tokens(ranked[0][0].content) > self._budget:
            # The ranking's own best answer is a case the budget cannot
            # express. Injecting it over budget beats substituting a sample of
            # atypically short low-ranked cases that misrepresents the bank.
            case, score = ranked[0]
            log.warning("read budget: top-ranked case does not fit R=%d tokens — "
                        "injecting it over budget (%d tokens; %d lower-ranked "
                        "case(s) displaced)", self._budget,
                        approx_tokens(case.content), len(packed))
            return [(case, score)]
        return packed
