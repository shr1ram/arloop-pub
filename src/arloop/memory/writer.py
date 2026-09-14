"""The episode writer: ok runs -> one whole-run episode case each.

Architectural invariant: no LLM call's input grows with the corpus size or the
run length. An attempt's evidence is its verbatim slice; a run's summary is a
FOLD over windows of `fold_window` step notes, so growth means more calls (each
node-cached, run in parallel), never bigger calls. Size caps are enforced by the
prompt, then by ONE `shorten` rewrite, then by dropping the unit whole — never
by truncation, which amputates a lesson head-biased.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from importlib.resources import files as pkg_files
from pathlib import Path

import yaml

from arloop.memory.bank import Bank
from arloop.llm import is_transient
from arloop.memory.cases import Case, make_case
from arloop.memory.retrievers import BM25Retriever
from arloop.trace import RunManifest, TraceEvent, read_events

log = logging.getLogger(__name__)

_DEFAULTS = yaml.safe_load(
    pkg_files("arloop.memory").joinpath("write_prompts.yaml").read_text())


@dataclass(frozen=True)
class WriteParams:
    """Write-side shape knobs — all of them are bank derivation identity."""

    c_step: int = 300
    fold_window: int = 8
    min_case_chars: int = 32
    dedupe: str = "llm"                  # llm | none
    dedupe_k: int = 4
    retrieval_context_chars: int = 240

    def __post_init__(self) -> None:
        if self.fold_window < 2:
            raise ValueError(f"fold_window must be >= 2, got {self.fold_window}")

    @property
    def episode_chars(self) -> int:
        """Episode cap — constant in the attempt count, so folds stay bounded."""
        return self.c_step * self.fold_window


@dataclass(frozen=True)
class WritePrompts:
    """Distillation prompt text (bank identity via write_prompts_hash)."""

    system: str = _DEFAULTS["system"]
    step: str = _DEFAULTS["step"]
    fold: str = _DEFAULTS["fold"]
    shorten: str = _DEFAULTS["shorten"]
    merge: str = _DEFAULTS["merge"]
    contextualize: str = _DEFAULTS["contextualize"]

    def __post_init__(self) -> None:
        # str.format ignores unknown kwargs, so a task-less override would
        # build a whole bank with the task header silently absent.
        for name in ("step", "fold", "contextualize"):
            if "{task}" not in getattr(self, name):
                raise ValueError(f"write prompt {name!r} lacks the {{task}} placeholder")


def write_prompts_hash(wp: WritePrompts = WritePrompts()) -> str:
    """Version stamp over every prompt component (bank derivation identity)."""
    payload = "\n".join(getattr(wp, f.name) for f in fields(wp))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def attempt_slice(events: list[TraceEvent]) -> tuple[str, list[int]]:
    """One attempt rendered verbatim: move, code, outcome, error, score."""
    parts: list[str] = []
    seqs = [e.seq for e in events]
    for event in events:
        p = event.payload
        if event.type == "llm_call":
            parts.append(f"move: {p.get('move', '?')}")
        elif event.type == "code_written":
            parts.append(f"code ({p.get('path')}):\n{p.get('content', '')}")
        elif event.type == "execution_result":
            outcome = p.get("classification")
            parts.append(f"outcome: {outcome}")
            if outcome != "success" and p.get("excerpt"):
                parts.append(f"error:\n{p['excerpt']}")
        elif event.type == "score":
            parts.append(f"proxy_score: {p.get('proxy_score')}")
    span = [min(seqs), max(seqs)] if seqs else [0, 0]
    return "\n".join(parts), span


def _skeleton_line(n: int, events: list[TraceEvent]) -> str:
    move = outcome = "?"
    score = None
    for event in events:
        p = event.payload
        if event.type == "llm_call":
            move = p.get("move", "?")
        elif event.type == "execution_result":
            outcome = p.get("classification", "?")
        elif event.type == "score":
            score = p.get("proxy_score")
    return f"attempt {n}: move={move} outcome={outcome} proxy={score}"


def run_skeleton(manifest: RunManifest, groups: dict[int, list[TraceEvent]]) -> str:
    """The run's factual spine — exact, no LLM, one line per attempt."""
    lines = [f"task: {manifest.task_id} (benchmark ale_bench)"]
    lines += [_skeleton_line(n, evs) for n, evs in sorted(groups.items())]
    # heldout_score must NEVER enter writer input: cases are retrieved into
    # running loops, and only validate_submission's (ok, reason) may reach one.
    lines.append(f"final: proxy={manifest.proxy_score} status={manifest.status}")
    return "\n".join(lines)


def best_line(groups: dict[int, list[TraceEvent]]) -> str:
    """The episode's ground-truth footer, written by code and never by the LLM."""
    best_n = best = None
    for n, evs in sorted(groups.items()):
        for e in evs:
            if e.type == "score":
                s = e.payload.get("proxy_score")
                if s is not None and (best is None or s > best):
                    best_n, best = n, s
    if best is None:
        return "Best: none - no attempt scored"
    return f"Best: attempt {best_n} -> {best}"


class NodeCache:
    """Content-addressed cache of LLM nodes. An optimisation, never identity."""

    def __init__(self, root: Path | None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self._lock = threading.Lock()

    @staticmethod
    def key(llm_key: tuple, prompt_name: str, system_text: str,
            prompt_text: str, inputs: list[str]) -> str:
        """Cache key = the full derivation of the completion."""
        payload = json.dumps([list(llm_key), prompt_name, system_text, prompt_text, inputs],
                             sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, key: str) -> dict | None:
        """The cached {text, tokens} node, or None."""
        if not self.root:
            return None
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        with self._lock:
            self.hits += 1
        return json.loads(path.read_text())

    def put(self, key: str, text: str, tokens: int) -> None:
        """Store a node atomically (concurrent builders may race on one key)."""
        if not self.root:
            return
        tmp = self.root / f"{key}.json.tmp{threading.get_ident()}"
        tmp.write_text(json.dumps({"text": text, "tokens": tokens}, ensure_ascii=False))
        tmp.replace(self.root / f"{key}.json")


class EpisodeWriter:
    """Distils a corpus of ok runs into one episode case per run."""

    def __init__(self, corpus: list[tuple[RunManifest, Path]], llm,
                 params: WriteParams = WriteParams(),
                 prompts: WritePrompts = WritePrompts(),
                 nodes_dir: Path | None = None,
                 llm_key: tuple | None = None,
                 workers: int = 8,
                 retry_wall_s: float = 1800.0,
                 retry_sleep_s: float = 30.0):
        self.corpus = sorted(corpus, key=lambda t: t[0].run_id)
        self.llm = llm
        self.params = params
        self.prompts = prompts
        self.cache = NodeCache(nodes_dir)
        # the cache key must pin everything that decides a completion, so it
        # is derived from the client rather than passed in beside it: a key
        # that disagreed with the client would serve another model's notes
        self.llm_key = llm_key or (llm.model, getattr(llm, "temperature", 0.0), 0)
        self.workers = max(1, workers)
        self.retry_wall_s = retry_wall_s
        self.retry_sleep_s = retry_sleep_s
        self.build_cost_tokens = 0
        self.llm_calls = 0
        self._stats_lock = threading.Lock()

    # ---------------------------------------------------------------- plumbing

    def _complete(self, prompt_name: str, user: str) -> str:
        """One node: cache lookup, then the LLM behind a retry wall."""
        key = NodeCache.key(self.llm_key, prompt_name, self.prompts.system,
                            getattr(self.prompts, prompt_name), [user])
        cached = self.cache.get(key)
        if cached is not None:
            return cached["text"]
        t0 = time.monotonic()
        tries = 0
        while True:
            try:
                resp = self.llm.complete(self.prompts.system, user)
                break
            except Exception as e:  # noqa: BLE001 — filtered by _is_transient
                if not is_transient(e):
                    raise
                tries += 1
                elapsed = time.monotonic() - t0
                if elapsed >= self.retry_wall_s:
                    raise
                log.warning("writer %s call failed (%s); retry %d, %.0fs into the %.0fs wall",
                            prompt_name, type(e).__name__, tries, elapsed, self.retry_wall_s)
                time.sleep(self.retry_sleep_s)
        with self._stats_lock:
            self.build_cost_tokens += resp.total_tokens
            self.llm_calls += 1
        text = resp.text.strip()
        self.cache.put(key, text, resp.total_tokens)
        return text

    def _map(self, fn, items: list):
        if self.workers == 1 or len(items) <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(fn, items))

    def _enforce_cap(self, text: str, cap: int, label: str) -> str:
        """Cap by prompt, then one `shorten` rewrite, then drop whole ("")."""
        text = text.strip()
        if len(text) <= cap:
            return text
        try:
            shortened = self._complete(
                "shorten", self.prompts.shorten.format(text=text, cap=cap)).strip()
        except Exception as e:  # noqa: BLE001 — deterministic 4xx from _complete
            status = getattr(e, "status_code", None)
            if status is None or not (400 <= status < 500):
                raise
            log.warning("%s shorten call is unservable (HTTP %s) — DROPPED whole", label, status)
            return ""
        if len(shortened) <= cap:
            log.warning("%s overran its cap (%d > %d chars); shorten brought it to %d",
                        label, len(text), cap, len(shortened))
            return shortened
        log.warning("%s still over cap after shorten (%d > %d chars) — DROPPED whole",
                    label, len(shortened), cap)
        return ""

    # ------------------------------------------------------------------ layers

    def _task_header(self, groups: dict[int, list[TraceEvent]]) -> str:
        """The run's task statement, verbatim, for the writer prompts."""
        payload = None
        for evs in groups.values():
            for e in evs:
                if e.type == "task_presented":
                    payload = e.payload
                    break
        goal = str((payload or {}).get("goal", "") or "").strip()
        ev = str((payload or {}).get("eval", "") or "").strip()
        parts = [goal] if goal else []
        if ev:
            parts.append(f"EVALUATION: {ev}")
        # a blank block under the prompt's label invites a hallucinated task
        return "\n\n".join(parts) or "(task statement unavailable for this run)"

    def _episode_note(self, task: str, skeleton: str,
                      notes: list[tuple[int, str]]) -> str:
        """Fold the step notes into a running summary, `fold_window` at a time."""
        F, cap = self.params.fold_window, self.params.episode_chars
        lines = skeleton.splitlines()
        header, attempts, final = lines[0], lines[1:-1], lines[-1]
        summary = ""
        for i in range(0, max(len(notes), 1), F):
            window = notes[i:i + F]
            # dropped ("") step notes leave the window text; their skeleton
            # lines still cover those attempts factually
            wnotes = "\n\n".join(f"[attempt {n}] {note}" for n, note in window if note)
            wskel = "\n".join([header] + attempts[i:i + F] + [final])
            summary = self._enforce_cap(self._complete("fold", self.prompts.fold.format(
                task=task, summary=summary or "(start of episode)", skeleton=wskel,
                notes=wnotes, cap=cap)), cap, "fold summary")
        return summary

    def _retrieval_context(self, task: str, outcome: str) -> str:
        """The situating prefix: indexed with the case, never injected into a prompt."""
        cap = self.params.retrieval_context_chars
        ctx = self._complete("contextualize", self.prompts.contextualize.format(
            task=task, outcome=outcome, cap=cap)).strip()
        if len(ctx) > cap:
            ctx = self._complete(
                "shorten", self.prompts.shorten.format(text=ctx, cap=cap)).strip()
        if len(ctx) > 2 * cap:
            # index-only tolerance band: over N is kept up to 2N, never cut
            log.warning("retrieval context is %d chars (> 2N=%d) — DROPPED", len(ctx), 2 * cap)
            return ""
        return ctx

    def _case_for_run(self, item: tuple[RunManifest, Path]) -> Case | None:
        m, run_dir = item
        groups: dict[int, list[TraceEvent]] = defaultdict(list)
        for event in read_events(run_dir):
            groups[event.attempt].append(event)
        groups = dict(groups)
        task = self._task_header(groups)
        skeleton = run_skeleton(m, groups)
        notes = []
        for n, evs in sorted(groups.items()):
            text, _ = attempt_slice(evs)
            note = self._enforce_cap(self._complete("step", self.prompts.step.format(
                task=task, context="", slice=text, cap=self.params.c_step)),
                self.params.c_step, "step note")
            notes.append((n, note))
        note = self._episode_note(task, skeleton, notes)
        if not note:
            return None
        # the mechanical footer sits OUTSIDE the cap deliberately: capping after
        # appending would let a shorten call eat the one line that must survive
        content = f"{note}\n{best_line(groups)}"
        seqs = [e.seq for evs in groups.values() for e in evs]
        span = [min(seqs), max(seqs)] if seqs else [0, 0]
        ctx = self._retrieval_context(task, skeleton.splitlines()[-1])
        return make_case(content, m.task_id, [{"run_id": m.run_id, "seq_span": span}],
                         {"retrieval_context": ctx})

    # ------------------------------------------------------------------ dedupe

    def _dedupe(self, cases: list[Case]) -> list[Case]:
        """Merge-instead-of-insert: one bounded LLM verdict per incoming case."""
        if self.params.dedupe != "llm":
            return cases
        cap = self.params.episode_chars
        recall = BM25Retriever()
        kept: list[Case] = []
        for case in cases:
            # merges are WITHIN-TASK only: a merged case keeps one task_id, so a
            # cross-task merge would index the absorbed content under the wrong task
            pool = Bank(bank_id="dedupe-pool",
                        cases=[c for c in kept if c.task_id == case.task_id],
                        manifest={}, dir=None)
            neighbours = [c for c, _ in recall.query(case.content, pool, self.params.dedupe_k)]
            hit = self._merge_decision(case, neighbours, cap)
            if hit is None:
                kept.append(case)
                continue
            anchor, text = hit
            idx = next(i for i, c in enumerate(kept) if c.id == anchor.id)
            kept[idx] = make_case(text, anchor.task_id,
                                  anchor.provenance + case.provenance, dict(anchor.meta))
        return kept

    def _merge_decision(self, case: Case, neighbours: list[Case],
                        cap: int) -> tuple[Case, str] | None:
        if not neighbours:
            return None
        listed = "\n\n".join(f"[{i + 1}] {c.content}" for i, c in enumerate(neighbours))
        reply = self._complete("merge", self.prompts.merge.format(
            new=case.content, neighbours=listed, cap=cap))
        m = re.match(r"\s*MERGE\s+(\d+)\s*\n(.+)", reply, re.DOTALL)
        if not m:
            return None                       # NO_MERGE or unparseable: keep both
        idx, text = int(m.group(1)), m.group(2).strip()
        if not 1 <= idx <= len(neighbours) or len(text) < self.params.min_case_chars:
            return None
        if len(text) > cap:
            # truncating would amputate the merged applicability conditions;
            # rejecting loses nothing (the duplicate simply survives)
            log.warning("dedupe merge rejected: merged text %d chars > cap %d", len(text), cap)
            return None
        return neighbours[idx - 1], text

    # -------------------------------------------------------------------- main

    def cases(self) -> list[Case]:
        """One episode case per ok run, deduped within task."""
        built = [c for c in self._map(self._case_for_run, self.corpus) if c is not None]
        return self._dedupe(built)

    def derivation(self) -> dict:
        """The write-side half of the bank's derivation record."""
        return {"write_params": asdict(self.params),
                "write_prompts_hash": write_prompts_hash(self.prompts),
                "llm": {"model": self.llm_key[0], "temperature": self.llm_key[1],
                        "seed": self.llm_key[2]}}
