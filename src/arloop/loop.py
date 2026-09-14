"""The agent loop: a budgeted greedy chain.

    draft -> debug-on-failure / improve-on-success, until the token budget is
    exhausted (checked at attempt boundaries; attempts are atomic, so a run
    overshoots slightly by design). `branch_policy` picks the improve move's
    base: "best" extends the best-scoring attempt so far, "last" the chain
    tip. Debug always repairs the tip.

All experimental variation lives in the memory view and the config; view=None
IS the memory-free baseline. The trace is the product of a run, not a log.
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from arloop import prompts
from arloop.budget import Budget
from arloop.feedback import Feedback, classify, excerpt as make_excerpt
from arloop.llm import ContextOverflowError
from arloop.prompts import PromptSet
from arloop.runner import Runner, Submission
from arloop.taxonomy import error_signature, normalize_code
from arloop.tokens import approx_tokens
from arloop.trace import RunManifest, TraceWriter, run_dir_for

log = logging.getLogger(__name__)

SOLUTION_NAME = "solution.py"

#: format-only submission verdict; the ONLY grader output a running loop sees
Validate = Callable[[Path], "tuple[bool, Optional[str]]"]
#: official evaluation on PUBLIC inputs -> an arbench Score. Typed loosely so
#: this module imports no benchmark: the grid glue passes the callables in.
PublicEval = Callable[[Path], object]
#: held-out grade, run once at the end on the best attempt's submission
Grade = Callable[[Path], object]


@dataclass
class TaskSpec:
    """The loop's view of a task — prose only, so no benchmark import."""

    task_id: str
    goal: str
    eval: str
    submission_filename: str = "submission.py"


@dataclass
class RunConfig:
    """Everything about a run that is not the task, budget or LLM."""

    run_id: str
    config_hash: str
    seed: int
    traces_root: Path
    workspace_root: Path
    command: str = "python solution.py"
    exec_timeout_s: int = 900
    code_budget_chars: int = 30_000
    branch_policy: str = "best"           # best | last
    history_write_tokens: int = 100
    cpu_cap: int = 1
    compute: str = "cpu"
    prompts: PromptSet = field(default_factory=PromptSet)
    llm_retry_wall_s: float = 1800.0
    llm_retry_sleep_s: float = 30.0
    manifest_extra: dict = field(default_factory=dict)


@dataclass
class RunOutcome:
    """The run's identity and its two scores."""

    run_id: str
    status: str                           # ok | failed
    proxy_score: float | None
    heldout_score: float | None
    n_attempts: int
    run_dir: Path
    failure_reason: str | None = None     # the infra failure, when status is failed


@dataclass
class _Attempt:
    index: int
    code: str
    feedback: Feedback
    workspace: Path | None                # None when the reply carried no code
    approach: str | None = None


def _md5_12(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


class _Run:
    """One run's mutable state; run_task drives it."""

    def __init__(self, task: TaskSpec, view, runner: Runner, budget: Budget,
                 config: RunConfig, llm, writer: TraceWriter,
                 public_eval: PublicEval, validate: Validate):
        self.task, self.view, self.runner = task, view, runner
        self.budget, self.config, self.llm = budget, config, llm
        self.writer, self.public_eval, self.validate = writer, public_eval, validate
        self.workspace_root = Path(config.workspace_root) / config.run_id
        self.system = prompts.render_system(
            config.prompts, config.exec_timeout_s, config.code_budget_chars,
            config.cpu_cap, config.compute)
        self.cases_text = ""
        self.attempts: list[_Attempt] = []
        self._base: _Attempt | None = None
        self._move: str | None = None
        # the solution written this attempt, carried to record_attempt
        self._written: tuple[str, str | None] = ("", None)   # code, approach
        self.infra_failure: str | None = None
        self.tool_call_missing = 0

    def _task_text(self) -> str:
        return f"{self.task.goal}\n{self.task.eval}"

    def _best_attempt(self) -> _Attempt | None:
        """Max proxy, earliest on ties — the same selection finalise uses."""
        scored = [a for a in self.attempts if a.feedback.proxy_score is not None]
        return (max(scored, key=lambda a: a.feedback.proxy_score)
                if scored else None)

    def next_move(self) -> "tuple[str, _Attempt | None]":
        """The move for the next attempt and the attempt it branches from."""
        prev = self.attempts[-1] if self.attempts else None
        if prev is None:
            self._base = None
            return "draft", None
        if prev.feedback.kind == "success":
            base = ((self._best_attempt() or prev)
                    if self.config.branch_policy == "best" else prev)
            self._base = base
            return "improve", base
        # debug is tagged by what failed: the reply's formatting (no code was
        # ever executed, so there is no workspace) or the code itself
        self._base = prev
        return ("debug:formatting" if prev.workspace is None else "debug:code"), prev

    def retrieve(self, move: str, base: "_Attempt | None") -> None:
        """Re-query the bank before every prompt, keyed on THIS move.

        Retrieval is conditioned on the agent's intention for the attempt,
        not on the task alone: the draft has no intention text yet, a code
        debug keys on the error, improve on the code it is extending. A
        formatting debug keys on the task alone: the "error" there is our own
        protocol message, not a retrieval key.
        """
        if self.view is None:
            return
        if move == "debug:code":
            query = f"{base.feedback.excerpt}\n{self._task_text()}"
        elif move == "improve":
            query = f"{self._task_text()}\n{base.code}"
        else:
            query = self._task_text()
        scored = self.view.query(query)
        self.writer.emit("memory_retrieval", {
            "query": query, "move": move,
            "realised_tokens": sum(approx_tokens(c.content) for c, _ in scored),
            "returned": [{"case_id": c.id, "score": score,
                          "tokens": approx_tokens(c.content),
                          "task_id": c.task_id} for c, score in scored]},
            attempt=len(self.attempts))
        self.cases_text = prompts.render_cases([c for c, _ in scored],
                                               self.config.prompts)

    def _history_entries(self) -> list:
        out = []
        for i, a in enumerate(self.attempts):
            prev = self.attempts[i - 1] if i else None
            if i == 0:
                move = "draft"
            elif prev.feedback.kind == "success":
                move = "improve"
            else:
                move = "debug:formatting" if prev.workspace is None else "debug:code"
            outcome = (f"proxy {a.feedback.proxy_score}"
                       if a.feedback.kind == "success" else "failed")
            out.append((a.index + 1, move, outcome, a.approach))
        return out

    def next_prompt(self, move: str, base: "_Attempt | None") -> str:
        """Render this attempt's user prompt."""
        self._move = move
        ps = self.config.prompts
        if move == "draft":
            return prompts.render_draft(ps, self.task.goal, self.task.eval,
                                        self.cases_text)
        history = prompts.render_history(self._history_entries(), ps)
        if move == "improve":
            fb = base.feedback
            prev_score = (f"{fb.proxy_score} ({fb.eval_note})" if fb.eval_note
                          else str(fb.proxy_score))
            return prompts.render_improve(ps, self.task.goal, self.task.eval,
                                          self.cases_text, history, base.code,
                                          prev_score)
        return prompts.render_debug(ps, self.task.goal, self.task.eval,
                                    self.cases_text, history, base.code,
                                    base.feedback.excerpt)

    def call_llm(self, move: str, user: str):
        """One completion, riding out provider outages on a wall budget.

        Provider failures are uncharged and retried; past the wall the run
        fails gracefully with best-so-far. A context overflow is
        deterministic, so it stops immediately instead of burning the wall.
        """
        n = len(self.attempts)
        t0 = time.monotonic()
        tries = 0
        while True:
            try:
                t_call = time.perf_counter()
                resp = self.llm.complete(self.system, user)
                call_ms = round((time.perf_counter() - t_call) * 1000.0, 1)
                break
            except ContextOverflowError as exc:
                detail = str(exc).replace("\n", " ")[:200]
                self.writer.emit("llm_call", {
                    "move": move, "error": f"context_overflow: {detail}",
                    "llm_retry": tries, "retry_elapsed_s": 0.0}, attempt=n)
                self.infra_failure = f"context_overflow: {detail}"
                return None
            except Exception as exc:  # noqa: BLE001 — provider weather
                elapsed = time.monotonic() - t0
                self.writer.emit("llm_call", {
                    "move": move, "error": f"{type(exc).__name__}: {exc}",
                    "llm_retry": tries,
                    "retry_elapsed_s": round(elapsed, 1)}, attempt=n)
                tries += 1
                if elapsed >= self.config.llm_retry_wall_s:
                    self.infra_failure = (
                        f"LLM provider failure persisted past "
                        f"{self.config.llm_retry_wall_s:.0f}s ({tries} rounds)")
                    return None
                # jittered: a fixed interval resynchronises every concurrent
                # worker onto the same retry window of a saturated endpoint
                time.sleep(self.config.llm_retry_sleep_s
                           * (0.5 + random.random()))
        self.budget.charge(tokens=resp.total_tokens)
        self.writer.emit("llm_call", self._call_payload(move, user, resp, call_ms),
                         attempt=n)
        if resp.thinking_ignored:
            # a silently unhonoured setting is indistinguishable in the
            # results from one that worked
            log.warning("attempt %d: thinking was requested OFF but the reply "
                        "carries reasoning (%d chars) — this arm is NOT "
                        "running the mode its config declares",
                        n, resp.reasoning_chars)
        if resp.tool_call_missing:
            log.warning("attempt %d: forced submit_solution tool_choice "
                        "returned NO tool call (finish_reason=%s) — falling "
                        "back to prose extraction", n, resp.finish_reason)
            self.tool_call_missing += 1
        return resp

    def _call_payload(self, move: str, user: str, resp, call_ms: float) -> dict:
        """The llm_call trace event for one completion."""
        payload = {
            "move": move,
            "base_attempt": self._base.index if self._base else None,
            "system": self.system, "prompt": user, "prompt_fp": _md5_12(user),
            "response": resp.text, "finish_reason": resp.finish_reason,
            "reasoning_chars": resp.reasoning_chars,
            "reasoning_stopped": resp.reasoning_stopped,
            "thinking_ignored": resp.thinking_ignored,
            "tool_call_missing": resp.tool_call_missing,
            "call_ms": call_ms,
            "tokens": {"prompt": resp.prompt_tokens,
                       "completion": resp.completion_tokens}}
        if resp.tool_args is not None:
            payload["tool_args"] = resp.tool_args
        if resp.tool_args_raw:
            payload["tool_args_raw"] = resp.tool_args_raw
        return payload

    def _reject(self, classification: str, message: str,
                code: str | None = None) -> None:
        """A malformed or over-budget reply consumed the turn: charge it,
        record it, and let the next move debug from it."""
        n = len(self.attempts)
        self.budget.charge(attempts=1)
        self.writer.emit("execution_result", {
            "exit_code": None, "classification": classification,
            "error_text": message, "excerpt": message,
            "error_sig_hash": None, "error_sig_norm": None,
            "log_ref": None}, attempt=n)
        if code is None:
            code = self.attempts[-1].code if self.attempts else ""
        self.attempts.append(_Attempt(
            n, code, Feedback("agent_fixable", None, message, message), None))

    def write_solution(self, resp) -> Path | None:
        """Extract the solution and write it; None means the turn was spent."""
        n = len(self.attempts)
        approach: str | None = None
        if resp.tool_args is not None:
            code = resp.tool_args.get("code")
            code = code.strip() if isinstance(code, str) else ""
            if not code:
                self._reject("no_code",
                             "submit_solution tool call had empty/missing code")
                return None
            code += "\n"
            raw_approach = resp.tool_args.get("approach")
            approach = raw_approach if isinstance(raw_approach, str) else None
        elif resp.tool_args_raw:
            self._reject("malformed_tool_call",
                         "submit_solution tool call returned unparseable "
                         "arguments")
            return None
        else:
            try:
                code = prompts.extract_python_block(resp.text or "")
            except ValueError as exc:
                self._reject("no_code", str(exc))
                return None
        if len(code) > self.config.code_budget_chars:
            # the ONLY bound on generated code size: the write side never
            # truncates, so this is a rejected turn, not a silent cut
            self._reject("code_over_budget",
                         f"solution.py was {len(code)} characters, over the "
                         f"{self.config.code_budget_chars}-character limit — "
                         f"submit a shorter solution")
            return None
        attempt_dir = self.workspace_root / f"attempt_{n}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / SOLUTION_NAME).write_text(code)
        form = normalize_code(code)
        best = self._best_attempt()
        self.writer.emit("code_written", {
            "path": SOLUTION_NAME, "content": code, "approach": approach,
            "code_fp": form.fingerprint, "code_mode": form.mode,
            "move": self._move,
            "parent_idx": self._base.index if self._base else -1,
            "incumbent_idx": best.index if best else -1}, attempt=n)
        self._written = (code, approach)
        return attempt_dir

    def execute(self, attempt_dir: Path) -> "tuple[Submission, Feedback]":
        """Run the attempt once. There is no uncharged execution retry."""
        n = len(self.attempts)
        self.writer.emit("submit", {"command": self.config.command,
                                    "timeout_s": self.config.exec_timeout_s},
                         attempt=n)
        sub = self.runner.submit(attempt_dir, self.config.command,
                                 self.config.exec_timeout_s)
        return sub, classify(sub)

    def check_submission(self, attempt_dir: Path, fb: Feedback,
                         log_text: str) -> Feedback:
        """A successful attempt must also produce a VALID submission file.

        Only the (ok, reason) verdict crosses back — never a score — and it
        reads like any other failure to the debug prompt. The script's own
        output rides along, because the agent's stderr usually names the bug.
        """
        if fb.kind != "success":
            return fb
        n = len(self.attempts)
        path = attempt_dir / self.task.submission_filename
        if not path.exists():
            ok, reason = False, f"{self.task.submission_filename} was not written"
        else:
            ok, reason = self.validate(path)
        self.writer.emit("submission_validation", {"ok": ok, "reason": reason},
                         attempt=n)
        if ok:
            return fb
        text = (f"The script ran to completion, but "
                f"{self.task.submission_filename} failed validation: {reason}")
        tail = make_excerpt(log_text) if log_text.strip() else ""
        if tail:
            text = f"{text}\nScript output:\n{tail}"
        return Feedback("agent_fixable", None, text, text)

    def apply_public_eval(self, attempt_dir: Path,
                          fb: Feedback) -> Feedback | None:
        """The official score on PUBLIC inputs — the loop's proxy.

        Everything the Score carries is built from public data the agent can
        already read, which is what makes forwarding it the one sanctioned
        crossing of the score firewall. None means an eval INFRA fault: the
        cell dies loudly rather than feeding an operator error to the agent.
        """
        if fb.kind != "success":
            return fb
        n = len(self.attempts)
        t_eval = time.perf_counter()
        score = self.public_eval(attempt_dir / self.task.submission_filename)
        eval_seconds = time.perf_counter() - t_eval
        self.budget.charge(grade_seconds=eval_seconds)
        details = getattr(score, "details", None) or {}
        if details.get("infra"):
            self.infra_failure = ("public_eval infra: "
                                  + str(details.get("reason", "unknown")))
            self.writer.emit("public_eval", {"valid": False, "infra": True,
                                             "reason": details.get("reason")},
                             attempt=n)
            return None
        note = details.get("feedback") or ""
        self.writer.emit("public_eval", {
            "valid": bool(score.valid), "value": score.value,
            "is_higher_better": bool(score.is_higher_better),
            "grade_seconds": round(eval_seconds, 3),
            "n_cases": details.get("n_cases"), "n_tle": details.get("n_tle"),
            "n_error": details.get("n_error"),
            "n_rejected": details.get("n_rejected"),
            "feedback": note, "cases": details.get("cases")}, attempt=n)
        if not score.valid:
            text = note or str(details.get("reason",
                                           "official public evaluation failed"))
            return Feedback("agent_fixable", None, text, text)
        proxy = float(score.value)
        if not score.is_higher_better:
            proxy = -proxy          # the loop's proxy is always higher-is-better
            note = ((note + " — ") if note else "") + \
                "minimise problem: scores are reported negated, higher is better"
        return Feedback("success", proxy, "", "", eval_note=note)

    def record_attempt(self, attempt_dir: Path, sub: Submission,
                       fb: Feedback) -> None:
        """Charge the attempt, archive its log, and append it to the chain."""
        n = len(self.attempts)
        code, approach = self._written
        self.budget.charge(attempts=1, exec_seconds=sub.duration_s)
        sig_hash, sig_norm = error_signature(fb.error_text, sub.exit_code)
        log_name = f"attempt_{n}.log.gz"
        with gzip.open(self.writer.artifact_path(log_name), "wt") as fh:
            fh.write(sub.log)
        self.writer.emit("execution_result", {
            "exit_code": sub.exit_code, "classification": fb.kind,
            "error_text": fb.error_text, "excerpt": fb.excerpt,
            "error_sig_hash": sig_hash, "error_sig_norm": sig_norm,
            "log_ref": f"artifacts/{log_name}"}, attempt=n)
        self.writer.emit("score", {
            "proxy_score": fb.proxy_score, "eval_note": fb.eval_note,
            "budget_spent": self.budget.snapshot()}, attempt=n)
        self.attempts.append(_Attempt(n, code, fb, attempt_dir, approach))

    def finalise(self, grade: Grade | None) -> RunOutcome:
        """Grade the best attempt held out, then seal the run."""
        best = self._best_attempt()
        proxy = best.feedback.proxy_score if best else None
        heldout = None
        if best is not None and grade is not None:
            submission = best.workspace / self.task.submission_filename
            try:
                t_grade = time.perf_counter()
                if submission.exists():
                    score = grade(submission)
                    heldout = float(score.value) if score.valid else None
                self.budget.charge(grade_seconds=time.perf_counter() - t_grade)
            except Exception as exc:  # noqa: BLE001 — grading is external
                # unguarded, a grader crash would seal the manifest failed
                # with no final_outcome despite real score events
                self.infra_failure = (self.infra_failure
                                      or f"grading_error: {type(exc).__name__}")
        status = "failed" if self.infra_failure else "ok"
        if self.infra_failure and not self.attempts:
            # nothing happened past task_presented: no experimental signal, so
            # discard rather than seal a corpse the driver must skip forever
            self.writer.discard()
            return RunOutcome(self.config.run_id, status, None, None, 0,
                              self.writer.run_dir, self.infra_failure)
        self.writer.emit("final_outcome", {
            "status": status, "infra_failure": self.infra_failure,
            "best_attempt": best.index if best else None,
            "proxy_score": proxy, "heldout_score": heldout,
            "attempts": len(self.attempts),
            "budget_spent": self.budget.snapshot()},
            attempt=self.attempts[-1].index if self.attempts else 0)
        if self.infra_failure:
            self.writer.manifest.extra["failure_reason"] = self.infra_failure
        spent = self.budget.spent
        self.writer.close(
            status, proxy_score=proxy, heldout_score=heldout,
            budget_spent=self.budget.spent_on_axis(),
            cost={"tokens": spent.tokens,
                  "exec_seconds": round(spent.exec_seconds, 3),
                  "grade_seconds": round(spent.grade_seconds, 3),
                  "tool_call_missing": self.tool_call_missing})
        return RunOutcome(self.config.run_id, status, proxy, heldout,
                          len(self.attempts), self.writer.run_dir,
                          self.infra_failure)


def run_task(task: TaskSpec, view, runner: Runner, budget: Budget,
             config: RunConfig, llm, *, public_eval: PublicEval,
             validate: Validate, grade: Grade | None) -> RunOutcome:
    """Run one (task, seed) cell to exhaustion and return its outcome."""
    if config.branch_policy not in ("best", "last"):
        # before the TraceWriter opens: a raise inside it seals a corpse the
        # driver re-mints every pass
        raise ValueError(f"branch_policy must be best|last, "
                         f"got {config.branch_policy!r}")
    manifest = RunManifest(
        run_id=config.run_id, task_id=task.task_id, seed=config.seed,
        config_hash=config.config_hash, model=llm.model,
        # the CLIENT's temperature: it is what the requests actually carry, so
        # a mis-wired cell records what it really sampled at
        temperature=llm.temperature,
        budget={"type": budget.kind, "limit": budget.limit},
        extra={"template_hash": prompts.template_hash(config.prompts),
               **config.manifest_extra})
    run_dir = run_dir_for(config.traces_root, task.task_id, config.run_id)

    with TraceWriter(run_dir, manifest) as writer:
        run = _Run(task, view, runner, budget, config, llm, writer,
                   public_eval, validate)
        writer.emit("task_presented", {"goal": task.goal, "eval": task.eval},
                    attempt=0)
        while not budget.exhausted() and run.infra_failure is None:
            move, base = run.next_move()
            run.retrieve(move, base)
            user = run.next_prompt(move, base)
            resp = run.call_llm(move, user)
            if resp is None:
                break
            attempt_dir = run.write_solution(resp)
            if attempt_dir is None:
                continue
            sub, fb = run.execute(attempt_dir)
            fb = run.check_submission(attempt_dir, fb, sub.log)
            fb = run.apply_public_eval(attempt_dir, fb)
            if fb is None:
                break
            run.record_attempt(attempt_dir, sub, fb)
        return run.finalise(grade)
