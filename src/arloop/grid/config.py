"""One YAML per arm, loaded into a frozen GridConfig.

config_hash is the arm's identity in the ledger: every field that changes what
a run produces, with the prompts resolved to their full text. Excluded are
name/tasks/seeds (the cell key already carries task and seed) and ops
(retries, workers, paths — they decide whether a run survives, never what it
produces).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from arloop.prompts import PromptSet

BUDGET_TYPES = ("tokens", "attempts")


@dataclass(frozen=True)
class GridConfig:
    name: str
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    model: str
    temperature: float = 0.5
    thinking: bool = False
    thinking_token_budget: int | None = None
    history_write_tokens: int = 100
    chars_per_token: float = 3.5
    branch_policy: str = "best"
    run_budget: dict[str, Any] = field(
        default_factory=lambda: {"type": "tokens", "limit": 100_000})
    exec_timeout_s: int = 900
    code_budget_chars: int = 30_000
    sandbox: bool = True
    memory: dict[str, Any] | None = None
    prompts: dict[str, str] | None = None
    ops: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Refuse configs whose failure would otherwise be invisible in the ledger."""
        if not self.model:
            raise ValueError("pin `model` explicitly in the arm config")
        if not self.tasks or not self.seeds:
            raise ValueError("tasks and seeds must be non-empty")
        if self.run_budget.get("type") not in BUDGET_TYPES:
            raise ValueError(
                f"unknown budget type {self.run_budget.get('type')!r}; "
                f"have {list(BUDGET_TYPES)}")
        if not isinstance(self.run_budget.get("limit"), int) or self.run_budget["limit"] <= 0:
            raise ValueError(
                f"run_budget.limit must be a positive int, "
                f"got {self.run_budget.get('limit')!r}")
        if self.branch_policy not in ("last", "best"):
            raise ValueError(
                f"branch_policy must be last|best, got {self.branch_policy!r}")
        if not isinstance(self.history_write_tokens, int) or self.history_write_tokens < 0:
            raise ValueError(
                f"history_write_tokens must be a non-negative int, "
                f"got {self.history_write_tokens!r}")
        if self.chars_per_token <= 0:
            raise ValueError(
                f"chars_per_token must be > 0, got {self.chars_per_token!r}")
        if self.thinking_token_budget is not None:
            budget = self.thinking_token_budget
            if not isinstance(budget, int) or budget <= 0:
                raise ValueError(
                    f"thinking_token_budget must be a positive int, got {budget!r}")
            if self.thinking is not True:
                # a budget with thinking off is a silent no-op: the chat
                # template never opens a reasoning span to bound, so the arm
                # would look like it capped reasoning and would not have
                raise ValueError(
                    "thinking_token_budget requires thinking: true")
        mem = self.memory or {}
        if mem:
            for key in ("bank", "embed_model", "reranker_model", "retrieval_tokens"):
                if key not in mem:
                    raise ValueError(f"memory is missing {key!r}")
            rt = mem["retrieval_tokens"]
            if not isinstance(rt, int) or rt <= 0:
                raise ValueError(
                    f"memory.retrieval_tokens must be a positive int, got {rt!r}")
            if not isinstance(mem.get("loto", False), bool):
                raise ValueError("memory.loto must be a boolean")


def resolved_prompts(cfg: GridConfig) -> PromptSet:
    """The PromptSet this arm actually runs: package defaults plus its overrides."""
    return PromptSet(**cfg.prompts) if cfg.prompts else PromptSet()


def config_hash(cfg: GridConfig) -> str:
    """The arm's ledger identity: every scientific field, prompts resolved."""
    d = asdict(cfg)
    for key in ("name", "tasks", "seeds", "ops"):
        d.pop(key)
    # the prompt TEXT is identity: an edit must re-key the arm rather than
    # silently alias two different experiments onto one ledger cell
    d["prompts"] = asdict(resolved_prompts(cfg))
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_config(path: str | Path) -> GridConfig:
    """Read an arm YAML, freeze its sequences, and validate it."""
    raw = yaml.safe_load(Path(path).read_text())
    for key in ("tasks", "seeds"):
        if raw.get(key) is not None:
            raw[key] = tuple(raw[key])
    cfg = GridConfig(**raw)
    cfg.validate()
    return cfg
