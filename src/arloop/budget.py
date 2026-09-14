"""The run's resource budget, checked at attempt boundaries.

Every axis is tracked so a run can be re-plotted at other thresholds;
`exhausted()` reads only the axis the budget was built on.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

KINDS = ("tokens", "attempts")


@dataclass
class Spent:
    tokens: int = 0
    attempts: int = 0
    exec_seconds: float = 0.0
    # grader wall-time (public_eval + heldout grade), separate from
    # exec_seconds (the submission subprocess) so wall clock decomposes
    grade_seconds: float = 0.0


class Budget:
    """A limit on one axis, plus cumulative spend on all of them."""

    def __init__(self, kind: str, limit: int):
        if kind not in KINDS:
            raise ValueError(f"budget kind {kind!r} not in {KINDS}")
        if limit <= 0:
            raise ValueError("budget limit must be > 0")
        self.kind = kind
        self.limit = limit
        self.spent = Spent()
        self._t0 = time.monotonic()

    def charge(self, *, tokens: int = 0, attempts: int = 0,
               exec_seconds: float = 0.0, grade_seconds: float = 0.0) -> None:
        """Add spend on any subset of the axes."""
        self.spent.tokens += tokens
        self.spent.attempts += attempts
        self.spent.exec_seconds += exec_seconds
        self.spent.grade_seconds += grade_seconds

    @property
    def wall_seconds(self) -> float:
        """Seconds since the budget was created."""
        return time.monotonic() - self._t0

    def spent_on_axis(self) -> int:
        """Spend on this budget's own axis; goes into the manifest."""
        return getattr(self.spent, self.kind)

    def exhausted(self) -> bool:
        """Has the budget's own axis reached its limit?"""
        return self.spent_on_axis() >= self.limit

    def snapshot(self) -> dict:
        """Cumulative spend on every axis; attached to each score event."""
        return {"type": self.kind, "limit": self.limit,
                "tokens": self.spent.tokens, "attempts": self.spent.attempts,
                "exec_seconds": round(self.spent.exec_seconds, 3),
                "grade_seconds": round(self.spent.grade_seconds, 3),
                "wall_seconds": round(self.wall_seconds, 3)}
