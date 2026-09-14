"""Budget charging and exhaustion."""
from __future__ import annotations

import pytest

from arloop.budget import Budget


def test_token_budget_exhausts_on_its_own_axis():
    b = Budget("tokens", 100)
    b.charge(tokens=60, attempts=5)
    assert not b.exhausted()
    b.charge(tokens=40)
    assert b.exhausted()


def test_attempt_budget_ignores_tokens():
    b = Budget("attempts", 2)
    b.charge(tokens=10_000_000, attempts=1)
    assert not b.exhausted()
    b.charge(attempts=1)
    assert b.exhausted()
    assert b.spent_on_axis() == 2


def test_every_axis_is_tracked_regardless_of_kind():
    b = Budget("tokens", 100)
    b.charge(tokens=1, attempts=2, exec_seconds=3.5, grade_seconds=0.25)
    snap = b.snapshot()
    assert snap["type"] == "tokens" and snap["limit"] == 100
    assert (snap["tokens"], snap["attempts"]) == (1, 2)
    assert (snap["exec_seconds"], snap["grade_seconds"]) == (3.5, 0.25)
    assert snap["wall_seconds"] >= 0.0


def test_unknown_kind_and_bad_limit_refused():
    with pytest.raises(ValueError):
        Budget("wall_seconds", 100)
    with pytest.raises(ValueError):
        Budget("tokens", 0)
