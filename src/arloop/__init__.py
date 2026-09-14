"""Budgeted agent loop, episode memory, and the allocation grid.

The loop runs a greedy draft/debug/improve chain under a token budget against
an arbench task; the memory module builds and retrieves episode cases from
earlier runs; the grid sweeps one arm's (task, seed) cells against the trace
manifests as its ledger.
"""

__version__ = "1.0.0"
