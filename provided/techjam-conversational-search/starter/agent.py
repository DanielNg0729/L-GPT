"""Entry point the official evaluator imports (`from starter.agent import Agent`).

The real system lives in `copilot/` at the repository root; this file only wires the
two together, so `provided/` stays as close to the shipped kit as possible.

    python -m evaluator.local_evaluator              # scores the copilot
    BASELINE=1 python -m evaluator.local_evaluator   # scores the organizer's weak BM25

The `BASELINE` switch re-exports `starter/weak_bm25.py`, which is the original starter
agent preserved verbatim so `docs/baseline_results.json` stays reproducible.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if os.environ.get("BASELINE"):
    from .weak_bm25 import Agent  # noqa: F401
else:
    from copilot import Agent  # noqa: F401

__all__ = ["Agent"]
