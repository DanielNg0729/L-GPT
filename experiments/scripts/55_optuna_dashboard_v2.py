"""Launch the local dashboard for the isolated Optuna v2 study."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "studies" / ".ml_deps"))

from optuna_dashboard import run_server  # noqa: E402


if __name__ == "__main__":
    storage = f"sqlite:///{(ROOT / 'experiments' / 'studies' / 'optuna_official_v2.db').as_posix()}"
    run_server(storage, host="127.0.0.1", port=8081)
