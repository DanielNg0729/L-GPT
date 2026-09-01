"""Record the provenance of an evaluation run, next to the `results.json` it describes.

WHY THIS EXISTS. The submission rules require teams to retain the generated `results.json`
"together with the submitted commit hash and relevant environment and execution details".
`results.json` carries the metrics and the per-session results but says nothing about which
commit produced it or what it ran on, and the evaluator is organizer-owned, so it cannot be
made to emit that. Without a record written at the same time, answering "which commit was
this, on what Python?" weeks later is guesswork.

    python -m evaluator.local_evaluator
    python tools/record_run.py

Writes `results_provenance.json` beside `results.json`. Keep both.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True,
                             timeout=15)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def main() -> None:
    results = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results.json")
    if not results.is_file():
        raise SystemExit(f"{results.name} not found. Run the evaluator first:\n"
                         f"    python -m evaluator.local_evaluator")
    scored = json.loads(results.read_text(encoding="utf-8"))

    dirty = git("status", "--porcelain")
    record = {
        "results_file": results.name,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git("rev-parse", "HEAD"),
        # A dirty tree means the numbers do not correspond to any commit, which is the one
        # thing this file exists to make impossible to overlook later.
        "working_tree_clean": dirty == "" if dirty is not None else None,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "reported": {
            "recommended_technical_score": scored.get("recommended_technical_score"),
            "hit_rate_at_10": scored.get("hit_rate_at_10"),
            "mrr": scored.get("mrr"),
            "mttc": scored.get("mttc"),
            "sample_count": scored.get("sample_count"),
            "reported_token_usage": scored.get("reported_token_usage"),
        },
    }
    out = results.with_name("results_provenance.json")
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out.name}")
    print(f"  commit  {record['commit']}  (clean: {record['working_tree_clean']})")
    print(f"  python  {record['python']} on {record['platform']}")
    print(f"  score   {record['reported']['recommended_technical_score']}")
    if record["working_tree_clean"] is False:
        print("  WARNING: the working tree is dirty, so these numbers do not correspond "
              "to the recorded commit.")


if __name__ == "__main__":
    main()
