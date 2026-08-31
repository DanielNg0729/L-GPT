"""Prove that every organizer-owned file in this repository is byte-identical to the kit.

WHY THIS EXISTS. The rules forbid modifying the evaluator, the public set, and the released
contracts. Saying "we did not modify them" is an assertion; this makes it checkable in one
command, by anyone, without trusting us. The idea is borrowed from a teammate's repository,
which vendors the organizer kit under `provided/` alongside a `SHA256SUMS` and an
`UPSTREAM_COMMIT.txt` -- a good habit worth copying.

It is deliberately NOT a test of our own code. It answers exactly one question: has anything
we were not allowed to touch been touched?

  python tools/verify_upstream_integrity.py            # verify against the manifest
  python tools/verify_upstream_integrity.py --write    # regenerate it (see the warning)

REGENERATING DEFEATS THE PURPOSE unless the kit itself was legitimately re-downloaded. If a
file's hash has drifted, the fix is to restore the file, not to rewrite the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "UPSTREAM_INTEGRITY.sha256"

# Files the competition rules place off-limits. Paths are repo-relative and POSIX-style so
# the manifest is identical on every platform.
PROTECTED = (
    "evaluator/local_evaluator.py",
    "evaluator/__init__.py",
    "data/public_set.jsonl",
    "docs/agent_api_contract.json",
    "docs/baseline_results.json",
    "docs/evaluation_config.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> list[tuple[str, str]]:
    rows = []
    for rel in PROTECTED:
        p = ROOT / rel
        if p.is_file():
            rows.append((rel, digest(p)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="regenerate the manifest instead of verifying it")
    args = ap.parse_args()

    rows = collect()
    if args.write:
        body = "".join(f"{h}  {rel}\n" for rel, h in rows)
        MANIFEST.write_text(
            "# SHA-256 of every organizer-owned file this submission may not modify.\n"
            "# Verify with:  python tools/verify_upstream_integrity.py\n"
            + body, encoding="utf-8")
        print(f"wrote {MANIFEST.name} ({len(rows)} files)")
        return 0

    if not MANIFEST.exists():
        print(f"missing manifest: {MANIFEST}")
        return 1
    expected = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, _, rel = line.partition("  ")
        expected[rel] = h

    ok = True
    seen = set()
    for rel, h in rows:
        seen.add(rel)
        want = expected.get(rel)
        if want is None:
            print(f"  UNLISTED  {rel}")
            ok = False
        elif want != h:
            print(f"  MODIFIED  {rel}")
            print(f"            expected {want}")
            print(f"            actual   {h}")
            ok = False
        else:
            print(f"  ok        {rel}")
    for rel in expected:
        if rel not in seen:
            print(f"  MISSING   {rel} (listed in the manifest, absent from the tree)")
            ok = False

    print("\nAll organizer-owned files are unmodified." if ok
          else "\nINTEGRITY CHECK FAILED. Restore the file; do not rewrite the manifest.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
