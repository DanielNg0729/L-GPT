"""The organizer-owned files must stay byte-identical to the released kit.

This is a rules check, not a behaviour check. `tools/verify_upstream_integrity.py` can be
run by hand; this makes the same assertion part of the suite so a modification cannot pass
review unnoticed.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class UpstreamIntegrityTest(unittest.TestCase):
    def test_protected_files_are_unmodified(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "verify_upstream_integrity.py")],
            capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(result.returncode, 0,
                         f"organizer-owned file changed:\n{result.stdout}\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
