"""The disclosure schedule is decided at import time from the DISCLOSURE variable."""
import importlib
import os
import sys
import unittest

import submission.agent as agent_mod


class TestDisclosureToggle(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("DISCLOSURE", None)
        importlib.reload(agent_mod)
        # Reloading submission.agent replaces its Agent class object; any module that
        # already re-exported the old one (starter.agent) must be reloaded too, or the
        # contract test's identity assertion sees two different classes.
        entry = sys.modules.get("starter.agent")
        if entry is not None:
            importlib.reload(entry)

    def _reload(self, value):
        os.environ.pop("DISCLOSURE", None)
        if value is not None:
            os.environ["DISCLOSURE"] = value
        return importlib.reload(agent_mod)

    def test_default_is_sequential(self):
        """A fresh clone must reproduce the runbook's official 0.9715 unchanged."""
        mod = self._reload(None)
        self.assertEqual(mod.Agent.DISCLOSURE, (1,) * 9 + (10,))

    def test_sequential_explicit(self):
        mod = self._reload("sequential")
        self.assertEqual(mod.Agent.DISCLOSURE, (1,) * 9 + (10,))

    def test_full_opt_in(self):
        mod = self._reload("full")
        self.assertEqual(mod.Agent.DISCLOSURE, (10,) * 10)

    def test_schedule_never_exceeds_contract(self):
        for value in (None, "sequential", "full"):
            mod = self._reload(value)
            self.assertEqual(len(mod.Agent.DISCLOSURE), 10)
            self.assertTrue(all(1 <= w <= 10 for w in mod.Agent.DISCLOSURE), value)


if __name__ == "__main__":
    unittest.main()
