from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from robustness.build_sets import scenario_sequence, weighted_without_replacement


ROOT = Path(__file__).resolve().parents[1]


class RobustnessBuilderTest(unittest.TestCase):
    def test_weighted_sample_is_distinct_and_deterministic(self) -> None:
        items = [f"P{i}" for i in range(30)]
        weights = [i + 1 for i in range(30)]
        first = weighted_without_replacement(items, weights, 20, seed=7)
        second = weighted_without_replacement(items, weights, 20, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_private_scenario_mix_is_exact(self) -> None:
        mix = {"buying": .40, "browsing": .40, "intent_override": .15, "boundary": .05}
        scenarios = scenario_sequence(800, mix, seed=9)
        self.assertEqual(scenarios.count("buying"), 320)
        self.assertEqual(scenarios.count("browsing"), 320)
        self.assertEqual(scenarios.count("intent_override"), 120)
        self.assertEqual(scenarios.count("boundary"), 40)

    def test_generated_sets_match_manifest_and_public_safety_invariants(self) -> None:
        sets_dir = ROOT / "robustness" / "sets"
        manifest = json.loads((sets_dir / "manifest.json").read_text(encoding="utf-8"))
        public_rows = [json.loads(line) for line in
                       (ROOT / "data" / "public_set.jsonl").read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        public_targets = {row["ground_truth"]["parent_asin"] for row in public_rows}

        self.assertEqual(manifest["disclosed_candidate_target_pool"], 1406)
        self.assertEqual(manifest["unseen_proxy_pool"], 1206)
        self.assertEqual(manifest["private_like_sessions"], 800)
        for name, info in manifest["sets"].items():
            path = sets_dir / info["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), info["sha256"])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            targets = [row["ground_truth"]["parent_asin"] for row in rows]
            scenarios = [row["scenario_type"] for row in rows]
            self.assertEqual(len(rows), 800, name)
            self.assertEqual(len(set(targets)), 800, name)
            self.assertFalse(set(targets) & public_targets, name)
            self.assertEqual(scenarios.count("buying"), 320, name)
            self.assertEqual(scenarios.count("browsing"), 320, name)
            self.assertEqual(scenarios.count("intent_override"), 120, name)
            self.assertEqual(scenarios.count("boundary"), 40, name)


if __name__ == "__main__":
    unittest.main()
