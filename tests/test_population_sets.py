from __future__ import annotations

import hashlib
import importlib
import json
import sys
import unittest
from pathlib import Path

from experiments.studies.build_sets import scenario_sequence, weighted_without_replacement
from experiments.studies.build_independent_validation_sets import counts_for


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
        sets_dir = ROOT / "experiments" / "datasets" / "sets"
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

    def test_optuna_v2_folds_are_stratified_and_public_disjoint(self) -> None:
        sets_dir = ROOT / "experiments" / "datasets" / "optuna_sets"
        manifest = json.loads((sets_dir / "manifest.json").read_text(encoding="utf-8"))
        public_rows = [json.loads(line) for line in
                       (ROOT / "data" / "public_set.jsonl").read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        public_targets = {row["ground_truth"]["parent_asin"] for row in public_rows}
        self.assertEqual(manifest["disclosed_candidate_target_pool"], 1406)
        for name, info in manifest["folds"].items():
            path = sets_dir / info["path"]
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            targets = {row["ground_truth"]["parent_asin"] for row in rows}
            self.assertEqual(len(rows), 800, name)
            self.assertEqual(len(targets), 800, name)
            self.assertFalse(targets & public_targets, name)
            for stratum in range(1, 11):
                subset = [row for row in rows if row["population_stratum"] == stratum]
                scenarios = [row["scenario_type"] for row in subset]
                self.assertEqual(len(subset), 80, (name, stratum))
                self.assertEqual(scenarios.count("buying"), 32)
                self.assertEqual(scenarios.count("browsing"), 32)
                self.assertEqual(scenarios.count("intent_override"), 12)
                self.assertEqual(scenarios.count("boundary"), 4)

    def test_optuna_v2_aggregate_matches_official_efficiency_formula(self) -> None:
        sys.path.insert(0, str(ROOT / "experiments" / "log"))
        module = importlib.import_module("55_optuna_official")
        public = {"sample_count": 200, "hit_rate_at_10": .995, "mrr": .995, "mttc": 2.32}
        private = {"sample_count": 800, "hit_rate_at_10": .9725,
                   "mrr": .97083375, "mttc": 2.975}
        result = module.aggregate(public, private)
        expected_mttc = (.2 * 2.32) + (.8 * 2.975)
        expected = (.5 * result["hit_rate_at_10"] + .3 * result["mrr"]
                    + .2 * ((11.0 - expected_mttc) / 10.0))
        self.assertAlmostEqual(result["technical_score"], expected)

    def test_disturbance_allocations_have_exact_total_variation(self) -> None:
        for tv in (.05, .10, .20):
            for direction in ("less_popular", "more_popular"):
                counts = counts_for(tv, direction)
                self.assertEqual(sum(counts), 800)
                empirical = .5 * sum(abs(count / 800 - .1) for count in counts)
                self.assertAlmostEqual(empirical, tv)
                if direction == "more_popular":
                    self.assertLess(counts[0], counts[-1])
                else:
                    self.assertGreater(counts[0], counts[-1])


if __name__ == "__main__":
    unittest.main()
