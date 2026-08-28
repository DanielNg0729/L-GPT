from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from submission.llm_rerank import LLMReranker


class LLMRerankerTest(unittest.TestCase):
    def test_valid_response_is_a_strict_permutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"GROQ_API_KEY": "test", "LLM_RERANK": "1"}, clear=False
        ):
            reranker = LLMReranker(cache_path=Path(directory) / "cache.json")
            with patch.object(reranker, "_call", return_value="2, 1, 3"):
                self.assertEqual(
                    reranker.rerank(["waterproof"], ["A", "B", "C"], ["a", "b", "c"]),
                    ["B", "A", "C"],
                )

    def test_invalid_response_keeps_the_caller_in_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"GROQ_API_KEY": "test", "LLM_RERANK": "1"}, clear=False
        ):
            reranker = LLMReranker(cache_path=Path(directory) / "cache.json")
            with patch.object(reranker, "_call", return_value="1, 3"):
                self.assertIsNone(reranker.rerank(["waterproof"], ["A", "B", "C"], ["a", "b", "c"]))


if __name__ == "__main__":
    unittest.main()
