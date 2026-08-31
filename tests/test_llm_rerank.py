from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from submission.llm_rerank import LLMReranker


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


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

    def test_api_usage_is_counted(self) -> None:
        payload = json.dumps({
            "choices": [{"message": {"content": "2, 1"}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 4},
        }).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"GROQ_API_KEY": "test", "LLM_RERANK": "1"}, clear=False
        ):
            reranker = LLMReranker(cache_path=Path(directory) / "cache.json")
            with patch("submission.llm_rerank.urlopen", return_value=_Response(payload)):
                self.assertEqual(reranker._call("prompt"), "2, 1")
            self.assertEqual(reranker.prompt_tokens, 17)
            self.assertEqual(reranker.completion_tokens, 4)


if __name__ == "__main__":
    unittest.main()
