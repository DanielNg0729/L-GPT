"""Dense, catalogue-grounded product-passage retrieval for the isolated V2 experiments."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from robustness.v2.semantic_grounding import MODEL_CACHE, MODEL_NAME


CACHE_PATH = MODEL_CACHE / "product_passage_index.npz"


class ProductPassageRetriever:
    """Retrieve only IDs whose visible frozen product documents support the query."""

    def __init__(self, index) -> None:
        self.index = index
        self.calls = {"queries": 0, "load_failure": 0}
        self._ids: list[str] | None = None
        self._matrix: np.ndarray | None = None
        self._model = None

    def _load(self) -> bool:
        if self._model is not None and self._matrix is not None and self._ids is not None:
            return True
        try:
            os.environ.setdefault("HF_HOME", str(MODEL_CACHE / "huggingface"))
            from sentence_transformers import SentenceTransformer
            MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                MODEL_NAME, cache_folder=str(MODEL_CACHE), device="cpu", local_files_only=True
            )
            if CACHE_PATH.exists():
                data = np.load(CACHE_PATH, allow_pickle=False)
                self._ids = data["ids"].tolist()
                self._matrix = data["embeddings"].astype(np.float32, copy=False)
                return True
            self._ids = sorted(self.index.doc)
            passages = [self.index.doc[asin][:1200] for asin in self._ids]
            vectors = self._model.encode(passages, batch_size=128, show_progress_bar=False,
                                         normalize_embeddings=True)
            self._matrix = np.asarray(vectors, dtype=np.float32)
            np.savez_compressed(CACHE_PATH, ids=np.asarray(self._ids), embeddings=self._matrix)
            return True
        except Exception:
            self._ids = self._matrix = self._model = None
            self.calls["load_failure"] += 1
            return False

    def search(self, query: str, top_k: int = 60) -> dict[str, float]:
        self.calls["queries"] += 1
        if not self._load():
            return {}
        assert self._model is not None and self._matrix is not None and self._ids is not None
        try:
            vector = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = self._matrix @ np.asarray(vector, dtype=np.float32)
            chosen = np.argpartition(scores, -top_k)[-top_k:]
            chosen = chosen[np.argsort(scores[chosen])[::-1]]
            return {self._ids[int(i)]: float(scores[int(i)]) for i in chosen}
        except Exception:
            self.calls["load_failure"] += 1
            return {}
