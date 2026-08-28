"""Guarded semantic phrase grounding for the isolated V2 program.

This module never changes the V1 submission route.  Its single responsibility is to map an
unfamiliar customer phrase to a *visible, catalogue-attested* feature phrase.  It has no
authority to classify dialogue intent, remove evidence, or rank products.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from submission.agent import raw_toks


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = ROOT / ".v2_model_cache"
CACHE_PATH = MODEL_CACHE / "feature_phrase_index.npz"
MIN_SIMILARITY = 0.52
MAX_FEATURE_WORDS = 14


def normalise(text: str) -> str:
    return " ".join(raw_toks(text))


def visible_feature_phrases(catalog_path: Path) -> list[str]:
    """Return unique, normalised feature strings from the visible frozen catalogue."""
    phrases: set[str] = set()
    with catalog_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for feature in row.get("features") or []:
                if not isinstance(feature, str):
                    continue
                phrase = normalise(feature)
                if 1 <= len(phrase.split()) <= MAX_FEATURE_WORDS:
                    phrases.add(phrase)
    return sorted(phrases)


class SemanticFeatureGrounder:
    """Nearest-feature resolver with explicit literal and confidence gates.

    `literal_df` is supplied by the caller's existing FTS index.  If it says the complete
    phrase exists, semantic computation is skipped.  The semantic result is returned only
    when its cosine score clears a fixed threshold and is itself a visible feature phrase.
    """

    def __init__(self, catalog_path: Path, literal_df: Callable[[str], int],
                 threshold: float = MIN_SIMILARITY) -> None:
        self.catalog_path = Path(catalog_path)
        self.literal_df = literal_df
        self.threshold = threshold
        self.calls = {
            "literal_present": 0,
            "literal_absent": 0,
            "semantic_accepted": 0,
            "below_threshold": 0,
            "load_failure": 0,
        }
        self._phrases: list[str] | None = None
        self._matrix: np.ndarray | None = None
        self._model = None

    def _load(self) -> bool:
        if self._matrix is not None and self._phrases is not None and self._model is not None:
            return True
        try:
            os.environ.setdefault("HF_HOME", str(MODEL_CACHE / "huggingface"))
            from sentence_transformers import SentenceTransformer

            MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE), device="cpu")
            if CACHE_PATH.exists():
                cached = np.load(CACHE_PATH, allow_pickle=False)
                self._phrases = cached["phrases"].tolist()
                self._matrix = cached["embeddings"].astype(np.float32, copy=False)
                return True
            self._phrases = visible_feature_phrases(self.catalog_path)
            vectors = self._model.encode(
                self._phrases, batch_size=128, show_progress_bar=False,
                normalize_embeddings=True,
            )
            self._matrix = np.asarray(vectors, dtype=np.float32)
            np.savez_compressed(CACHE_PATH, phrases=np.asarray(self._phrases), embeddings=self._matrix)
            return True
        except Exception:
            self._phrases = self._matrix = self._model = None
            self.calls["load_failure"] += 1
            return False

    def resolve(self, text: str) -> str | None:
        phrase = normalise(text)
        if not phrase:
            return None
        if self.literal_df(phrase) > 0:
            self.calls["literal_present"] += 1
            return None
        self.calls["literal_absent"] += 1
        if not self._load():
            return None
        assert self._model is not None and self._matrix is not None and self._phrases is not None
        try:
            query = self._model.encode([phrase], normalize_embeddings=True, show_progress_bar=False)
            scores = self._matrix @ np.asarray(query[0], dtype=np.float32)
            index = int(np.argmax(scores))
            if float(scores[index]) < self.threshold:
                self.calls["below_threshold"] += 1
                return None
            self.calls["semantic_accepted"] += 1
            return str(self._phrases[index])
        except Exception:
            self.calls["load_failure"] += 1
            return None
