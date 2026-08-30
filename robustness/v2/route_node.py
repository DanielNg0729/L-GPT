"""V2 Node 1: strict-gated local dialogue-act routing.

This module is deliberately outside ``submission/``.  The submitted V1 agent does
not import this model or its dependencies.  V2 may select ``RouteOnlyV2Agent`` to
route unfamiliar wrappers, while released literal wrappers always take the inherited
deterministic V1 path.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

from submission.agent import Agent, recognised
from submission.agent import CAT, MINED, raw_toks
from evaluator.local_evaluator import coarse_category

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / ".v2_model_cache" / "shared_sixway_phrase_augmented_cuda"
LABELS = (
    "buying_opening", "constraint_update", "no_evidence",
    "override_opening", "override_update", "plain_opening",
)
OPENING = frozenset({"buying_opening", "plain_opening", "override_opening"})


class StrictGatedRouteNode:
    """Lazy six-route classifier with a deterministic turn-phase mask.

    ``classify`` returns ``None`` for a literal organizer message, a disabled node,
    or any inference failure.  ``None`` always means the caller uses V1 unchanged.
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir or os.environ.get("V2_ROUTE_MODEL_DIR", DEFAULT_MODEL))
        flag = os.environ.get("V2_ROUTE", "1").strip().lower()
        self.enabled = flag not in {"0", "false", "no", "off"}
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None
        self.model_loads = 0
        self.inferences = 0
        self.failures = 0
        self.actions: dict[str, int] = {}
        self.disabled_reason: str | None = None

    def _disable(self, reason: str) -> None:
        self.disabled_reason = self.disabled_reason or reason
        self._model = self._tokenizer = self._torch = None

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        if not self.enabled or self.disabled_reason is not None or not self.model_dir.is_dir():
            if not self.model_dir.is_dir():
                self._disable("model directory missing")
            return False
        try:
            import torch  # Deliberately lazy: never imported on a literal V1 path.
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir, local_files_only=True
            )
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
            self._torch = torch
            self.model_loads += 1
            return True
        except Exception as exc:
            self.failures += 1
            self._disable(f"load failed: {type(exc).__name__}")
            return False

    def classify(self, message: str, turn: int) -> str | None:
        """Classify only after exact literal recognition fails."""
        if recognised(message) or not self._ensure():
            return None
        try:
            encoded = self._tokenizer(
                [message], padding=True, truncation=True, max_length=80, return_tensors="pt"
            )
            # The tokenizer inherited from the scaffolding asset emits token type IDs,
            # while its DistilBERT sequence-classification head does not accept them.
            # Training used this same two-field interface.
            encoded = {key: value.to(self._device) for key, value in encoded.items()
                       if key in {"input_ids", "attention_mask"}}
            with self._torch.no_grad():
                logits = self._model(**encoded).logits[0]
            allowed = OPENING if turn == 1 else frozenset(LABELS).difference(OPENING)
            action = max(
                (label for label in LABELS if label in allowed),
                key=lambda label: float(logits[LABELS.index(label)]),
            )
            self.inferences += 1
            self.actions[action] = self.actions.get(action, 0) + 1
            return action
        except Exception:
            self.failures += 1
            return None

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "model_dir": str(self.model_dir),
            "device": str(self._device) if self._device is not None else None,
            "model_loads": self.model_loads,
            "inferences": self.inferences,
            "failures": self.failures,
            "actions": self.actions,
            "disabled_reason": self.disabled_reason,
        }


class RouteOnlyV2Agent(Agent):
    """V1 plus Node 1 routing only. Span and semantic-value handling stay in V1.

    The route node adds exactly two unknown-wrapper state transitions which cannot be
    recovered from a literal pattern: no-evidence avoids false mining, and an override
    clears prior rejection feedback before reranking. All evidence extraction remains
    the inherited parser, local tagger, and lexical miner.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        super().__init__(catalog_path)
        self.route_node = StrictGatedRouteNode()

    def _observe(self, state, message: str) -> None:
        action = self.route_node.classify(message, state.turn)
        if action == "no_evidence":
            # Preserve V1's label-free wrapper-shift telemetry even though this route
            # intentionally bypasses inherited mining.
            self._seen_messages = getattr(self, "_seen_messages", 0) + 1
            self._unrecognised = getattr(self, "_unrecognised", 0) + 1
            return
        if action == "override_update":
            state.rejected.clear()
        super()._observe(state, message)


class ExactCatalogueSpanNode:
    """Exact category and short-attribute recovery for unfamiliar wrappers only."""

    def __init__(self, catalog_path: str | Path) -> None:
        categories: set[tuple[str, ...]] = set()
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                phrase = tuple(raw_toks(coarse_category([str(x) for x in product.get("categories") or []])))
                if len(phrase) >= 2:
                    categories.add(phrase)
        dictionary_path = ROOT / "robustness" / "v2" / "catalogue_attribute_dictionary.jsonl"
        self.categories = sorted(categories, key=len, reverse=True)
        self.attributes = frozenset(
            json.loads(line)["canonical"] for line in dictionary_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def extract(self, text: str, action: str) -> tuple[str | None, set[str]]:
        tokens = raw_toks(text)
        category: str | None = None
        if action in {"buying_opening", "override_opening", "plain_opening"}:
            for pattern in self.categories:
                width = len(pattern)
                hit = next((start for start in range(len(tokens) - width + 1)
                            if tuple(tokens[start:start + width]) == pattern), None)
                if hit is not None:
                    category = " ".join(pattern)
                    tokens = tokens[:hit] + tokens[hit + width:]
                    break
        if action not in {"buying_opening", "override_opening", "constraint_update", "override_update"}:
            return category, set()
        attributes = {
            " ".join(tokens[start:start + width])
            for width in (1, 2, 3)
            for start in range(len(tokens) - width + 1)
            if " ".join(tokens[start:start + width]) in self.attributes
        }
        return category, attributes


class RouteAndSpanV2Agent(RouteOnlyV2Agent):
    """V2 Node 1 plus the accepted Node 2 lexical integration candidate.

    This class is unreachable for recognized organizer traffic.  On an unknown wrapper,
    the BERT is used only to retain product-content words; all added evidence is an exact
    string attested in the visible fixed catalogue.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        super().__init__(catalog_path)
        self.span_node = ExactCatalogueSpanNode(catalog_path)
        self.span_diagnostics = {"category": 0, "short_attribute": 0, "clean_mined": 0}

    def _add(self, state, text: str, tier: str) -> None:
        for phrase in self._resolve(text):
            if phrase and phrase not in state.evidence:
                df = self.ix.df(phrase)
                state.evidence[phrase] = (df if df > 0 else self.ix.DF_CAP * 2, tier)

    def _observe(self, state, message: str) -> None:
        action = self.route_node.classify(message, state.turn)
        if action is None:
            super(RouteOnlyV2Agent, self)._observe(state, message)
            return
        if action == "no_evidence":
            self._seen_messages = getattr(self, "_seen_messages", 0) + 1
            self._unrecognised = getattr(self, "_unrecognised", 0) + 1
            return
        if action == "override_update":
            state.rejected.clear()

        # Call the retained V1 content model once.  Its output is a lexical candidate
        # source, not an unverified model decision.
        tagger = getattr(self, "tagger", None)
        cleaned = None
        if tagger is not None and tagger.enabled:
            try:
                cleaned = tagger.strip(message)
            except Exception:
                cleaned = None
        candidate_text = cleaned or message

        # Preserve raw long-span V1 mining through the inherited method, but prevent a
        # second BERT invocation.  The cleaned long-span miner is unioned below.
        self.tagger = None
        try:
            super(RouteOnlyV2Agent, self)._observe(state, message)
        finally:
            self.tagger = tagger
        for phrase, _ in self.ix.mine(candidate_text):
            before = len(state.evidence)
            self._add(state, phrase, MINED)
            self.span_diagnostics["clean_mined"] += int(len(state.evidence) > before)

        category, attributes = self.span_node.extract(candidate_text, action)
        if category:
            self._add(state, category, CAT)
            for token in raw_toks(category):
                self._add(state, token, CAT)
            self.span_diagnostics["category"] += 1
        for attribute in attributes:
            before = len(state.evidence)
            self._add(state, attribute, MINED)
            self.span_diagnostics["short_attribute"] += int(len(state.evidence) > before)
