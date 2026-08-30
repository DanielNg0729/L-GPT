"""V2.02: pessimistic dense product-passage RAG on Official200 value-only perturbations."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from robustness.v2.provenance_gate import assess
from robustness.v2.semantic_rag import ProductPassageRetriever
from submission.agent import Agent, CONSTRAINT


OUT = ROOT / "robustness" / "v2" / "results" / "public_value_only_dense_rag.json"


class DensePassageRAGAgent(Agent):
    """Candidate-union RAG with a pessimistic confidence-scaled ranking contribution."""

    RAG_WEIGHT = 0.15
    RAG_TOP_K = 60

    def __init__(self, catalog_path: Path) -> None:
        super().__init__(catalog_path)
        self.retriever = ProductPassageRetriever(self.ix)
        self.semantic_scores: dict[str, dict[str, float]] = defaultdict(dict)
        self.gate = defaultdict(int)

    def _observe(self, st, msg: str) -> None:
        super()._observe(st, msg)
        for text, tier in self._extract_templated(msg):
            if tier != CONSTRAINT:
                continue
            status = assess(self, text)
            self.gate["seen"] += 1
            if status.full_attested:
                self.gate["full_attested"] += 1
            if status.known_construction:
                self.gate["known_construction"] += 1
            if not status.eligible:
                self.gate["blocked"] += 1
                continue
            self.gate["eligible"] += 1
            attenuation = 1.0 - status.confidence
            for asin, similarity in self.retriever.search(status.phrase, self.RAG_TOP_K).items():
                signal = max(0.0, similarity) * attenuation
                self.semantic_scores[str(st.sid)][asin] = max(
                    self.semantic_scores[str(st.sid)].get(asin, 0.0), signal
                )

    def _candidates(self, st, message: str) -> list[str]:
        # Preserve the V1 lexical pool and its population observation. RAG only appends.
        pool = super()._candidates(st, message)
        seen = set(pool)
        extra = self.semantic_scores.get(str(st.sid), {})
        for asin, _ in sorted(extra.items(), key=lambda item: item[1], reverse=True):
            if asin not in seen:
                seen.add(asin)
                pool.append(asin)
        return pool

    def _rank(self, st, pool: list[str], top_k: int) -> list[str]:
        # The V1 score is copied exactly, then receives only the bounded semantic term.
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {asin: i for i, asin in enumerate(pool)}
        w_pop = self._w_pop_effective()
        semantic = self.semantic_scores.get(str(st.sid), {})

        def score(asin: str) -> tuple[float, int]:
            value = sum(weight for phrase, weight in wmap.items() if self.ix.covers(asin, phrase))
            if self.W_TITLE:
                value += self.W_TITLE * sum(
                    weight for phrase, weight in wmap.items() if self.ix.in_title(asin, phrase)
                )
            value += w_pop * (self.ix.pop.get(asin, 0.0) / (self.ix.max_pop or 1.0))
            if self.W_PROFILE and st.tags:
                blob = self.ix.blob.get(asin, "")
                value += self.W_PROFILE * sum(f" {tag} " in blob for tag in st.tags) / len(st.tags)
            value += self.RAG_WEIGHT * semantic.get(asin, 0.0)
            return (-value, order[asin])

        return sorted(pool, key=score)[:top_k]


def compact(result: dict) -> dict:
    return {key: result[key] for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")}


def run(dataset: Path, ids, categories, products, catalog: Path) -> dict:
    agent = DensePassageRAGAgent(catalog)
    result = evaluate(agent, load_jsonl(dataset), ids, categories, products)
    return {"result": compact(result), "gate": dict(agent.gate), "retriever": dict(agent.retriever.calls)}


def main() -> None:
    catalog = ROOT / "data" / "catalog.jsonl"
    ids, categories, products = catalog_index(catalog)
    suite = ROOT / "robustness" / "v2" / "public_value_only"
    output = {
        "candidate": "dense_product_passage_rag",
        "rag_weight": DensePassageRAGAgent.RAG_WEIGHT,
        "canonical_replay": run(suite / "official200_canonical_replay.jsonl", ids, categories, products, catalog),
        "attribute_paraphrase": run(suite / "official200_attribute_paraphrase_dev.jsonl", ids, categories, products, catalog),
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
