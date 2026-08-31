"""V2.04: test whether dense RAG adds value after guarded attribute tagging."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.studies.provenance_gate import assess
from experiments.studies.semantic_rag import ProductPassageRetriever

# Numeric experiment files are loaded explicitly so their classes remain importable by script.
import importlib
TagBase = importlib.import_module("03_attribute_tag_guess").AttributeTagAgent

OUT = ROOT / "experiments" / "results" / "public_value_only_attribute_tag_plus_rag.json"


class TagPlusRAGAgent(TagBase):
    RAG_WEIGHT = 0.15

    def __init__(self, catalog_path: Path) -> None:
        super().__init__(catalog_path)
        self.retriever = ProductPassageRetriever(self.ix)
        self.rag_scores: dict[str, dict[str, float]] = defaultdict(dict)
        self.rag_gate = defaultdict(int)

    def _observe(self, st, msg: str) -> None:
        super()._observe(st, msg)
        for text, tier in self._extract_templated(msg):
            if tier != "con":
                continue
            status = assess(self, text)
            if not status.eligible:
                self.rag_gate["blocked"] += 1
                continue
            self.rag_gate["eligible"] += 1
            for asin, similarity in self.retriever.search(status.phrase, 60).items():
                signal = max(0.0, similarity) * (1.0 - status.confidence)
                self.rag_scores[str(st.sid)][asin] = max(self.rag_scores[str(st.sid)].get(asin, 0.0), signal)

    def _candidates(self, st, message: str) -> list[str]:
        pool = super()._candidates(st, message)
        seen = set(pool)
        for asin, _ in sorted(self.rag_scores.get(str(st.sid), {}).items(), key=lambda item: item[1], reverse=True):
            if asin not in seen:
                seen.add(asin)
                pool.append(asin)
        return pool

    def _rank(self, st, pool: list[str], top_k: int) -> list[str]:
        wmap = {p: self._weight(p, df, tier) for p, (df, tier) in st.evidence.items()}
        order = {asin: i for i, asin in enumerate(pool)}; w_pop = self._w_pop_effective(); rag = self.rag_scores.get(str(st.sid), {})
        def score(asin: str):
            value = sum(weight for phrase, weight in wmap.items() if self.ix.covers(asin, phrase))
            value += w_pop * (self.ix.pop.get(asin, 0.0) / (self.ix.max_pop or 1.0))
            value += self.RAG_WEIGHT * rag.get(asin, 0.0)
            return (-value, order[asin])
        return sorted(pool, key=score)[:top_k]


def compact(result: dict) -> dict:
    return {key: result[key] for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")}


def run(path, ids, cats, products, catalog):
    agent=TagPlusRAGAgent(catalog)
    return {"result":compact(evaluate(agent,load_jsonl(path),ids,cats,products)),"tag_gate":dict(agent.gate),"rag_gate":dict(agent.rag_gate),"retriever":dict(agent.retriever.calls)}


def main():
    catalog=ROOT/'data/catalog.jsonl'; ids,cats,products=catalog_index(catalog); suite=ROOT/'experiments/datasets/public_value_only'
    result={"candidate":"attribute_tag_plus_rag","canonical_replay":run(suite/'official200_canonical_replay.jsonl',ids,cats,products,catalog),"attribute_paraphrase":run(suite/'official200_attribute_paraphrase_dev.jsonl',ids,cats,products,catalog)}
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
