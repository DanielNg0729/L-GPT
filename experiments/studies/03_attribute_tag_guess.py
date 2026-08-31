"""V2.03: guarded semantic tag prediction that emits catalogue-attested canonical evidence."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.studies.provenance_gate import assess
from experiments.studies.semantic_grounding import MODEL_CACHE, MODEL_NAME
from submission.agent import Agent, CONSTRAINT


OUT = ROOT / "experiments" / "results" / "public_value_only_attribute_tag.json"
CANONICAL = {
    "cotton":"cotton","leather":"leather","polyester":"polyester","nylon":"nylon","rubber":"rubber","suede":"suede","canvas":"canvas","wool":"wool",
    "black":"black","white":"white","blue":"blue","red":"red","pink":"pink","green":"green","brown":"brown","beige":"beige","gray":"gray",
    "imported":"imported","buckle":"buckle closure","pull_on":"pull on closure","zipper":"zipper closure","lace":"lace up closure","button":"button closure",
    "water_resistant":"water resistant","waterproof":"waterproof","machine_washable":"machine washable","hand_wash":"hand wash","lightweight":"lightweight",
    "breathable":"breathable","slip_resistant":"slip resistant","adjustable":"adjustable","elastic":"elastic",
}
PROTOTYPES = {
    "cotton":"cotton natural plant textile material","leather":"leather animal hide material","polyester":"polyester synthetic textile material","nylon":"nylon synthetic polyamide material","rubber":"rubber elastic polymer material","suede":"suede brushed animal hide material","canvas":"canvas heavy woven cloth material","wool":"wool animal fleece fibre material",
    "black":"black dark colour","white":"white pale neutral colour","blue":"blue ocean sky colour","red":"red crimson colour","pink":"pink rosy colour","green":"green leaf colour","brown":"brown tan colour","beige":"beige sand colour","gray":"gray ash neutral colour",
    "imported":"imported made in another country","buckle":"buckle clasp closure fastener","pull_on":"pull on closure no fastener","zipper":"zipper toothed closure fastener","lace":"lace up tied cord closure","button":"button closure round fastening",
    "water_resistant":"water resistant repels moisture","waterproof":"waterproof prevents water penetration","machine_washable":"machine washable laundry care","hand_wash":"hand wash manual cleaning care","lightweight":"lightweight low weight","breathable":"breathable air circulation","slip_resistant":"slip resistant grip","adjustable":"adjustable resizable fit","elastic":"elastic stretch material",
}


class AttributeTagAgent(Agent):
    MIN_SIMILARITY = 0.45
    MIN_MARGIN = 0.03

    def __init__(self, catalog_path: Path) -> None:
        super().__init__(catalog_path)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE), device="cpu", local_files_only=True)
        self.names = list(PROTOTYPES)
        self.prototype_matrix = np.asarray(self.model.encode([PROTOTYPES[n] for n in self.names], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        self.gate = defaultdict(int)

    def _observe(self, st, msg: str) -> None:
        super()._observe(st, msg)
        for text, tier in self._extract_templated(msg):
            if tier != CONSTRAINT:
                continue
            status = assess(self, text)
            self.gate["seen"] += 1
            if not status.eligible:
                self.gate["blocked"] += 1
                continue
            self.gate["eligible"] += 1
            vector = np.asarray(self.model.encode([status.phrase], normalize_embeddings=True, show_progress_bar=False)[0], dtype=np.float32)
            scores = self.prototype_matrix @ vector
            order = scores.argsort()[::-1]
            best, second = int(order[0]), int(order[1])
            similarity, margin = float(scores[best]), float(scores[best] - scores[second])
            if similarity < self.MIN_SIMILARITY or margin < self.MIN_MARGIN:
                self.gate["low_confidence"] += 1
                continue
            phrase = CANONICAL[self.names[best]]
            for resolved in self._resolve(phrase):
                if resolved not in st.evidence:
                    df = self.ix.df(resolved)
                    if df > 0:
                        st.evidence[resolved] = (df, CONSTRAINT)
                        self.gate["accepted"] += 1


def compact(result: dict) -> dict:
    return {key: result[key] for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")}


def run(path: Path, ids, cats, products, catalog: Path) -> dict:
    agent = AttributeTagAgent(catalog)
    return {"result":compact(evaluate(agent,load_jsonl(path),ids,cats,products)),"gate":dict(agent.gate)}


def main() -> None:
    catalog=ROOT/'data/catalog.jsonl'; ids,cats,products=catalog_index(catalog); suite=ROOT/'experiments/datasets/public_value_only'
    out={"candidate":"attribute_tag_guess","canonical_replay":run(suite/'official200_canonical_replay.jsonl',ids,cats,products,catalog),"attribute_paraphrase":run(suite/'official200_attribute_paraphrase_dev.jsonl',ids,cats,products,catalog)}
    OUT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8'); print(json.dumps(out,indent=2))


if __name__=='__main__': main()
