"""Analysis-only local MiniLM phrase-verification fallback.

The model is used only after the existing template extractor fails. It ranks phrases
already mined from the visible customer message and already attested in the frozen
catalogue; it cannot generate requirements or product IDs.  The base agent remains
unchanged.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "studies" / ".ml_deps"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402
from evaluator.local_evaluator import catalog_index, load_jsonl  # noqa: E402
from submission.agent import Agent, CONSTRAINT, PAT_NOINFO  # noqa: E402

stress = importlib.import_module("31_paraphrase_stress")
MODEL = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


class BertPhraseFallback(Agent):
    """Promote only the MiniLM-selected, locally mined phrase(s) to constraint tier."""
    TAKE = 1

    def _observe(self, st, msg: str) -> None:
        if PAT_NOINFO.search(msg):
            return
        # Preserve all normal behaviour if the simulator's constraint framing survives.
        if any(tier == CONSTRAINT for _, tier in self._extract_templated(msg)):
            return super()._observe(st, msg)
        mined = self.ix.mine(msg)
        if not mined:
            return super()._observe(st, msg)
        phrases = [phrase for phrase, _ in mined]
        try:
            values = list(MODEL.rerank("Shopper request: " + msg, phrases))
            chosen = sorted(range(len(phrases)), key=lambda i: values[i], reverse=True)[:self.TAKE]
        except Exception:
            return super()._observe(st, msg)
        # First retain the ordinary hybrid evidence, then safely raise the selected
        # phrase's tier.  This can only use text that occurred in the message/catalogue.
        super()._observe(st, msg)
        for i in chosen:
            phrase = phrases[i]
            df = self.ix.df(phrase)
            if df > 0:
                st.evidence[phrase] = (df, CONSTRAINT)


def shared(base: Agent, take: int) -> BertPhraseFallback:
    agent = object.__new__(BertPhraseFallback)
    agent.ix, agent.sessions, agent.llm, agent.TAKE = base.ix, {}, None, take
    return agent


def compact(result: dict) -> dict:
    return {key: result[key] for key in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}


def main() -> None:
    os.environ["LLM_RERANK"] = "0"
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    ids, cats, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    transforms = {"T0_exact": stress.TRANSFORMS["T0 identity (control)"],
                  "T1_surface": stress.TRANSFORMS["T1 scaffold reworded"],
                  "T5_realistic": stress.TRANSFORMS["T5 realistic (T1+T3)"]}
    output: dict = {}
    for take in (1, 2):
        for name, transform in transforms.items():
            result = stress.evaluate_transformed(shared(base, take), samples, ids, cats, products, transform)
            output[f"bert_top{take}|{name}"] = compact(result)
            print(f"bert_top{take:<2} {name:<12} {result['recommended_technical_score']:.6f}")
    (ROOT / "experiments" / "results" / "out_32_local_bert_phrase_verifier.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
