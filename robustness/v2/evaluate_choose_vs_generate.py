"""V2.48: does showing the model retrieved candidates beat letting it answer unaided?

THE THREE SHAPES, AND WHY THE THIRD IS THE ONE THAT MATTERS
-----------------------------------------------------------
V2.46 measured `generate`: the model deparaphrases from parametric knowledge with no
catalogue context, and `df(proposal) > 0` accepts or discards. It reached ~96% of the
V2.43 oracle. `choose` -- retrieve k candidates, model picks one -- was described but
never built. This builds it, and a third shape that neither of the first two covers.

    generate   phrase ----------------------> LLM -> proposal -> df>0 -> accept
    choose     phrase -> retrieve k -> LLM picks one of the k, or NONE -> accept
    hybrid     phrase -> retrieve k -> LLM picks one, OR answers freely -> df>0 -> accept

`choose` is bounded above by recall@k: if the right canonical is not in the list, no
amount of model skill recovers it. That bound was badly mis-estimated until V2.47. Reading
V2.41's k<=10 numbers (dev200 0.0896-0.1493) suggested a ceiling near 0.15 and an arm not
worth building. Extending the curve to the full index showed it does not flatten -- the
correct canonical sits at MEDIAN RANK 56-96, so the ceiling at k=100 is near 0.76, and the
signal is diffuse rather than absent.

The retriever actually used here is weaker than that, and deliberately so: it is
e5-base-v2, selected on the train-only corpus, whose dev200 R@100 is 0.5373. So `choose`
runs against a real ceiling of roughly 0.54 on this suite, not 0.76 and not 0.15.

`hybrid` is the shape that should dominate both, and the argument is structural rather
than empirical: candidates are supplied as EVIDENCE the model may use or ignore, so the
information available to it is a strict superset of what `generate` had. If hybrid loses
to generate, that is not a retrieval failure -- it is the candidates ANCHORING the model
onto a plausible-looking wrong answer it would not otherwise have given. That is a real
and well-documented failure mode, and distinguishing it from a recall failure is the point
of running all three.

WHAT EACH OUTCOME WOULD MEAN
    hybrid > generate       retrieval adds real information; build the retrieval path
    hybrid ~ generate       candidates are inert; parametric knowledge already had it
    hybrid < generate       candidates ANCHOR the model onto wrong answers; retrieval is
                            actively harmful here, which is a stronger result than a null

GUARDS carried over unchanged from V2.46, and not negotiable
  * PROVENANCE. `df(proposal) > 0` before anything becomes evidence, in every arm.
  * NO EMITTABILITY FILTER. Restricting proposals to values `intent_card()` can emit would
    score better and would be gaming: the suite's canonicals are emittable by construction.
  * WEAK TIER. Proposals enter at the attenuated weight, not at CONSTRAINT weight. V2.46
    showed attenuation is worth ~15 points of oracle, because it caps what a wrong
    proposal costs.
  * k AND THE RETRIEVER ARE SELECTED ON TRAIN-ONLY DATA, never on dev200 and never on the
    end-to-end score. Recall@k rises monotonically, so an argmax over k has a trivial
    direction; k is fixed at 100 from the train-only corpus curve before any arm runs, and
    held constant across arms. Comparing arms at a k chosen to flatter one would be
    circular, and choosing either on dev200 would be selecting on the evaluation data.

THE SUITE REMAINS TOO SMALL TO SETTLE ANY OF THIS. 27 distinct paraphrases, ~25 resolver
calls per arm. One phrase moving changes the rate by 3.7 points. Nothing here may be
keyed, sized or tuned to those phrases, and a difference between arms smaller than a few
points is noise, not a ranking.

Run:  PYTHONIOENCODING=utf-8 python -u robustness/v2/evaluate_choose_vs_generate.py
"""
from __future__ import annotations

import importlib.util as ilu
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V2 = ROOT / "robustness" / "v2"
PVO = V2 / "public_value_only"
CACHE = ROOT / ".v2_model_cache"
OUT = V2 / "results" / "choose_vs_generate_v2_48.json"

SEM = "sem"
# RETRIEVER AND K ARE SELECTED ON THE TRAIN-ONLY SYNONYMY CORPUS, never on dev200.
# dev200 shares paraphrase vocabulary with the suite this script scores, so choosing
# either there would be selecting on the evaluation data. V2.47's train-only column
# picks e5-base-v2 (R@100 0.9396, the best of six) and k=100 (where that curve has
# nearly saturated: 0.8846 -> 0.9396 -> 0.9560 for k = 50 -> 100 -> 200).
#
# The protocol has a visible cost, which is the evidence that it is real: on dev200
# e5-base-v2 is the WORST of the six (R@100 0.5373 against MiniLM's 0.7612). Selecting
# on the evaluation surface would have handed `choose` a materially better retriever.
# It is not selected, and this arm runs handicapped.
K = int(os.environ.get("CHOOSE_K", "100"))
RETRIEVER = os.environ.get("CHOOSE_RETRIEVER", "intfloat/e5-base-v2")
# E5 is trained with asymmetric prefixes and scores materially worse without them.
QPRE = os.environ.get("CHOOSE_QPRE", "query: ")
DPRE = os.environ.get("CHOOSE_DPRE", "passage: ")

_s = ilu.spec_from_file_location("_v2_32", V2 / "evaluate_node4_cluster_aware.py")
_m = ilu.module_from_spec(_s)
_s.loader.exec_module(_m)
normalise = _m.normalise

BASE_RULES = (
    "Answer with the catalogue's own short attribute value and NOTHING else -- no "
    "sentence, no explanation, no quotes. If the description is a NEGATION or you are "
    "not confident which single value it names, answer exactly NONE. Answering NONE is "
    "correct and expected whenever you would otherwise guess."
)
SYS_GENERATE = (
    "You map a shopper's description of a product attribute onto the exact wording an "
    "e-commerce catalogue would use for it. Prefer the plainest, most common trade term: "
    "for 'made from a soft plant fibre' answer 'cotton'. " + BASE_RULES
)
SYS_CHOOSE = (
    "You map a shopper's description of a product attribute onto a catalogue value. You "
    "will be given the description and a numbered list of CANDIDATE values retrieved from "
    "the catalogue. Choose the ONE candidate that the description refers to, and copy it "
    "exactly. If NO candidate is right, answer exactly NONE -- do not pick the closest "
    "one. Answer with the candidate text and nothing else."
)
SYS_HYBRID = (
    "You map a shopper's description of a product attribute onto the exact wording an "
    "e-commerce catalogue would use for it. You will be given the description and a "
    "numbered list of CANDIDATE values retrieved from the catalogue. The candidates are "
    "HINTS, not a menu: they are often wrong or incomplete. If one of them is right, copy "
    "it exactly. Otherwise ignore them entirely and answer with the catalogue term you "
    "believe is correct. Prefer the plainest, most common trade term. " + BASE_RULES
)


class Resolver:
    """Paraphrase -> catalogue-attested phrase, or None, in one of three shapes."""

    def __init__(self, index, model, key, endpoint, mode, retrieve=None) -> None:
        self.ix, self.model, self.key, self.endpoint = index, model, key, endpoint
        self.mode, self.retrieve = mode, retrieve
        self.calls = self.accepted = self.abstained = self.unattested = self.failed = 0
        self.off_list = 0                      # hybrid: answered outside the candidates
        self.cache: dict[str, str] = {}

    def _ask(self, system: str, user: str) -> str | None:
        body = json.dumps({
            "model": self.model, "temperature": 0.0,
            # GPT-OSS bills hidden reasoning against max_tokens and returns EMPTY content
            # when the budget runs out mid-thought: HTTP 200 with nothing to parse.
            "max_tokens": 512, "reasoning_effort": "low",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode("utf-8")
        req = Request(self.endpoint, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
            # Groq's edge answers Python's default urllib identity with a Cloudflare 403.
            "User-Agent": "Mozilla/5.0"})
        for attempt in range(3):
            try:
                with urlopen(req, timeout=30) as r:
                    return json.loads(r.read().decode("utf-8"))[
                        "choices"][0]["message"]["content"].strip()
            except Exception:
                if attempt == 2:
                    return None
                time.sleep(2 * (attempt + 1))
        return None

    def resolve(self, phrase: str) -> str | None:
        if phrase in self.cache:
            return self.cache[phrase] or None
        self.calls += 1
        cands: list[str] = []
        if self.mode == "generate":
            raw = self._ask(SYS_GENERATE, phrase)
        else:
            cands = self.retrieve(phrase, K)
            listing = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cands))
            user = f"Description: {phrase}\n\nCandidates:\n{listing}"
            raw = self._ask(SYS_CHOOSE if self.mode == "choose" else SYS_HYBRID, user)

        if raw is None or not raw.strip():
            self.failed += 1                            # a failure is not an abstention
            return None
        prop = " ".join(raw.lower().split()).strip(".")
        if prop == "none":
            self.abstained += 1
            self.cache[phrase] = ""
            return None
        if self.mode == "choose" and prop not in {normalise(c) for c in cands}:
            # Strict choose may not leave the menu. An off-menu answer is a refusal to
            # comply, not a resolution, and counting it would silently turn this arm into
            # the hybrid one.
            self.abstained += 1
            self.cache[phrase] = ""
            return None
        if self.mode == "hybrid" and prop not in {normalise(c) for c in cands}:
            self.off_list += 1
        if self.ix.df(prop) <= 0:                       # PROVENANCE
            self.unattested += 1
            self.cache[phrase] = ""
            return None
        self.accepted += 1
        self.cache[phrase] = prop
        return prop


def main() -> None:
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["BERT_EXTRACT"] = "0"
    from submission.llm_rerank import ENDPOINT, _load_project_env
    _load_project_env()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY not set -- nothing to evaluate."); return
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import CONSTRAINT, Agent, raw_toks
    device = "cuda" if torch.cuda.is_available() else "cpu"

    canonicals = [json.loads(l)["canonical"]
                  for l in (V2 / "catalogue_attribute_dictionary.jsonl").open(encoding="utf-8")
                  if l.strip()]
    tok = AutoTokenizer.from_pretrained(RETRIEVER, cache_dir=str(CACHE))
    mdl = AutoModel.from_pretrained(RETRIEVER, cache_dir=str(CACHE)).to(device).eval()

    def embed(texts, prefix="", bs=256):
        outs = []
        for i in range(0, len(texts), bs):
            chunk = [prefix + t for t in texts[i:i + bs]]
            b = tok(chunk, padding=True, truncation=True, max_length=64,
                    return_tensors="pt").to(device)
            with torch.no_grad():
                h = mdl(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                outs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(outs).numpy()

    matrix = embed(canonicals, DPRE)
    print(f"retriever={RETRIEVER}  index={len(canonicals):,}  k={K}  model={model}\n")

    def retrieve(phrase: str, k: int) -> list[str]:
        sims = (embed([phrase], QPRE) @ matrix.T)[0]
        return [canonicals[int(i)] for i in np.argsort(-sims)[:k]]

    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    para = load_jsonl(PVO / "official200_attribute_paraphrase_dev.jsonl")

    def arm(res: Resolver, weight: float = 0.15):
        class Arm(Agent):
            def _observe(self, st, msg):
                super()._observe(st, msg)
                for text, tier in super()._extract_templated(msg):
                    if tier != CONSTRAINT:
                        continue
                    toks = raw_toks(text)[:self.RESOLVE_CAP]
                    if not toks or self.ix.df(" ".join(toks)) > 0:
                        continue                    # attested: suppression never fired
                    prop = res.resolve(" ".join(toks))
                    if prop and prop not in st.evidence:
                        st.evidence[prop] = (self.ix.df(prop), SEM)

            def _weight(self, phrase, df, tier):
                if tier == SEM:
                    return weight / (1.0 + df) ** self.IDF_POW
                return super()._weight(phrase, df, tier)
        return Arm

    def run(cls, samples):
        a = object.__new__(cls)
        a.ix, a.sessions = base.ix, {}
        a.llm = a.llm_extract = a.tagger = None
        return round(evaluate(a, samples, cid, cats, prods)[
            "recommended_technical_score"], 6)

    t0 = time.time()
    floor = run(Agent, para)
    results = {"0 suppression (floor, no LLM)": {"score": floor}}
    for mode in ("generate", "choose", "hybrid"):
        res = Resolver(base.ix, model, key, ENDPOINT, mode, retrieve)
        score = run(arm(res), para)
        results[mode] = {
            "score": score, "calls": res.calls, "accepted": res.accepted,
            "abstained": res.abstained, "unattested": res.unattested,
            "failed": res.failed, "off_list": res.off_list,
        }

    print(f"{'arm':<32}{'attr-para':>11}{'vs floor':>11}{'vs generate':>13}")
    print("-" * 67)
    g = results["generate"]["score"]
    print(f"{'0 suppression (floor)':<32}{floor:>11.6f}{0.0:>+11.6f}{floor-g:>+13.6f}")
    for mode in ("generate", "choose", "hybrid"):
        s = results[mode]["score"]
        print(f"{mode:<32}{s:>11.6f}{s-floor:>+11.6f}{s-g:>+13.6f}")

    print(f"\n{'arm':<12}{'calls':>7}{'accepted':>10}{'abstained':>11}"
          f"{'no-df':>7}{'failed':>8}{'off-list':>10}")
    print("-" * 65)
    for mode in ("generate", "choose", "hybrid"):
        r = results[mode]
        print(f"{mode:<12}{r['calls']:>7}{r['accepted']:>10}{r['abstained']:>11}"
              f"{r['unattested']:>7}{r['failed']:>8}{r['off_list']:>10}")

    print("\n  `choose` is bounded above by recall@k -- see V2.47. `hybrid` had a strict")
    print("  superset of generate's information, so hybrid < generate means the candidates")
    print("  ANCHORED the model onto answers it would not otherwise have given, which is a")
    print("  stronger finding than a null. `off-list` counts hybrid answers that ignored")
    print("  every candidate: a high count means retrieval contributed nothing.")
    print(f"\n  {time.time()-t0:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"experiment": "V2.48 choose vs generate vs hybrid", "model": model,
         "retriever": RETRIEVER, "k": K, "floor": floor,
         "selection_surface": "train-only synonymy corpus (never dev200)",
         "results": results},
        indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
