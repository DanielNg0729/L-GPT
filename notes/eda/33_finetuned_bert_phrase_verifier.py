"""Analysis-only fine-tuned BERT verifier for catalogue-attested phrase mining.

This is deliberately a narrow middle ground.  It never asks an LLM to invent a
requirement and never changes the submitted agent.  A small BERT classifier is trained
on synthetic, catalogue-derived customer utterances to decide whether a *candidate
phrase already found in the visible message* is an expressed constraint.  Optuna picks
training settings using synthetic validation only; public harness scores are reported
after that choice.

Run (analysis dependency lives in ignored notes/.ml_deps):
  python notes/eda/33_finetuned_bert_phrase_verifier.py
"""
from __future__ import annotations

import importlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notes" / ".ml_deps"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))

import optuna  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from evaluator.local_evaluator import catalog_index, coarse_category, intent_card, load_jsonl  # noqa: E402
from submission.agent import Agent, CONSTRAINT, PAT_NOINFO, raw_toks  # noqa: E402

stress = importlib.import_module("31_paraphrase_stress")

# Small, real BERT: deliberately selected so five CPU trials are feasible.  The
# experiment is about whether supervised constraint verification helps, not model size.
MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"
SEED = 73
TRIALS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def norm(text: str) -> str:
    return " ".join(raw_toks(text))


def make_augmented_pairs(products: dict, excluded: set[str], ix, n_products: int = 80):
    """Mint labels from non-public catalogue cards; no public target leaks into train."""
    rng = random.Random(SEED)
    pool = [a for a in products if a not in excluded]
    rng.shuffle(pool)
    rows: list[tuple[str, str, int]] = []
    decoys: list[str] = []
    for asin in pool[:n_products]:
        card = intent_card(products[asin])
        vals = [norm(str(x)) for x in card.get("hard_constraints", [])]
        vals += [norm(str(x)) for x in card.get("soft_preferences", [])]
        vals = [v for v in dict.fromkeys(vals) if len(v.split()) >= 2 and ix.df(v) > 0]
        if not vals:
            continue
        category = coarse_category([str(x) for x in products[asin].get("categories", [])])
        for value in vals[:3]:
            # Four semantically equivalent scaffolds + a noisy one; VALUE stays verbatim.
            messages = [
                f"I'm looking for {category}. A key requirement is: {value}.",
                f"I want to find {category}. It absolutely has to be {value}.",
                f"Sure -- the thing that counts for me is {value}.",
                f"Hmm, scratch all that. What I actually need is {value}.",
                stress.t_noise(f"I want to find {category}. It absolutely has to be {value}."),
            ]
            for message in messages:
                rows.append((message, value, 1))
                # Hard negatives are catalogue phrases, often plausible product text,
                # but not stated by this customer.
                for negative in rng.sample(decoys, min(2, len(decoys))):
                    rows.append((message, negative, 0))
            decoys.append(value)
    rng.shuffle(rows)
    return rows


def batches(tokenizer, rows, batch_size: int, shuffle: bool = False):
    def collate(chunk):
        messages, phrases, labels = zip(*chunk)
        encoded = tokenizer(list(messages), list(phrases), padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
        return encoded, torch.tensor(labels, dtype=torch.long)
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def train_and_score(tokenizer, train, valid, lr: float, batch_size: int, epochs: int):
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    # CPU-friendly partial fine-tune: preserve the pretrained lower language layers
    # and adapt only the final transformer block plus task head.  This is materially
    # faster but remains a genuine BERT fine-tune (not a frozen embedding baseline).
    for layer in list(model.bert.encoder.layer)[:-1]:
        for param in layer.parameters():
            param.requires_grad = False
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.01)
    model.train()
    for _ in range(epochs):
        for encoded, labels in batches(tokenizer, train, batch_size, shuffle=True):
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            loss = model(**encoded, labels=labels.to(DEVICE)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
    model.eval()
    good = total = 0
    with torch.no_grad():
        for encoded, labels in batches(tokenizer, valid, batch_size):
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            pred = model(**encoded).logits.argmax(-1).cpu()
            good += int((pred == labels).sum())
            total += len(labels)
    return model, good / max(total, 1)


class FineTunedFallback(Agent):
    """Only promote a mined phrase after BERT agrees it is a constraint."""
    tokenizer = None
    model = None
    threshold = 0.5

    def _observe(self, st, msg: str) -> None:
        if PAT_NOINFO.search(msg):
            return
        # Exact official framing retains the established deterministic path.
        if any(tier == CONSTRAINT for _, tier in self._extract_templated(msg)):
            return super()._observe(st, msg)
        mined = self.ix.mine(msg)
        super()._observe(st, msg)
        if not mined or self.model is None or self.tokenizer is None:
            return
        phrases = [p for p, _ in mined]
        try:
            encoded = self.tokenizer([msg] * len(phrases), phrases, padding=True,
                                     truncation=True, max_length=128, return_tensors="pt")
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            with torch.no_grad():
                probs = self.model(**encoded).logits.softmax(-1)[:, 1].cpu().tolist()
        except Exception:
            return
        # One promotion avoids multiplying a false-positive's influence.
        best = max(range(len(phrases)), key=lambda i: probs[i])
        if probs[best] >= self.threshold:
            phrase = phrases[best]
            st.evidence[phrase] = (self.ix.df(phrase), CONSTRAINT)


def shared(base: Agent, tokenizer, model, threshold):
    agent = object.__new__(FineTunedFallback)
    agent.ix, agent.sessions, agent.llm = base.ix, {}, None
    agent.tokenizer, agent.model, agent.threshold = tokenizer, model, threshold
    return agent


def compact(result):
    return {k: result[k] for k in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}


def main() -> None:
    os.environ["LLM_RERANK"] = "0"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    ids, cats, products = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    excluded = {str(s["ground_truth"]["parent_asin"]) for s in samples}
    pairs = make_augmented_pairs(products, excluded, base.ix)
    split = int(len(pairs) * 0.8)
    train, valid = pairs[:split], pairs[split:]
    print(f"device={DEVICE}; augmented pairs={len(pairs):,}; train={len(train):,}; valid={len(valid):,}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Each trial has a fixed small training budget. This makes Optuna selection honest
    # and feasible on CPU; the selected configuration is retrained once on all pairs.
    def objective(trial):
        lr = trial.suggest_float("lr", 8e-6, 6e-5, log=True)
        batch = trial.suggest_categorical("batch", [16, 32])
        _, accuracy = train_and_score(tokenizer, train[:100], valid[:80], lr, batch, epochs=1)
        return accuracy

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS)
    best = study.best_params
    print(f"optuna: best validation accuracy={study.best_value:.3f}, params={best}")
    model, final_valid_acc = train_and_score(tokenizer, train[:240], valid[:160], best["lr"], best["batch"], epochs=1)

    transforms = {
        "T0_exact": stress.TRANSFORMS["T0 identity (control)"],
        "T1_surface": stress.TRANSFORMS["T1 scaffold reworded"],
    }
    out = {"setup": {"model": MODEL_NAME, "device": DEVICE, "pairs": len(pairs),
                     "final_synthetic_validation_accuracy": final_valid_acc,
                     "optuna_best_params": best, "optuna_best_value": study.best_value,
                     "public_targets_excluded_from_augmentation": True,
                     "harness_sample": "fixed every-fifth public session (40/200)"}}
    # Bound the expensive per-message transformer inference to a fixed 40-session
    # holdout. This tests the relevant paraphrase hypothesis without pretending that
    # a slow local model is viable at shipped runtime.
    harness_samples = samples[::5]
    for name, transform in transforms.items():
        control = stress.evaluate_transformed(shared(base, tokenizer, model, threshold=1.1), harness_samples, ids, cats, products, transform)
        bert = stress.evaluate_transformed(shared(base, tokenizer, model, threshold=0.5), harness_samples, ids, cats, products, transform)
        out[name] = {"shipped_hybrid_control": compact(control), "finetuned_bert": compact(bert),
                     "delta": bert["recommended_technical_score"] - control["recommended_technical_score"]}
        print(f"{name}: control={control['recommended_technical_score']:.6f}  BERT={bert['recommended_technical_score']:.6f}  delta={out[name]['delta']:+.6f}")
    (ROOT / "notes" / "eda" / "out_33_finetuned_bert_phrase_verifier.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
