"""Experiment 48: coarse global hyper-parameter search (Optuna), parked in a dashboard-readable DB.

SCOPE. Deliberately COARSE and exploratory. It is not the final tuning run: it exists to map
which knobs matter across ~20 parameters that have mostly never been swept together, and to
surface candidates for a later, stricter pass. Nothing here ships automatically -- adoption
still requires the pre-registered rule used all project: no regression on ANY of the seven
stress conditions, checked separately from this study.

THE OBJECTIVE IS NOT THE PUBLIC SCORE, AND THAT IS THE WHOLE POINT.
Optimising the public 200 is exactly what produced `IDF_POW = 0.35` -- a value that was
actively harmful on every other axis (uniform population -0.039, paraphrase -0.041) and
completely invisible on the set it was fitted to. So each trial is scored on three
conditions that stress different things:

    public-tune   100 real sessions (index-even half)      -- does it work at all
    synth-N       an UNSEEN minted draw, seed ROTATING per  -- does it generalise
                  trial so no single synthetic set can be
                  memorised the way the public 200 was
    para-T1       100 real sessions, scaffolding reworded   -- does it survive paraphrase

and the objective is their mean. Rotating the synthetic seed is the cheap analogue of
cross-validation: a configuration that wins only on one draw cannot accumulate a good mean.

PRUNING. Conditions are evaluated cheapest-first and reported as intermediate values, so
MedianPruner kills hopeless configurations after ~4 s instead of ~35 s.

STORAGE. SQLite at experiments/studies/optuna_coarse.db, so the study is inspectable live:

    optuna-dashboard sqlite:///experiments/studies/optuna_coarse.db

and resumable -- re-running this script continues the same study rather than starting over.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/scripts/48_optuna_coarse.py --hours 3 --workers 6
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

DB = ROOT / "experiments" / "studies" / "optuna_coarse.db"
STORAGE = f"sqlite:///{DB.as_posix()}"
STUDY = "track4_coarse_v1"

SYNTH_N = 300          # per-trial synthetic draw; small enough to keep trials ~30 s
PARA_N = 100

_G: dict = {}


def _boot():
    """Build the shared, read-only pieces once per worker process (~12 s)."""
    if _G:
        return _G
    os.environ["LLM_EXTRACT"] = "0"          # the search must be offline and deterministic
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import Agent
    p30 = __import__("30_robustness_benchmark")
    p31 = __import__("31_paraphrase_stress")
    samples = load_jsonl(ROOT / "data" / "public_set.jsonl")
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    _G.update(
        Agent=Agent, evaluate=evaluate, cid=cid, cats=cats, prods=prods, base=base,
        tune=[s for i, s in enumerate(samples) if i % 2 == 0],
        para_set=samples[:PARA_N],
        pub_targets={str(s["ground_truth"]["parent_asin"]) for s in samples},
        profiles=[s["user_profile"] for s in samples],
        mint=p30.mint, ev_t=p31.evaluate_transformed, TR=p31.TRANSFORMS,
    )
    return _G


def suggest(trial) -> dict:
    """~20 knobs. Ranges from docs/design/hyperparameters.md; wide, because this pass is a map."""
    return {
        # --- ranking weights
        "W_CATEGORY": trial.suggest_float("W_CATEGORY", 0.2, 1.4),
        "W_MINED": trial.suggest_float("W_MINED", 0.02, 0.8),
        "IDF_POW": trial.suggest_float("IDF_POW", 0.0, 0.5),
        "W_POP": trial.suggest_float("W_POP", 0.0, 0.6),
        "MINED_LEN_DIV": trial.suggest_float("MINED_LEN_DIV", 3.0, 16.0),
        # --- retrieval
        "POOL": trial.suggest_int("POOL", 150, 1200, step=50),
        "STRONG_DF": trial.suggest_int("STRONG_DF", 100, 1500, step=50),
        "DF_CAP": trial.suggest_int("DF_CAP", 2000, 30000, log=True),
        "STRONG_CAP": trial.suggest_int("STRONG_CAP", 3, 16),
        "OR_CAP": trial.suggest_int("OR_CAP", 6, 30),
        # --- mining: the highest-value untuned knobs, per the registry
        "MINE_MAXN": trial.suggest_int("MINE_MAXN", 5, 14),
        "MINE_MINN": trial.suggest_int("MINE_MINN", 2, 5),
        "RESOLVE_CAP": trial.suggest_int("RESOLVE_CAP", 6, 24),
        # --- BM25 column weights, never swept on unseen sessions before
        "BM_TITLE": trial.suggest_float("BM_TITLE", 0.0, 10.0),
        "BM_CATS": trial.suggest_float("BM_CATS", 0.0, 10.0),
        "BM_FEAT": trial.suggest_float("BM_FEAT", 0.0, 10.0),
        "BM_DETAILS": trial.suggest_float("BM_DETAILS", 0.0, 10.0),
        "BM_STORE": trial.suggest_float("BM_STORE", 0.0, 5.0),
        "BM_DESC": trial.suggest_float("BM_DESC", 0.0, 5.0),
    }


def build_agent(p: dict):
    g = _boot()
    Agent = g["Agent"]

    class Tuned(Agent):
        W_CATEGORY = p["W_CATEGORY"]
        W_MINED = p["W_MINED"]
        IDF_POW = p["IDF_POW"]
        W_POP = p["W_POP"]
        POOL = p["POOL"]
        STRONG_DF = p["STRONG_DF"]

        def _weight(self, phrase, df, tier):
            from submission.agent import CAT, CONSTRAINT, LLM, MINED
            base = {CONSTRAINT: self.W_CONSTRAINT, CAT: self.W_CATEGORY,
                    LLM: self.W_LLM, MINED: self.W_MINED}.get(tier, self.W_MINED)
            if tier == MINED:
                base *= min(1.0, len(phrase.split()) / p["MINED_LEN_DIV"])
            return base / (1.0 + df) ** self.IDF_POW

        def _resolve(self, text, cap=None):
            return super()._resolve(text, p["RESOLVE_CAP"])

        def _candidates(self, st, message):
            ev = sorted(st.evidence.items(), key=lambda kv: kv[1][0])
            strong = [q for q, (df, _) in ev if df <= self.STRONG_DF]
            pool, seen = [], set()

            def add(expr, limit):
                for asin in self.ix.search(expr, limit):
                    if asin not in seen:
                        seen.add(asin)
                        pool.append(asin)

            quoted = [f'"{q}"' for q in strong[:p["STRONG_CAP"]]]
            if quoted:
                add(" AND ".join(quoted), self.POOL)
                for k in range(len(quoted) - 1, 0, -1):
                    if len(pool) >= self.POOL:
                        break
                    add(" AND ".join(quoted[:k]), self.POOL)
                add(" OR ".join(quoted), self.POOL)
            if len(pool) < self.POOL and ev:
                add(" OR ".join(f'"{q}"' for q, _ in ev[:p["OR_CAP"]]), self.POOL)
            if not pool:
                from submission.agent import content_toks
                terms = list(dict.fromkeys(content_toks(message)))[:40]
                if terms:
                    add(" OR ".join(f'"{t}"' for t in terms), self.POOL)
            self._sample_population(st, pool)
            return pool

        def _observe(self, st, msg):
            ix = self.ix
            om, ocap, obm = ix.mine, ix.DF_CAP, ix.BM25
            ix.DF_CAP = p["DF_CAP"]
            ix.mine = lambda text, maxn=p["MINE_MAXN"], minn=p["MINE_MINN"]: \
                om(text, maxn=maxn, minn=minn)
            try:
                return super()._observe(st, msg)
            finally:
                ix.mine, ix.DF_CAP, ix.BM25 = om, ocap, obm

    o = object.__new__(Tuned)
    o.ix, o.sessions, o.llm, o.llm_extract = g["base"].ix, {}, None, None
    return o


def objective(trial):
    g = _boot()
    p = suggest(trial)
    bm = (f'bm25(p, 0.0, {p["BM_TITLE"]:.3f}, {p["BM_CATS"]:.3f}, {p["BM_FEAT"]:.3f}, '
          f'{p["BM_DETAILS"]:.3f}, {p["BM_STORE"]:.3f}, {p["BM_DESC"]:.3f})')
    saved_bm = g["base"].ix.BM25
    g["base"].ix.BM25 = bm
    try:
        scores = []

        # 1. cheapest first, so hopeless configs die in ~4 s
        r = g["evaluate"](build_agent(p), g["tune"], g["cid"], g["cats"], g["prods"])
        scores.append(r["recommended_technical_score"])
        trial.report(scores[0], 0)
        if trial.should_prune():
            import optuna
            raise optuna.TrialPruned()

        # 2. an UNSEEN draw, rotating so no single synthetic set can be memorised
        seed = 5000 + (trial.number % 25)
        synth = g["mint"](g["prods"], g["pub_targets"], g["profiles"], "reviews",
                          SYNTH_N, seed=seed)
        r = g["evaluate"](build_agent(p), synth, g["cid"], g["cats"], g["prods"])
        scores.append(r["recommended_technical_score"])
        trial.report(sum(scores) / len(scores), 1)
        if trial.should_prune():
            import optuna
            raise optuna.TrialPruned()

        # 3. paraphrase
        r = g["ev_t"](build_agent(p), g["para_set"], g["cid"], g["cats"], g["prods"],
                      g["TR"]["T1 scaffold reworded"])
        scores.append(r["recommended_technical_score"])

        trial.set_user_attr("tune", scores[0])
        trial.set_user_attr("synth", scores[1])
        trial.set_user_attr("para_T1", scores[2])
        trial.set_user_attr("synth_seed", seed)
        return sum(scores) / len(scores)
    finally:
        g["base"].ix.BM25 = saved_bm


def worker(deadline: float):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=STUDY, storage=STORAGE)
    _boot()
    while time.time() < deadline:
        study.optimize(objective, n_trials=1, catch=(Exception,))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=STUDY, storage=STORAGE, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=17, n_startup_trials=40),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=1),
    )
    # The shipped configuration, enqueued so the study always has a known-good reference
    # and TPE starts from somewhere sensible rather than pure random.
    if not study.trials:
        study.enqueue_trial({
            "W_CATEGORY": 0.75, "W_MINED": 0.15, "IDF_POW": 0.0, "W_POP": 0.25,
            "MINED_LEN_DIV": 8.0, "POOL": 400, "STRONG_DF": 500, "DF_CAP": 12000,
            "STRONG_CAP": 8, "OR_CAP": 14, "MINE_MAXN": 9, "MINE_MINN": 3,
            "RESOLVE_CAP": 12, "BM_TITLE": 6.0, "BM_CATS": 4.0, "BM_FEAT": 2.5,
            "BM_DETAILS": 2.5, "BM_STORE": 1.5, "BM_DESC": 1.0,
        })

    deadline = time.time() + args.hours * 3600
    print(f"study     : {STUDY}")
    print(f"storage   : {STORAGE}")
    print(f"dashboard : optuna-dashboard {STORAGE}")
    print(f"budget    : {args.hours} h across {args.workers} workers")
    print(f"existing  : {len(study.trials)} trials\n", flush=True)

    procs = [mp.Process(target=worker, args=(deadline,)) for _ in range(args.workers)]
    for pr in procs:
        pr.start()
    try:
        while any(pr.is_alive() for pr in procs):
            time.sleep(120)
            s = optuna.load_study(study_name=STUDY, storage=STORAGE)
            done = [t for t in s.trials if t.value is not None]
            if done:
                best = s.best_trial
                print(f"[{time.strftime('%H:%M:%S')}] {len(s.trials)} trials "
                      f"({len(done)} complete)  best {best.value:.5f}  "
                      f"tune {best.user_attrs.get('tune', 0):.4f} "
                      f"synth {best.user_attrs.get('synth', 0):.4f} "
                      f"para {best.user_attrs.get('para_T1', 0):.4f}", flush=True)
    finally:
        for pr in procs:
            pr.join(timeout=30)
            if pr.is_alive():
                pr.terminate()

    s = optuna.load_study(study_name=STUDY, storage=STORAGE)
    print(f"\nfinished: {len(s.trials)} trials, best {s.best_value:.5f}")
    print("best params:")
    for k, v in sorted(s.best_params.items()):
        print(f"  {k:<16}{v}")
    print("\nNOTHING IS ADOPTED FROM THIS. Candidates must clear the seven-condition")
    print("no-regression rule in a separate validation pass before they can ship.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
