"""Optuna v2: official 200 + fixed stratified private-like 800.

This study fixes the defects in v1:
  * every trial sees identical data;
  * only COMPLETE trials are eligible for later validation;
  * paraphrasing is absent from the objective;
  * the private proxy has 800 distinct targets sampled from a disclosed-size pool; and
  * independently seeded population folds are held out rather than rotated by trial.

The objective is the exact 1,000-session aggregate of the official 200 and primary 800.
Nothing ships automatically. Top completed candidates must preserve the public score and
survive every held-out population fold.
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notes" / "eda"))
sys.path.insert(0, str(ROOT / "notes" / ".ml_deps"))

DB = ROOT / "notes" / "eda" / "optuna_official_v2.db"
STORAGE = f"sqlite:///{DB.as_posix()}"
STUDY = "track4_official_v2"
PRIMARY = ROOT / "robustness" / "optuna_v2_sets" / "primary_800.jsonl"

_P48 = importlib.import_module("48_optuna_coarse")
_G: dict = {}

SHIPPED = {
    "W_CATEGORY": 0.75, "W_MINED": 0.15, "IDF_POW": 0.0, "W_POP": 0.25,
    "MINED_LEN_DIV": 8.0, "POOL": 400, "STRONG_DF": 500, "DF_CAP": 12000,
    "STRONG_CAP": 8, "OR_CAP": 14, "MINE_MAXN": 9, "MINE_MINN": 3,
    "RESOLVE_CAP": 12, "BM_TITLE": 6.0, "BM_CATS": 4.0, "BM_FEAT": 2.5,
    "BM_DETAILS": 2.5, "BM_STORE": 1.5, "BM_DESC": 1.0,
}


def boot() -> dict:
    if _G:
        return _G
    os.environ["LLM_EXTRACT"] = "0"
    os.environ["LLM_RERANK"] = "0"
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    from submission.agent import Agent
    public = load_jsonl(ROOT / "data" / "public_set.jsonl")
    primary = load_jsonl(PRIMARY)
    cid, cats, prods = catalog_index(ROOT / "data" / "catalog.jsonl")
    base = Agent(ROOT / "data" / "catalog.jsonl")
    _G.update(Agent=Agent, evaluate=evaluate, cid=cid, cats=cats, prods=prods,
              base=base, public=public, primary=primary)
    _P48._G.clear()
    _P48._G.update(Agent=Agent, evaluate=evaluate, cid=cid, cats=cats, prods=prods,
                   base=base, tune=public, para_set=[], pub_targets=set(), profiles=[],
                   mint=None, ev_t=None, TR=None)
    return _G


def aggregate(public: dict, private: dict) -> dict:
    n_public, n_private = public["sample_count"], private["sample_count"]
    n = n_public + n_private
    hr = (n_public * public["hit_rate_at_10"] + n_private * private["hit_rate_at_10"]) / n
    mrr = (n_public * public["mrr"] + n_private * private["mrr"]) / n
    mttc = (n_public * public["mttc"] + n_private * private["mttc"]) / n
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {"sample_count": n, "hit_rate_at_10": hr, "mrr": mrr, "mttc": mttc,
            "efficiency": efficiency,
            "technical_score": 0.50 * hr + 0.30 * mrr + 0.20 * efficiency}


def build_agent(params: dict):
    agent = _P48.build_agent(params)
    agent.tagger = None
    return agent


def evaluate_params(params: dict) -> tuple[dict, dict, dict]:
    g = boot()
    saved = g["base"].ix.BM25
    g["base"].ix.BM25 = (
        f'bm25(p, 0.0, {params["BM_TITLE"]:.3f}, {params["BM_CATS"]:.3f}, '
        f'{params["BM_FEAT"]:.3f}, {params["BM_DETAILS"]:.3f}, '
        f'{params["BM_STORE"]:.3f}, {params["BM_DESC"]:.3f})'
    )
    try:
        public = g["evaluate"](build_agent(params), g["public"], g["cid"], g["cats"], g["prods"])
        private = g["evaluate"](build_agent(params), g["primary"], g["cid"], g["cats"], g["prods"])
        return public, private, aggregate(public, private)
    finally:
        g["base"].ix.BM25 = saved


def objective(trial):
    g = boot()
    params = _P48.suggest(trial)
    saved = g["base"].ix.BM25
    g["base"].ix.BM25 = (
        f'bm25(p, 0.0, {params["BM_TITLE"]:.3f}, {params["BM_CATS"]:.3f}, '
        f'{params["BM_FEAT"]:.3f}, {params["BM_DETAILS"]:.3f}, '
        f'{params["BM_STORE"]:.3f}, {params["BM_DESC"]:.3f})'
    )
    try:
        public = g["evaluate"](build_agent(params), g["public"], g["cid"], g["cats"], g["prods"])
        public_score = public["recommended_technical_score"]
        trial.set_user_attr("public_score", public_score)
        trial.report(public_score, 0)
        if trial.should_prune():
            import optuna
            raise optuna.TrialPruned()

        private = g["evaluate"](build_agent(params), g["primary"], g["cid"], g["cats"], g["prods"])
        combined = aggregate(public, private)
        trial.set_user_attr("private_proxy_score", private["recommended_technical_score"])
        trial.set_user_attr("combined_1000_score", combined["technical_score"])
        trial.set_user_attr("public_hr", public["hit_rate_at_10"])
        trial.set_user_attr("private_hr", private["hit_rate_at_10"])
        trial.report(combined["technical_score"], 1)
        return combined["technical_score"]
    finally:
        g["base"].ix.BM25 = saved


def worker(deadline: float) -> None:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=STUDY, storage=STORAGE)
    boot()
    study.optimize(objective, timeout=max(0.0, deadline - time.time()), catch=(Exception,))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--create-only", action="store_true")
    args = parser.parse_args()
    if not PRIMARY.exists():
        raise FileNotFoundError(f"build fixed folds first: {PRIMARY}")

    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=STUDY, storage=STORAGE, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=29, n_startup_trials=40, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=1),
    )
    if not study.trials:
        study.enqueue_trial(SHIPPED)
    print(f"study={STUDY}\nstorage={STORAGE}\nexisting={len(study.trials)}", flush=True)
    if args.create_only:
        return

    deadline = time.time() + args.hours * 3600
    workers = [mp.Process(target=worker, args=(deadline,)) for _ in range(args.workers)]
    for process in workers:
        process.start()
    try:
        while any(process.is_alive() for process in workers):
            time.sleep(30)
            current = optuna.load_study(study_name=STUDY, storage=STORAGE)
            complete = [t for t in current.trials if t.state == TrialState.COMPLETE]
            if complete:
                best = max(complete, key=lambda t: t.value)
                print(f"trials={len(current.trials)} complete={len(complete)} "
                      f"best=#{best.number} {best.value:.6f} "
                      f"public={best.user_attrs.get('public_score', 0):.6f} "
                      f"proxy={best.user_attrs.get('private_proxy_score', 0):.6f}", flush=True)
    finally:
        for process in workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()


if __name__ == "__main__":
    mp.freeze_support()
    main()
