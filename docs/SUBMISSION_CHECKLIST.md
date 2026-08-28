# Submission Compliance Checklist

Audit date: 2026-08-28

This checklist maps the repository to `docs/submission_rules.md`,
`docs/competition_specification.md`, and the deliverables in
`docs/competition/track4_brief.md`.

## Code and interface

| Requirement | Status | Evidence |
|---|---|---|
| Export one Python `Agent` | Pass | `submission/agent.py` exports `Agent` |
| Implement `reset` and `respond` | Pass | Both methods match the required signatures |
| Return a string message | Pass | Contract audit across the public evaluator |
| Return an allowed attribute or null | Pass | Probe order is a subset of the allowed enum |
| Return ordered catalogue-valid recommendations | Pass | Official evaluator score and normalization audit |
| Report non-negative token counts | Pass | Default returns zero; optional components now report per-turn API usage |
| Avoid uncaught response exceptions | Pass | Safety envelope and failure-mode tests |

## Contents and security

| Requirement | Status | Evidence |
|---|---|---|
| No private evaluation data | Pass | Only participant-visible public and generated proxy sets are tracked |
| No organizer-only files | Pass | `organizer/` and `secure/` are excluded and absent from Git |
| No API keys or secrets | Pass | Credential scan is clean; `.env` is ignored |
| No privileged host access | Pass | Agent uses ordinary file reads, SQLite, and optional HTTPS |
| Do not modify evaluator files | Pass | Agent is imported by the unchanged official harness |
| No required external service | Pass | Official path is offline and external features default off |

## Reproducibility and reporting

| Requirement | Status | Evidence |
|---|---|---|
| Python requirement | Pass | Python 3.10 or newer is stated |
| Dependency installation | Pass | `python -m pip install -r submission/requirements.txt` |
| One official harness command | Pass | `python -m evaluator.local_evaluator` |
| Environment variables | Pass | `.env.example` and `submission/README.md` document all supported variables |
| Architecture, models, and limitations | Pass | Root and submission READMEs plus `docs/design/` |
| Latency, tokens, and cost | Pass | 17.93 seconds, zero tokens, and $0.00 for the audited official run |
| Demonstrated multi-turn session | Pass | `python -m submission.demo`; annotated transcript in `docs/DEMO.md` |
| Exact experiment record | Pass | `experiments/EXPERIMENT_INDEX.md` and versioned raw results |

## External release actions

These items cannot be satisfied by source changes alone:

| Requirement | Current status | Required action |
|---|---|---|
| Public GitHub repository | Action required | Change `DanielNg0729/L-GPT` from private to public before submission |
| Submission on the default branch | Action required | Merge or replace `main` with `codex/submission-ready` |
| Team member contributions | Action required if this is a team entry | Add each member and concrete contribution to the root README and Devpost |
| Public YouTube demo video | Action required | Record the official evaluation or `submission.demo`, upload publicly, and link it in Devpost |
| Devpost written description | Action required | Include tools, APIs, libraries, datasets, architecture, limitations, and the repository link |

## Packaging risk requiring organizer interpretation

The optional DistilBERT asset is approximately 265 MB and is stored through Git LFS. Local
models are permitted, but the submission rules describe allowed assets as lightweight.
The model is lazily loaded and is unreachable on official message forms, so it does not
affect the measured latency, score, token use, or API cost. If the organizer imposes a
strict bundle-size limit, submit the deterministic agent without this optional asset or
obtain confirmation that the LFS model is accepted.
