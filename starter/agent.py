"""
Submission entry point.

The organizer's harness loads the agent from here -- `evaluator/local_evaluator.py` does
`from starter.agent import Agent` -- so this module is the required surface. The
implementation lives in `submission/`, and this file re-exports it rather than duplicating
it.

WHY A RE-EXPORT AND NOT A COPY. It used to be a byte copy kept in step with `cp`, alongside
copies of the optional component modules. An audit found the copies had drifted: the
`starter/` copy of the scaffolding tagger still carried an older default, and four of the
five copied modules were never imported by anything, because this file imports its
components from `submission.*` regardless. A single definition removes that whole class of
bug -- there is now exactly one `Agent` in the repository and no way for the shipped and
scored versions to disagree.

Everything the agent needs is under `submission/`:

    agent.py                             the deterministic pipeline
    span_node.py                         exact catalogue span recovery
    route_node.py                        the dialogue-act router (Node 1)
    bert_extract.py                      scaffolding tagger
    llm_resolve.py                       deparaphraser (unattested value -> attested)
    llm_rescue.py                        whole-transcript recovery on a stall
    llm_message.py                       optional phrasing writer (off by default)
    llm_extract.py, llm_rerank.py        optional layers, both measured negative and off
    catalogue_attribute_dictionary.jsonl frozen attribute vocabulary
    models/                              local checkpoints

`tests/test_submission_contract.py` asserts that the object exported here is the object
defined there, so the delegation cannot silently break.
"""
from __future__ import annotations

from submission.agent import (  # noqa: F401  -- re-exported as the harness's entry point
    CAT,
    CONSTRAINT,
    LLM,
    MINED,
    SEM,
    Agent,
    SessionState,
    raw_toks,
    recognised,
)

__all__ = ["Agent", "SessionState", "raw_toks", "recognised",
           "CAT", "CONSTRAINT", "MINED", "LLM", "SEM"]
