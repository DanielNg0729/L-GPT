# TechJam Conversational Search Agent

## Deliverable

This package provides the required Python `Agent` for the frozen Amazon Clothing catalogue.
The evaluator entry point is `starter.agent.Agent`, which re-exports
`submission.agent.Agent`. The agent keeps a per-session evidence ledger, extracts only
catalogue-attested customer evidence, retrieves candidates from an in-memory SQLite index,
asks a useful next question, and returns ordered catalogue `parent_asin` values.

## Setup

Use Python 3.10 or later.

```bash
python -m pip install -r submission/requirements.txt
```

The organizer supplies the frozen catalogue and calls:

```python
from starter.agent import Agent

agent = Agent("data/catalog.jsonl")
agent.reset(session_id, user_profile)
response = agent.respond(session_id, user_message, turn, top_k)
```

`respond` returns the required `message`, `ask_attribute`, `recommendations`, and `usage`
fields. Recommendations contain only `parent_asin` values from the supplied catalogue.

## Method

The deterministic path parses the released message shapes, mines exact phrases from visible
customer text, and validates every retained phrase against the frozen catalogue. It ranks
with weighted phrase coverage, controlled candidate backoff, session rejection feedback,
and an information-oriented clarification policy.

For unfamiliar wording, the agent can use a local route classifier, a local token tagger,
and exact lookups in the included catalogue-derived attribute dictionary. If local model
weights or their dependencies are unavailable, these layers fail closed and the
deterministic lexical path remains valid.

Two optional Groq helpers are available only when `GROQ_API_KEY` is supplied: an
unattested-value resolver and a late whole-transcript recovery step. Both are catalogue
attested before they affect retrieval. They are not required for final scoring, and the
agent makes no external request without a key. Set `LLM_RESOLVE=0` or `LLM_RESCUE=0` to
disable either helper explicitly.

## Disclosure and limitations

- Required network access: none.
- Required API key: none.
- Offline public-development result: TechnicalScore 0.971500, HR@10 0.9950, MRR 0.9950,
  MTTC 2.225.
- Offline token usage: zero prompt tokens and zero completion tokens.
- Optional hosted model: Groq `openai/gpt-oss-20b`; usage and cost depend on the team's
  credential and are reported in the response `usage` fields when invoked.
- The deterministic pipeline relies on customer constraints being recoverable as exact,
  catalogue-attested wording. The optional recovery helpers are conservative fallbacks for
  unfamiliar wording, not a replacement for the frozen catalogue.
