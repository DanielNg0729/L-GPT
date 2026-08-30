# The optional LLM rescue

The only part of this agent that uses a language model. It is **off by default** and the
scored path never depends on it.

Code: [`copilot/llm_rescue.py`](../copilot/llm_rescue.py).

---

## 1. What it does, and what it is not allowed to do

From turn 5, if nothing has converged, a model re-reads the shopper's own messages out of
the session graph and returns their requirements as structured JSON. Those go into the same
search as everything else.

That is its entire job. The boundaries are strict and structural:

| | |
|---|---|
| Sees | the conversation transcript, nothing else |
| Returns | plain text strings |
| **Never** | ranks products |
| **Never** | chooses the question |
| **Never** | produces a product id |

Because it is never shown the catalog, it **cannot invent a product that does not exist**.
Every product id in our answer still comes from the index. That is a property of the wiring,
not a guardrail we have to trust.

---

## 2. The handoff, traced end to end

**What the model is given** — the conversation only:

```text
Turn 1, shopper: Hi, I need Women Jeans. It has to have 73% Cotton, 25% Polyester, 2% Spandex.
Turn 2, shopper: Here's what counts - Imported and Zipper fly with button closure.

We already asked about: other
We have shown 30 products and none was right.
```

**What it returns** — strings, no ids:

```json
{"category": "Women Jeans",
 "requirements": ["Imported", "Zipper fly with button closure"],
 "material": "73% Cotton, 25% Polyester, 2% Spandex",
 "color": null, "price": null, "exclude": []}
```

**What happens next** — exactly the normal pipeline:

```text
phrase_docs("Zipper fly with button closure")  ->  16 products
phrase_docs("Imported")                        ->  15,300 products
intersection                                   ->  shortlist
pick_top_10                                    ->  product ids, from the catalog
```

So retrieval is unchanged. The model sits *upstream* of it and replaces one step: the one
that reads English.

---

## 3. Why only query understanding

Because that is the only stage where a model is better than the code it would replace.

- **Understanding the shopper** — this is the one stage that genuinely breaks if the shopper
  stops speaking in templates. A model is good at it.
- **Picking 10 products** — set intersection over 50,000 rows. A model cannot read them, and
  by the time we are at 10 candidates there is nothing left to decide.
- **Choosing the question** — one value from a 10-item list, decided by arithmetic over
  measured numbers ([ASK_POLICY.md](ASK_POLICY.md)). Not a language problem.

---

## 4. Failure is free

Any exception, timeout, refusal or unparseable answer returns nothing, and the turn proceeds
exactly as it would have. There is no retry storm and no error surfaced to the shopper.

This matters beyond tidiness: `submission_rules.md` warns that *"organizer policy may disable
network access"* during final scoring. An agent that needs a model API could score zero
through no fault of its own. Ours scores **0.892686** with the model switched off entirely,
which is the default.

---

## 5. Is it worth it? We measured the ceiling first

Before tuning a prompt, we ran an **oracle** rescue — one that cheats by returning the
shopper's true requirements. No real model can beat that, so it bounds what the idea is worth
at all.

| | no rescue | gpt-oss-20b | oracle ceiling |
|---|---|---|---|
| normal conversations | 0.8927 | 0.8927 | 0.8927 |
| sentences reworded | 0.8440 | **0.8623** | 0.8640 |
| heavily reworded, synonyms swapped | 0.8136 | 0.8140 | 0.8480 |

Three readings:

**On normal conversations the ceiling is zero.** It fires 3 times in 200 conversations and
changes nothing, because our parser already had everything. No prompt and no bigger model
would help — that is worth knowing before spending a day on prompt engineering.

**When wording changes, it captures nearly the whole ceiling** — +0.018 of an available
+0.020 — and restores Hit@10 from 0.985 back to 0.995.

**Under heavy synonym swapping it cannot help, and that is not a model failure.** The model
faithfully copies the shopper's (now corrupted) wording, which is what we asked for. The
original catalog phrasing is no longer present anywhere in the conversation. Only the oracle
recovers it, by reading the answer key. That 0.8480 is unreachable by anything that only
reads the chat.

---

## 6. The bug worth knowing about: reasoning models need headroom

The first full run gained almost nothing. The counter said why:
`rescue invoked: 24, returned usable data: 7`. Two thirds of calls were failing silently
inside the catch-all. Surfacing the exception gave two errors:

```
Tool choice is required, but model did not call a tool   (failed_generation: '')
Failed to parse tool call arguments as JSON              ({"category":"Shoes","color":null,"exclude":[]"
```

Both are the same root cause. **gpt-oss is a reasoning model**: it thinks before it answers,
and that thinking is billed against the same output budget as the structured response. At
`max_tokens=512` the reasoning consumed the budget, so either nothing came back or the JSON
was cut off mid-object. The successful smoke test had used 486 of 512 — right at the edge.

Fixed by raising to 3072 with `reasoning_effort="low"`, plus one retry:

| | before | after |
|---|---|---|
| hand-checked cases | 2 / 4 | **4 / 4** |
| full run, wording changed | 4 / 15 | **15 / 15** |
| full run, heavy rewrite | 7 / 24 | **24 / 24** |

The general lesson: when a reasoning model does structured output, budget for the reasoning
*and* the answer. And never let a broad `except` hide a systematic failure — a success
counter next to the call is what made it visible.

---

## 7. Configuration

```python
CopilotConfig(
    enable_llm_rescue = False,          # default
    llm_rescue_turn   = 5,
    llm_provider      = "groq",         # or "ollama" for fully local
    llm_model         = "openai/gpt-oss-20b",
    llm_max_tokens    = 3072,
)
```

Credentials come from the environment. Copy [`.env.example`](../.env.example) to `.env` and
add a key; `.env` is gitignored and **must never be committed**. For a fully offline setup,
`ollama pull gpt-oss:20b` and set `llm_provider="ollama"` — no key needed.

Cost across 200 conversations: 15 calls, 6,947 prompt + 2,829 completion tokens, about
**$0.002**.

One caveat on reproducibility: even at `temperature=0`, a hosted model is not bit-for-bit
deterministic. Runs vary in the fourth decimal. That is a further reason it stays off for the
headline number, which must be exactly reproducible.
