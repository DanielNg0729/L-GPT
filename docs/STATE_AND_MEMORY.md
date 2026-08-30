# State and memory: the two graphs

The design calls for two graphs. They are different objects with different rules, and
mixing them up is the easiest way to get this wrong.

Code: [`copilot/session_graph.py`](../copilot/session_graph.py),
[`copilot/knowledge_graph.py`](../copilot/knowledge_graph.py),
[`copilot/graph.py`](../copilot/graph.py).

---

## 1. Two graphs, two clocks

| | Product index (knowledge graph) | Session graph |
|---|---|---|
| Covers | every product, every conversation | one conversation |
| Written | once, at startup | every single turn |
| Read | constantly | by the ranker |
| Lifetime | the whole run | dies with the conversation |
| Size | ~100 MB | a few kilobytes |

Both are plain JSON structures. **There is no graph database anywhere in this project** —
"graph" here means nodes and edges held in dictionaries.

The separation is what makes one index safe to share across 200 conversations. Nothing a
shopper says is ever written back to the product index, so no conversation can affect
another.

---

## 2. What the session graph records

```json
{
  "session_id": "public_a1b2",
  "turns": [
    {"turn": 1, "user_message": "...", "opening_type": "buying",
     "constraints_added": ["cotton"]}
  ],
  "product_nodes": {
    "B08P4SSFX4": {"shown_at": [{"turn": 1, "rank": 3, "scored": true}],
                   "best_rank": 3, "provably_wrong": true}
  },
  "attribute_nodes": {"material:cotton": {"source": "user", "turn": 1}},
  "edges": [{"type": "shown_at_turn", "to": "B08P4SSFX4", "turn": 1, "rank": 3}],
  "exhausted_attributes": ["material"],
  "asked": ["other", "other"],
  "hit_blocked_until": 1
}
```

Four things it is used for:

- **Not repeating ourselves.** Products already shown, and now known wrong, get pushed down
  the next list.
- **Not asking a dead question.** Once the shopper says they have nothing more on
  *material*, we never ask again.
- **Giving the optional LLM rescue something to read.** The `turns` list holds every message
  verbatim, which is exactly what it needs.
- **Debugging.** Set `session_graph_dir` and every conversation is written out as readable
  JSON you can replay.

### Marked, never deleted

A rejected product gets an edge, not a removal. Delete it and the next search simply finds
it again — the search has no memory of its own. Marking is what makes the exclusion stick,
and it also keeps a record of *why*, which a deletion throws away.

### The subtle rule: `hit_blocked_until`

> A product shown on an earlier turn is **not automatically known to be wrong.**

In a change-of-mind conversation the evaluator refuses to record a win until the new intent
arrives, on turn 3 or 4. A product shown on turn 1 of such a conversation may still be the
right answer.

So `record_shown()` takes a flag saying whether the evaluator could have scored that list,
and only lists that were actually scored mark their products wrong. Getting this wrong cost
us real points twice — see [RANKING.md §2](RANKING.md#2-the-40-penalty-and-the-trap-inside-it).

---

## 3. Conversation memory is LangGraph's, not ours

The graph is compiled with a checkpointer and every turn runs under
`thread_id = session_id`:

```python
graph = builder.compile(checkpointer=InMemorySaver())
state = graph.invoke({...}, config={"configurable": {"thread_id": session_id}})
```

So the `Agent` object holds **no per-conversation state at all** beyond the shopper profile.
Everything that survives between turns — the shopper's intent, the session graph — lives in
the checkpoint. That means a conversation can be inspected mid-flight, and replayed.

### What is deliberately kept out of the saved state

**The product index.** It is global, read-only, and identical for every conversation.
Pushing 50,000 product nodes through a checkpointer on every turn would be absurd. It is
bound into the node functions instead.

**The per-turn numpy arrays.** The candidate list and coverage vector travel from the search
node to the ranking node through a scratch dictionary keyed by conversation id. They are
large, they are not JSON, and they are meaningless once the turn ends.

What is left in the saved state is small, JSON-clean and replayable. That is the property
worth protecting.

---

## 4. The shopper's intent

The one object every stage reads. Built once from the opening line, then **updated** every
later turn — never rebuilt.

```json
{
  "opening_type": "buying",
  "category_raw": "Women Jeans",
  "category_terms": ["women", "jeans"],
  "constraints": [
    {"text": "cotton", "attribute": "material", "weight": 1.0,
     "turn": 1, "superseded": false}
  ],
  "facets": {"material": ["cotton"], "color": [], "price": null},
  "exhausted": [],
  "nothing_left_to_learn": false,
  "changed_mind": false
}
```

Nothing downstream ever reads the raw message text. That single discipline is what lets the
whole pipeline stay testable: to reproduce any turn, you only need this object.

### Change of mind: down-weight, never delete

When the shopper says *"actually, ignore my earlier preference"*, we do **not** clear what
they told us before. Two reasons:

- *"Ignore my earlier preference"* does not say **which** one. Deleting everything throws
  away good evidence on a guess.
- On the public set, the "new" intent is usually something they already said. Treating a
  restatement as a reversal once dropped a product from rank 1 to outside the top 10
  entirely.

So we distinguish the two cases. If the new value is something we already hold, it is a
**re-affirmation**: boost it, change nothing else. If it is genuinely new, it is a real
pivot: the new requirement leads, and the earlier ones drop to weight 0.35 — still there,
still able to help, no longer in charge.

The result: change-of-mind conversations score **Hit@10 1.000, MRR 0.894** — the highest MRR
of any conversation type.
