# Mining vs LLM Extraction vs Shipped Hybrid

All conditions keep the official evaluator and hidden target unchanged. Only the text passed to the agent is transformed.

## Matched pilot  -  20 sessions

| Condition | Mining only | LLM extraction only | Shipped hybrid |
|---|---:|---:|---:|
| Exact official text | 0.9640 | 0.9730 | **0.9750** |
| Surface rewrite | 0.9630 | **0.9730** | 0.9640 |
| Surface rewrite + reordered clauses | 0.9630 | **0.9650** | 0.9630 |

## Full harness evidence  -  200 sessions

| Condition | Mining only | LLM extraction only | Shipped hybrid |
|---|---:|---:|---:|
| Exact official text | 0.81405 | 0.95200 | **0.96755** |
| Surface rewrite |  -  |  -  | **0.84520** |
| Surface rewrite + reordered clauses |  -  |  -  | **0.84360** |

The full LLM/mining paraphrase cells have not been run, so they are intentionally blank rather than extrapolated from the pilot. The LLM arm is analysis-only and its API/cache cost is not part of the submitted path.

## Condition boundaries

`surface` changes template wording and adds discourse framing, while retaining literal requirement strings. `reordered` additionally changes clause order and punctuation, again retaining literal requirement strings. These are structure and wording robustness probes, **not** a semantic-substitution benchmark.

Source results: `experiments/results/out_21_llm_extraction.json`, `experiments/results/out_21_hybrid_paraphrase.json`, and `experiments/results/out_21_pilot_3way.json`.
