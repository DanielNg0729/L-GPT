"""How large does the rescue's max_tokens actually need to be? Measured without calling the API.

WHY SIMULATE. The rescue ships `max_tokens=3072`, and the provider reserves
`prompt + max_tokens` against a tokens-per-DAY allowance rather than charging actual usage.
At 3072 that is roughly 3,145 reserved per call, which is about 63 calls per day against a
200,000 TPD limit -- and `openai/gpt-oss-20b` was measured at 199,739/200,000 used, blocking
other work for hours. So the number is worth pinning down. Calling the API to find it is
self-defeating when the goal is to stop spending tokens.

Everything except one quantity can be counted locally with the model's own tokenizer:

  prompt   the system prompt is a constant, and the user turn is the shopper transcript,
           which the suite determines exactly.
  output   the schema is a fixed Pydantic model and the values are the shopper's own
           phrases, so the emitted JSON can be reconstructed from the intent card.

WHAT CANNOT BE SIMULATED, AND WHY IT IS THE WHOLE PROBLEM. Reasoning tokens are internal to
the model. They are billed against the same budget as the output and are not a function of
anything visible here. The rescue's author already measured what happens when the budget is
too small -- at 512, two thirds of calls failed, either returning nothing because the entire
budget went to thinking, or emitting a JSON object truncated mid-write.

So this gives a FLOOR, not an answer: prompt + structured output is the part that must fit
no matter what, and whatever remains of the configured budget is the reasoning headroom.
It says how much of the 3072 is structural and how much is margin; it cannot say how much
margin the model needs. Narrowing that further requires calls, and the honest experiment is
a descending ladder on a handful of prompts, not a guess.

Run:  PYTHONIOENCODING=utf-8 python -u experiments/studies/simulate_rescue_token_budget.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LGPT = ROOT / ".review-l-gpt-shopping-copilot" / "copilot" / "llm_rescue.py"
SUITES = (
    ("Template200", ROOT / "data" / "public_set.jsonl"),
    ("Wrapper800", ROOT / "experiments" / "datasets" / "open_vocabulary"
     / "review800_canonical_replay.jsonl"),
)
CONFIGURED = 3072


def system_prompt() -> str:
    """Read the rescue's own constant rather than restating it."""
    src = LGPT.read_text(encoding="utf-8")
    start = src.index("SYSTEM_PROMPT = (")
    depth, i = 0, src.index("(", start)
    for j in range(i, len(src)):
        depth += (src[j] == "(") - (src[j] == ")")
        if depth == 0:
            break
    return " ".join(eval(src[i:j + 1]).split())          # noqa: S307 - our own file


def main() -> None:
    from transformers import AutoTokenizer
    from evaluator.local_evaluator import (catalog_index, load_jsonl,
                                           materialize_hidden_fields)

    # The public set ships its intent cards HIDDEN; the evaluator fills them from the
    # target product at run time. Reading them straight off disk measures empty cards and
    # reports a transcript of 31 tokens, which is not what the model would ever see.
    _ids, _cats, products = catalog_index(ROOT / "data" / "catalog.jsonl")

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    n = lambda s: len(tok.encode(s))                      # noqa: E731

    sysmsg = system_prompt()
    sys_n = n(sysmsg)
    # The schema is injected as a tool definition, so it is part of the prompt too.
    schema_json = json.dumps({
        "name": "RescuedIntent",
        "description": "Requirements recovered from the conversation.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "Product category the shopper named"},
            "requirements": {"type": "array", "items": {"type": "string"},
                             "description": "Exact phrases the shopper used to describe what they need"},
            "color": {"type": ["string", "null"], "description": "Colour, if stated"},
            "material": {"type": ["string", "null"], "description": "Material or fabric, if stated"},
            "price": {"type": ["number", "null"], "description": "Target price in dollars, if stated"},
            "exclude": {"type": "array", "items": {"type": "string"},
                        "description": "Things the shopper said they do not care about"}}}})
    schema_n = n(schema_json)

    print(f"system prompt        {sys_n:>5} tokens")
    print(f"tool schema          {schema_n:>5} tokens")
    print(f"fixed overhead       {sys_n + schema_n:>5} tokens\n")

    print(f"{'suite':<14}{'sessions':>9}{'prompt p50':>12}{'prompt p95':>12}"
          f"{'output p50':>12}{'output p95':>12}{'floor p95':>11}")
    print("-" * 82)
    rows = {}
    for name, path in SUITES:
        samples = load_jsonl(path)
        prompts, outputs = [], []
        for s in samples:
            card, _behavior = materialize_hidden_fields(s, products)
            hard = [str(v) for v in card.get("hard_constraints", [])]
            soft = [str(v) for v in card.get("soft_preferences", [])]
            values = hard + soft
            # The transcript the rescue sees: an opening plus the turns that disclose
            # constraints, two at a time, up to the turn-5 trigger.
            lines = ["Turn 1, shopper: Hi, I need a product."]
            for i in range(0, min(len(values), 8), 2):
                lines.append(f"Turn {len(lines) + 1}, shopper: For that, what matters is: "
                             + "; ".join(values[i:i + 2]) + ".")
            lines.append("We already asked about: other")
            lines.append("We have shown 40 products and none was right.")
            prompts.append(n("\n".join(lines)))
            # The structured object it must emit: the shopper's own phrases, verbatim.
            outputs.append(n(json.dumps({
                "category": "", "requirements": values,
                "color": None, "material": None, "price": None, "exclude": []})))
        pq = statistics.quantiles(prompts, n=20)[18] if len(prompts) > 1 else prompts[0]
        oq = statistics.quantiles(outputs, n=20)[18] if len(outputs) > 1 else outputs[0]
        rows[name] = {"sessions": len(samples),
                      "prompt_p50": statistics.median(prompts), "prompt_p95": pq,
                      "output_p50": statistics.median(outputs), "output_p95": oq,
                      "fixed_overhead": sys_n + schema_n}
        print(f"{name:<14}{len(samples):>9}{statistics.median(prompts):>12.0f}{pq:>12.0f}"
              f"{statistics.median(outputs):>12.0f}{oq:>12.0f}{oq:>11.0f}")

    worst = max(r["output_p95"] for r in rows.values())
    print(f"\n  Structured output needs at most ~{worst:.0f} tokens at p95.")
    print(f"  Configured budget is {CONFIGURED}, so ~{CONFIGURED - worst:.0f} tokens "
          f"({100 * (CONFIGURED - worst) / CONFIGURED:.0f}%) is reasoning headroom.")
    print(f"\n  That headroom is the part this cannot size. The rescue's author measured")
    print(f"  512 failing two thirds of the time -- the budget went to thinking and the")
    print(f"  object came back truncated or absent -- so the margin is doing real work.")
    print(f"  The remaining question is only how much, and a descending ladder over a")
    print(f"  handful of prompts answers it for the price of a handful of prompts.")

    out = ROOT / "experiments" / "results" / "out_75_rescue_token_budget.json"
    out.write_text(json.dumps({
        "experiment": "local token-budget simulation for the transcript rescue",
        "method": "gpt-oss-20b tokenizer; no provider calls",
        "cannot_simulate": "reasoning tokens, which are internal and billed against the same budget",
        "configured_max_tokens": CONFIGURED,
        "suites": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
