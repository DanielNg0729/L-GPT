# Demonstrated Multi-Turn Session

This demonstration runs the submitted agent against released public sample `public_0002`
using the official simulator functions. It uses no organizer-private labels or state.

Run it from the repository root:

```bash
python -m submission.demo --sample-id public_0002
```

## Scenario

- Type: Intent Override
- Public target: `B071X54486`
- Outcome: hit at rank 1 on turn 3, immediately after the new intent became eligible
- External API calls: zero
- Reported tokens: zero

## Transcript

### Turn 1

Customer:

> I'm looking for Accessories Belts. Buckle closure

Agent:

> Here's my closest match so far. Is there a specific feature it needs to have?

- Asked attribute: `feature`
- Recommendation: `B08FLYBP2J`
- Result: no hit

### Turn 2

Customer:

> For that, what matters is: Imported; Buckle closure.

Agent:

> Here's my closest match so far. Do you have a material preference?

- Asked attribute: `material`
- Recommendation: `B071X54486`
- Result: not eligible to convert before the intent override

### Turn 3

Customer:

> Actually, ignore my earlier preference. What I need is: leather.

Agent:

> Here's my closest match so far. Anything else that matters to you?

- Asked attribute: `other`
- Recommendation: `B071X54486`
- Result: hit at rank 1

This session demonstrates state accumulation, structured clarification, override handling,
rejection-state reset, exact catalogue matching, sequential disclosure, and offline token
reporting through the same code used by the official evaluator.
