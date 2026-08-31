# L-GPT Shopping Copilot — demo UI

Not part of the submission. This directory is gitignored and nothing in the scored path
imports it. It exists so a person can talk to the agent in a browser.

## Run

Two terminals.

```bash
pip install -r .web/requirements.txt
python .web/server.py            # builds the catalogue index once, ~15s
```

```bash
cd .web && npm install && npm run dev
```

Then open http://localhost:5173. Vite proxies `/api` to the agent on port 8000.

## What you are looking at

Type in your own words. Almost nothing a person types matches the simulator's message
shapes, so every turn takes the **unfamiliar-wording** path: the dialogue-act router reads
what the turn means, exact span recovery pulls catalogue-attested values out of it, and the
tagger strips filler before mining.

Those layers record **0 inferences across the entire scored benchmark**, because nothing
there is unfamiliar. Here they run on every turn. The demo shows the half of the system the
score cannot reach — and retrieval is correspondingly harder than the benchmark suggests,
which is honest rather than a defect: off-template wording is the case the benchmark does
not contain.

The badge in the header reports which layers actually loaded, not which are enabled. Those
are different claims and only one of them is checkable.

## Configuration

All five configurable layers are on by default here, which is the showcase configuration,
not the scored one. The deterministic span node is core and does not count toward this
badge. The three hosted layers additionally need `GROQ_API_KEY` and go inert without it.

| variable | effect |
|---|---|
| `LLM_MESSAGE=0` | authored template phrasing instead of model-written |
| `LLM_RESOLVE=0` | no attribute deparaphrasing |
| `V2_ROUTE=0`, `BERT_EXTRACT=0` | disable the local classifiers |
| `MESSAGE_VARIETY=0` | one fixed sentence rather than the 1,344 assembled forms |
