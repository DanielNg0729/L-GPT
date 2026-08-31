# Track 4 — Shopping Copilot: AI Conversational Search and Recommendations

> TikTok TechJam 2026 · Technical Workshop Webinar: **August 28, 2026, 4:00–4:45pm SGT**
> Summarized from the Early Bird Access edition — Tracks & Problem Statements.

---

## 1. Context & problem

Traditional e-commerce search leans heavily on **static keyword matching**, so it fails to capture:

- The buyer's mindset, which shifts continuously over the course of a conversation
- The difference between **browsing** (casual, needs not yet defined) and **buying** (high purchase intent, hard constraints)

The task: build a **conversational shopping agent** over an Amazon catalog that understands context deeply,
adapts its workflow at runtime, and is commercially effective — measured against real purchase records in the dataset.

---

## 2. The four required pillars

### I. Core Architecture — Intent Routing & Hybrid Pipeline

**Dual-Track Routing** — detect intent up front, then branch:

| Track | When | How it's handled |
|---|---|---|
| **Buying** | Clear purchase intent with hard constraints (size, color, price, brand) | High-precision filter track, hard constraints locked in |
| **Browsing** | Open-ended, vague, exploratory | Diverse dense retrieval, unlocking cross-category scenario matching |

**Base pipeline** — an **in-memory** data flow:

```
Multi-Route Retrieval  →  LLM Semantic Ranking
  ├── keyword (BM25)
  ├── category
  └── vector similarity
```

### II. Dialog Strategy — Multi-Turn Scenario Evolution

**Dynamic State Machine** — a conversational state tracker that handles two opposite situations:

- **Information Accumulation** — slots build up turn by turn (constraints added)
- **Intent Override** — the user abruptly changes their mind → **clear and rewrite the slots**, never accumulate incorrectly

**Proactive Guidance** — on **over-generality** (the candidate pool is too large):
cut retrieval short immediately and **proactively generate a structured clarifying question** to converge faster.

### III. Self-Evolution — Dynamic Context Programming

- **Runtime Adaptation** — *personalized context distillation* from the conversation history,
  continuously updating **short-term session state** + **long-term user profile**
- **Adaptive Orchestration** — use dynamic Context Programming to **re-orchestrate the workflow at runtime**
  and align strategy, so the agent refines its own guidance logic each round

### IV. Evaluation Matrix — Product & Efficiency

Anchored to the **products actually purchased** in the Amazon dataset:

| Dimension | Metric | What it measures |
|---|---|---|
| **Coverage** | Hit Rate@K | Recall and catalog coverage at the retrieval stage |
| **Precision** | MRR / Top-K Hit Rate | Whether the LLM pushes the actually-purchased item to the **top** of the list |
| **Efficiency** | **MTTC** (Mean Turns to Conversion) | Heavily rewards systems that guide the user to the right product in **fewer turns**; penalizes unnecessary cognitive load |

> MTTC is the single biggest difference from an ordinary retrieval task:
> an extra clarifying question has to **pay for itself** — otherwise it just lowers your score.

---

## 3. Constraints & scope

### ✅ In scope
- A sensitive intent-detection module splitting traffic into "Buying" / "Browsing"
- Heterogeneous retrieval routing: weights, custom dynamic truncation, **time-based slot decay**
- A runtime-adaptive memory layer for personalized context distillation
- Fine-tuning prompt strategy / local scoring logic at the LLM ranking stage to compress the decision path

### ❌ Out of scope
- **UI/UX** — judged entirely through an automated backend API + headless pipeline
- Training or full-parameter fine-tuning of a base foundational LLM
- Deploying a heavy industrial external vector DB cluster — **must run entirely in-memory**
- Multi-modal — text catalog, structured metadata, and text dialog only

### ⚠️ Limits (easiest ways to lose points)
- **Max 10 turns/session — going over means forced termination and a ZERO SCORE**
- **Catalog is read-only** — no structural mutation, no fake ASINs

### Allowed assumptions
- Input is already clean text (no need to handle typos, spelling correction, or ASR noise)
- Catalog, prices, and the category tree are **static** for the whole hackathon
- Each session is one independent user (no need to handle multi-user concurrency)

---

## 4. Resources & data

A frozen, reproducible kit derived from **Amazon Reviews 2023**.

**Competition data**
- Frozen catalog of **50,000 products**, category `Clothing_Shoes_and_Jewelry`
- **200 labeled public dev sessions** — for local testing & iteration
- **800 private sessions** held by the organizers for final judging
- Public and private sets use **completely disjoint users and target products**

**Participant resources**
- A **weak BM25** starter agent (Python)
- A **deterministic local evaluator**: Hit Rate@10, MRR, MTTC, Efficiency, and an aggregate `TechnicalScore`
- Python Agent interface + machine-readable API contract
- Evaluation config, reproducible baseline results, data docs, submission rules
- A SHA256 checksum file to verify the downloaded catalog

**Links**
- Repo: https://github.com/TechJam2026/techjam-conversational-search
- Participant kit: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit
- Original data: https://amazon-reviews-2023.github.io/

You may replace or modify the starter agent, but you must still use the official local evaluator.
The kit supports: keyword retrieval, rule-based approaches, dense retrieval, hybrid retrieval, reranking, local models, and external model APIs.

> ⚠️ **The organizers do NOT provide** hosted model access, API keys, model tokens, or third-party credits.
> **Using a paid LLM is not required.** If you use an external service, you supply your own credentials
> and cover your own costs — and **never commit secrets to the repo**.
> There is no need to download or rebuild the full original Amazon Reviews 2023 dataset.

---

## 5. Deliverables

1. **Written project description (Devpost)** — how the solution addresses the problem statement; development tools (VSCode, Colab, Jupyter…); APIs used; libraries & frameworks; datasets & assets.
2. **Public GitHub repository** — clearly structured, well-commented code; README covering: project overview, setup & installation, steps to reproduce the results, a reflection on limitations + what you'd improve with more time, and each member's contributions.
3. **Demo video** — an end-to-end run, uploaded to YouTube as **public**, linked from Devpost, with no third-party trademarks or copyrighted content.
   *Backend/NLP tracks without a front end may instead submit an API usage walkthrough / inference examples / result analysis.*

---

## 6. Judging criteria

| Criterion | Weight | What it covers |
|---|---|---|
| **Technical Execution** | 35% | Solid engineering fundamentals, well-structured code, thoughtful architecture, effective use of APIs/models, a stable demo, and technical complexity that reflects deliberate decisions |
| **Innovation & Problem Insight** | 20% | Originality in both the idea and the approach; sharpness in framing the problem — why it matters and how the solution addresses it head-on |
| **Impact & Relevance** | 20% | Potential to create real value for users/stakeholders, with concrete reach and benefit beyond the scope of the hackathon prompt |
| **Feasibility & Practicality** | 15% | Realistic and buildable beyond a prototype; sensible resource use, an architecture that holds up under real conditions, and a grounded rather than hand-wavy implementation |
| **Presentation & Communication** | 10% | *[Final Event only]* A coherent pitch from problem → solution → potential, with substantive answers to questions |

---

## 7. Execution checklist

**Day 1 — Baseline & scaffolding**
- [ ] Clone the participant kit, verify the catalog via SHA256
- [ ] Get the starter BM25 agent + local evaluator running, record the baseline scores (Hit Rate@10 / MRR / MTTC)
- [ ] Build a harness that runs all 200 public sessions and prints the results table in one command
- [ ] Lock in the architecture: intent router → multi-route retrieval → LLM rank

**Day 2 — The four pillars**
- [ ] Buying vs Browsing intent classifier (measure its accuracy separately)
- [ ] Multi-route retrieval: BM25 + category filter + vector (in-memory, e.g. FAISS/numpy)
- [ ] Dialog state machine: slot accumulation **and** intent override (write dedicated tests for the override case)
- [ ] Over-generality threshold → generate structured clarifying questions
- [ ] LLM semantic ranking + prompt tuning

**Day 3 — Score optimization & submission**
- [ ] **Optimize MTTC**: as few turns as possible; ablate "is one more question worth it?"
- [ ] Context distillation / short-term + long-term user profile
- [ ] Verify that **no session exceeds 10 turns** (hard guard in code)
- [ ] Scan the repo to confirm **no API keys are exposed**
- [ ] README + demo video + Devpost description

---

## 8. Risks to watch

| Risk | Consequence | Mitigation |
|---|---|---|
| A session exceeds 10 turns | **Zero score for that session** | Hard counter in the agent loop, force a result at turn 9 |
| Too many clarifying questions | Poor MTTC → Efficiency drops | Only ask when the pool really is too broad; measure the before/after delta |
| Overfitting to the 200 public sessions | Poor performance on the 800 private sessions | Cross-validate; avoid hard-coding rules for specific samples |
| Dependence on a paid LLM API | Running out of credit mid-run / not reproducible | Keep a local scoring fallback; cache LLM responses |
| Committing `.env` / API keys | Rule violation, lost points | `.gitignore` + scan the repo before pushing |
| Heavy vector index | Violates "in-memory only" | Lightweight embeddings, in-process index, no external service |
