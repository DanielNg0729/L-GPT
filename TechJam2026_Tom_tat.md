# TikTok TechJam 2026 — Overview of all 5 tracks & problem statements

> Source: `TikTok_TechJam_2026_Tracks_and_Problem_Statements.pdf` (Early Bird Access, Feishu Docs).
> The PDF contains the **complete** content for all 5 tracks (the HTML version only shows Track 1, due to Feishu lazy-loading).

## Webinar (worth watching before picking a track)

- **Date:** Friday, August 28, 2026, 1–6pm SGT (GMT+8), 45 minutes per track
- **Link:** https://vc-my.larkoffice.com/j/484622806 (the official link is shared after the public release)

| Time | Track |
|---|---|
| 1:00–1:45pm | 1. Agent Launchpad: Design and Build Lightweight Agent Middleware |
| 2:00–2:45pm | 2. Autonomous ML Research Agent for Recommender Systems |
| 3:00–3:45pm | 3. Implement a GPU Kernel for a Transformer Layer |
| 4:00–4:45pm | 4. Shopping Copilot: AI Conversational Search and Recommendations |
| 5:00–5:45pm | 5. Robust Detection of AI-Generated Images Under Real-World Transformations |

## Quick comparison

| # | Track | Core skills | Main scoring driver | Difficulty / risk |
|---|---|---|---|---|
| 1 | Agent Middleware | Full-stack TS (React + Fastify), Docker, security/observability | 40% end-to-end middleware behavior | Moderate. A working starter kit and an open-ended prompt → easy to get into, but you have to scope wisely |
| 2 | ML Research Agent | LLM agent + RecSys (CTR/CVR), MLOps | Score delta on a hidden test (measured automatically) | **Highest.** Needs a GPU, ~80M samples of data, and training time |
| 3 | GPU Kernel | CUDA/Triton, profiling, optimization | 35% Technical Execution | Demanding expertise, but tight, clear scope; needs your own GPU |
| 4 | Shopping Copilot | RAG / hybrid retrieval, dialog state, prompt engineering | 35% Technical Execution + automated evaluator | **Most manageable.** In-memory, no GPU needed, local evaluator available |
| 5 | AIGC Image Detection | CV, robustness, augmentation | 35% Technical Execution | Moderate. Public datasets available, models limited to < 2B params |

Shared judging criteria for tracks 3, 4, and 5 (and close to the same for tracks 1 and 2):
Technical Execution 35% · Innovation & Problem Insight 20% · Impact & Relevance 20% ·
Feasibility & Practicality 15% · Presentation & Communication 10% *(Final Event only)*.

Shared deliverables for tracks 2–5: **Devpost description + public GitHub repo (with a complete README) + public YouTube demo video**.

---

# Track 1 — Agent Launchpad: Lightweight Agent Middleware

**Starter kit:** https://github.com/RrankPyramid/CodeJam
**Slogan:** *"Build the missing middleware, not the platform."*

## Context
An agent platform needs more than a chat frame: operators must be able to **understand what happened**,
**control what the agent can access**, and **isolate unsafe execution**. The starter kit handles all of the
platform work so teams can spend all 3 days on a single infrastructure problem.

## What the starter kit already provides
- Create/start/stop/delete agents from the browser, a multi-turn Playground, asynchronous Run status polling
- Codex CLI writing files & running commands in each agent's own workspace, with session resume
- Agent/message/Run metadata persisted to a local JSON store
- Each turn runs in a disposable Docker/Colima/Podman container
- BytePlus ModelArk connectivity (Responses-compatible endpoint)
- Optional deployment to BytePlus ECS (Terraform)

**Deliberately missing (your part):** user identity, trace timeline, audit model, hard sandbox policy.
This is a single-user POC; the bearer token is not an identity system; the JSON store is single-process;
ordinary containers are not a multi-tenant isolation boundary.

## Division of responsibility

| Area | Starter kit | Your work |
|---|---|---|
| Product experience | React UI, agent list, forms, lifecycle, Playground, Run status | Keep the baseline working; add only enough UI to expose the middleware |
| Control plane | Fastify API, validation, async Runs, AgentService, JSON persistence | Wire real middleware behavior into the backend path |
| Agent runtime | Codex CLI, durable sessions, per-agent workspaces, disposable containers | Hook middleware into the correct execution boundary |
| Infrastructure | Docker, Colima, Podman, Compose, ECS, Terraform | Use the smallest runtime path that proves the point. Cloud is **optional** |
| Middleware | (empty) | Choose / adapt / combine / **invent** a coherent set of capabilities |

## Running the baseline
Requirements: macOS/Linux · Node ≥ 22, npm ≥ 10 · a container engine · an **Ark model API key** + an `ep-...` endpoint

```bash
git clone https://github.com/RrankPyramid/CodeJam.git && cd CodeJam
ARK_API_KEY=your-ark-api-key ARK_MODEL=ep-your-endpoint-id npm run poc
# rootless podman: add CONTAINER_ENGINE=podman
# open http://localhost:3000
```
⚠️ `ARK_API_KEY` must be an **Ark model API key**, NOT your BytePlus account AK/SK — the wrong one gives a 401.

**Baseline acceptance test:** create an agent → send a Playground task ("Create a TypeScript hello-world CLI,
add a test, run it, and summarize the files you created") → the Run completes with a response → a follow-up
continues in the same session → stop/restart the agent and the workspace is still there.
**Do not start on middleware until the baseline passes.**
Before submitting you must run `npm run check` (TS check + server tests + production build).

## Middleware design requirements
- **Keep the baseline intact** (CRUD, lifecycle, Playground, persistence, model execution all still work)
- **Real behavior**: it has to run on the backend / runtime / data / infra path. A static screen with hard-coded messages **does not count**
- **Define the boundary**: which component owns the decision/event, what data flows through, what happens on failure
- **Convincing evidence**: demo both the happy path **and** failure/denial/recovery/degraded/abuse cases
- **Automated tests** for the core middleware behavior (not just UI render tests)
- **No leaked secrets** in source, git history, logs, traces, screenshots, browser storage, or demo output
- Prefer the smallest infrastructure; **local is the default judging path, ECS earns no extra points**

## Suggested middleware directions (not mandatory)
1. **Identity & authorization** — separate human principal from agent principal; per-agent identity that can be rotated/revoked; scoped, time-bound delegated authority; policy enforcement in the backend (not the UI); an approval boundary for risky actions; action attribution; secret handling & revocation. *Mock identity is acceptable* (e.g. prove that User A's agent cannot read User B's resources). A login screen without server-side authz does not count.
2. **Trace, audit & observability** — a Run is a linked chain of spans, not scattered logs. Stable IDs (agent/version/run/session/trace/span/actor), timings + status + error + retry, span categories (orchestration, model call, tool call, memory, sandbox, policy decision, approval, cloud op), redacted input/output, tokens/cost. UI: Run list + trace detail as a tree/timeline, filters, and a way to find the failing step.
3. **Layered agent architecture** — Experience / Control Plane / Identity & Policy / Agent Runtime / Execution & Data / Observability / Cloud Resource. Document the contract between layers.
4. **Threat modeling & safety** — credential theft, privilege escalation / confused deputy, prompt injection & tool misuse, sandbox escape, cross-user data exfiltration, runaway execution/cost (timeout, quota, budget, stop control), sensitive trace capture. *Note: the built-in CPU/memory/PID limits do NOT count as a new capability.*
5. **Multi-agent coordination** — shared session/topic, turn-selection rules, shared state, event history, timeout/retry. Sample demo: several agents counting down 10→1 in one shared conversation, with no duplicated or missing numbers.
6. **Your own idea** — lifecycle reconciliation & recovery, memory governance, human-in-the-loop, cost/budget control, provider abstraction, versioning & rollback, tool/model routing, credential exchange, auto-diagnosis & remediation.

## Suggested 3-day plan (from the prompt)
| Day | Goal | End-of-day evidence |
|---|---|---|
| 1 | Run & understand the baseline, settle on the problem + middleware story, define contracts, finish the first backend path | Baseline passes; one real middleware behavior can be triggered from the API or a test |
| 2 | Complete the core path, persist evidence, minimal UI, the key success/failure cases | The full scenario runs end-to-end from the browser down to backend/runtime/data/infra |
| 3 | Automated tests, error handling & cleanup, architecture diagram + README, rehearse the demo | `npm run check` passes; the demo fits in **3 minutes** |

## Deliverables & scoring
**Submit:** a 3-minute live demo · a one-page architecture diagram (middleware, data flow, trust boundaries, enforcement/instrumentation/recovery points) · the code repo (setup, rationale, design summary, tests, demo steps, limitations, no secrets).

| Criterion | Weight |
|---|---|
| End-to-end middleware behavior | **40%** |
| Technical design & integration | 25% |
| Verification & robustness | 20% |
| Demo & reproducibility | 15% |

**Start reading the code at:** `apps/server/src/types.ts`, `app.ts`, `agent-service.ts`, both `AgentRunner` implementations, then `apps/web/src/App.tsx`.

---

# Track 2 — Autonomous ML Research Agent for Recommender Systems

## The problem
Build an **Autonomous ML Research Agent** that runs the full MLE loop on its own:
read the prompt → EDA → feature engineering → train + tune → evaluate → reflect & revise → repeat.
The code for each stage is written by the agent itself; none of it is provided.
Prior work for reference: MLE-Bench (OpenAI), AIDE (Weco AI), AI-Scientist-v2 (Sakana AI).

The agent must: (1) reproduce the official baseline, (2) iterate on the pipeline autonomously, and (3) beat the baseline
on a hidden test. Only train + validation may be used — the agent **never sees the hidden test**.

## Benchmarks
| Dataset | Description | Metric | Scale |
|---|---|---|---|
| **AliCCP** (required, 100% of the primary score) | Taobao, impression→click→conversion funnel | CTR AUC (all impressions) / CVR AUC (clicked subset) | ~80 million samples |
| **KuaiRand** (bonus) | Short-video feed, 12 feedback signals + randomized exposure | NDCG@10 / Recall@50, click = positive | ~tens of millions of interactions |

Links: AliCCP https://tianchi.aliyun.com/dataset/408 · KuaiRand https://kuairand.com
Official baselines: NISE (AliCCP), CWM (KuaiRand).

## Rules
- **In scope:** any open-source library, any paper / public solution / pretrained weights, modifying any stage
- **Out of scope:** ❌ external training data; ❌ pretrained weights trained on these two benchmarks' test labels; ❌ touching the hidden test during development
- Compute budget: TBD. The hidden test is scored **exactly once**, on the final submission

## Scoring
| Criterion | Weight | What it covers |
|---|---|---|
| Technical Execution | 35% | **Primary metric**: `delta(m) = score_agent(m) − score_baseline(m)`, averaged across metrics. Scored at the **converged point** (validation score improves by less than ε over N rounds, or the budget runs out), using the validation-best checkpoint. Plus **robustness**: on failure it recovers/retries/routes around, without crashing, stalling, or diverging |
| Innovation & Problem Insight | 20% | Which improvements the agent chose and **why** — the implementation itself is not scored here |
| Impact & Relevance | 20% | **Autonomy** — measured by the **number of manual interventions**; fewer is better |
| Feasibility & Practicality | 15% | **Resources**: total tokens (in+out) + total **GPU-hours** to reach convergence |
| Presentation & Communication | 10% | Final Event only |

## Track-specific deliverables
Beyond Devpost + repo: **run/iteration logs** (per round: hypothesis, code diff, metric, error & recovery),
the number of manual interventions, final predictions/checkpoint in the specified schema, a results table with the delta vs baseline,
and a report of token and GPU-hour usage.

> There is an Appendix A: a recommender-systems primer (recall→pre-rank→rank→re-rank pipeline, CTR/CVR, multi-task, AUC/NDCG/Recall, embeddings & feature crossing) — 1–2 hours of reading is enough grounding.
> Suggested tooling: any LLM coding agent, or ByteDance's **Trae** (7-day trial).

---

# Track 3 — Implement a GPU Kernel for a Transformer Layer

## The problem
Optimize the runtime of a Transformer layer on GPU using an **AI-assisted** approach.
Submit one or more GPU kernels implementing that layer and pass the test cases.

- Test cases are written in **PyTorch or TensorFlow** (picking one is enough); you may modify the layer's implementation → you decide what to fuse into a single kernel
- **Allowed error: relative error < 0.02, absolute error < 0.002** against the original
- Multiple shapes are tested (batch size, sequence length, large/small dimensions) — you may write shape checks and use different implementations per shape; **all shape combinations will be published in advance**
- Run & optimize **on your own machine** (your own GPU) → the optimization is card-dependent
- Using AI tools is strongly encouraged; **a clear tech report on the AI skills/tools used earns bonus points**

## Suggested optimization directions
Operator fusion · memory layout · reduced precision · tensor cores · softmax optimization ·
custom CUDA / Triton / TF / PyTorch implementations.

## Scope
In scope: AI-based code generation, GPU kernel fusion, use of profiling tools.
Out of scope: production-ready deployment.

## What to do
1. Download the benchmark script (`torch_transformer_benchmark.py` **or** `tensorflow_transformer_benchmark.py`)
2. Implement the customized-implementation section and optimize it as far as you can
3. Run it on your own machine
4. Write a tech report: environment (CPU/GPU/DISK), the optimizations you applied, and the final test results

Scoring: the shared 35/20/20/15/10 criteria.

---

# Track 4 — Shopping Copilot: AI Conversational Search & Recommendations

## The problem
Build a conversational shopping agent over an Amazon dataset that goes beyond static keyword matching.
Four required pillars:

**I. Core Architecture — Intent Routing & Hybrid Pipeline**
- **Dual-track routing**: detect intent → a high-precision filter track for "Buying" (hard constraints locked in) / a diverse dense-retrieval track for "Browsing" (cross-category)
- **Pipeline**: in-memory, "Multi-Route Retrieval → LLM Semantic Ranking" (keyword + category + vector similarity)

**II. Dialog Strategy — Multi-Turn Scenario Evolution**
- **Dynamic state machine**: handle both information accumulation (slots build up) and intent override (clear & rewrite slots)
- **Proactive guidance**: on over-generality (candidate pool too large) → cut retrieval short, proactively ask a structured clarifying question

**III. Self-Evolution — Dynamic Context Programming**
- **Runtime adaptation**: personalized context distillation from the conversation history, updating short-term session state + long-term user profile
- **Adaptive orchestration**: re-orchestrate the workflow at runtime, refining the guidance logic autonomously

**IV. Evaluation Matrix** (anchored to real purchase records in the dataset)
- **Coverage**: Hit Rate@K (retrieval stage)
- **Precision**: MRR / Top-K Hit Rate (does the LLM push the purchased item to the top?)
- **Efficiency**: **MTTC (Mean Turns to Conversion)** — fewer turns = higher score

## Constraints
- ❌ UI/UX (judged entirely through the backend API + headless pipeline)
- ❌ Training / full fine-tuning of a base LLM
- ❌ External vector DB clusters — **must run entirely in-memory**
- ❌ Multi-modal — text catalog, structured metadata, and text dialog only
- ⚠️ **Hard limit of 10 turns/session — going over means termination and a zero score**
- ⚠️ Catalog is read-only; no mutation and no fake ASINs
- Assumptions: input is already clean (no typo/ASR-noise handling needed), the catalog is static, one user per session

## Resources
- A frozen catalog of **50,000 products** from Amazon Reviews 2023 (Clothing_Shoes_and_Jewelry)
- **200 public sessions** for development + **800 private sessions** for final judging (disjoint users & target products)
- A weak BM25 starter agent, a **deterministic local evaluator** (Hit Rate@10, MRR, MTTC, Efficiency, TechnicalScore), a Python agent interface + API contract, SHA256 checksums
- Repo: https://github.com/TechJam2026/techjam-conversational-search
  (Kit release: `/releases/tag/participant-kit`) · Original data: https://amazon-reviews-2023.github.io/
- ⚠️ **The organizers do NOT provide API keys / model access / credits** — if you use an external LLM you pay for it yourself, and **secrets must never be committed**. Using a paid LLM is not required.

Scoring: the shared 35/20/20/15/10 criteria.

---

# Track 5 — Robust Detection of AI-Generated Images

## The problem
Distinguish AI-generated images from real ones while **retaining accuracy after the image has been post-processed or re-shared**.
Accuracy on clean data is not enough — you must explicitly discuss the trade-off between robustness, generalization, and false positives.

## Transformations the detector must survive
| Transform | Parameters | Real-world situation |
|---|---|---|
| JPEG compression | quality 90 / 70 / 50 / 30 | Re-encoding on social media and messaging apps |
| Gaussian blur | σ = 0.5 / 1.0 / 2.0 | Out of focus |
| Resize | scale 0.5× / 0.25×, then upscaled back | Thumbnail generation |
| Gaussian noise | σ = 0.02 / 0.05 / 0.10 | Sensor noise in low light |
| Color jitter | brightness/contrast/saturation ±20% | App filters, auto-enhance |
| Center crop | 80% crop | Cropping for a profile picture |

## Constraints
- In scope: image-level AIGC detection, robustness, feature engineering, model design, evaluation design, error analysis, explainability
- Out of scope: production deployment, platform-wide moderation systems, video/audio
- ⚠️ **The model must be < 2B parameters**; hackathon prototype scale

## Datasets
- https://huggingface.co/datasets/saberzl/SID_Set
- https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- https://modelscope.cn/datasets/hy2628982280/WildFake/summary
- **Demo validation set (NOT for training, not scored):** a WildFake subset — non-AIGC COCO val2017 (4,998 images), AIGC DALL·E Advanced (8,843 images)

## Track-specific deliverables
Beyond Devpost + repo + demo video:
- **A script that takes an image directory and outputs JSON containing `image_path` and `pred`** (confidence that it is AIGC)
- **A robustness evaluation summary**: a table/chart comparing clean vs transformed
- **An error analysis note**: representative false positives / false negatives and the trade-offs

Scoring: the shared 35/20/20/15/10 criteria.
