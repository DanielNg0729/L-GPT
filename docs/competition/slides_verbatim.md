# Slide 1

## Multi-turn conversational e-commerce search

- 72-hour challenge for university students
- Build an Agent that can search, ask, remember and rank

---

# Slide 2

## A vague request must become a useful search plan

### CUSTOMERS REVEAL INTENT GRADUALLY

category → use case → material → style → budget

**CUSTOMER** “I need shoes for a trip.”

**AGENT** “Long walks? Any material or budget preference?”

**CUSTOMER** “Water-resistant, comfortable and under $80.”

**STATE** travel · long walking · water-resistant · comfort · budget ≤ $80

**ACTION** search → ask → remember → re-rank → Top 10

**A better question can be more valuable than another retrieval call.**

---

# Slide 3

## A vague request must become a useful search plan

### CUSTOMERS REVEAL INTENT GRADUALLY

category → use case → material → style → budget

**CUSTOMER** “I need shoes for a trip.”

**AGENT** “Long walks? Any material or budget preference?”

**CUSTOMER** “Water-resistant, comfortable and under $80.”

**STATE** travel · long walking · water-resistant · comfort · budget ≤ $80

**ACTION** search → ask → remember → re-rank → Top 10

**A better question can be more valuable than another retrieval call.**

---

# Slide 4

## One runnable Agent must find one hidden target

### GOAL

- Find the hidden target product as early and as highly ranked as possible.

### THE ORGANIZER PROVIDES

- Frozen catalog · public sessions · simulator · evaluator · starter code

### THE TEAM BUILDS

- One Python Agent that asks useful questions, keeps active constraints and returns up to 10 ranked parent_asin values.

### SESSION LIMIT

- The conversation stops after a valid hit or after turn 10. No hosted service is required.

---

# Slide 5

## Keep the work focused on agent intelligence

### IN SCOPE

- Keyword, dense or hybrid retrieval · query rewriting · semantic reranking
- Conversation state · clarification strategy · safe profile use

### NOT REQUIRED

- User interface · full-model training · multimodal search
- Real transactions · catalog modification · production infrastructure

A beginner can start with BM25 and rules.

**Strong teams win through better retrieval and dialogue decisions.**

---

# Slide 6

## Real amazon records create a scalable clothing benchmark

### DATA ORIGIN

- Amazon Reviews 2023 · McAuley Lab, UCSD

### INPUT SCOPE

- 2,524,981 official Clothing 5-core leave-last-out records
- 50,000 frozen catalog products

### REAL SIGNAL

- Earlier eligible purchases form history; the final eligible purchase becomes the hidden target.
- Customer dialogue is simulated  -  it is not copied from Amazon reviews.

### QUALITY AUDIT

- Catalog-joined eligible records&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;10,187
- Distinct candidate targets&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1,406
- Public/private target overlap&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0

---

# Slide 7

## The participant release is explicit, compact and downloadable

### VISIBLE CATALOG FIELDS

- parent_asin · title · features · details · description · categories
- store · average_rating · rating_number · price

### RELEASE SCALE

- 50,000 frozen catalog products
- 200 labeled public development sessions
- 800 organizer-only final sessions
- 1,000 user- and target-disjoint benchmark sessions

### DOWNLOADS

- catalog.jsonl.gz and techjam-participant-kit.zip
- Repository github.com/TechJam2026/techjam-conversational-search
- Data release github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit

---

# Slide 8

## A reproducible pipeline creates the public and private sets

- Start from the official Clothing 5-core leave-last-out split.
- Join targets and visible history to the frozen catalog.
- Require usable pre-target catalog history.
- Select distinct users and target products deterministically.
- Build anonymous profiles and organizer-only intent cards.
- Split public/private by user and target, then freeze checksums.

2,524,981 official leave-last-out records

→ 10,187 eligible catalog-joined records

→ 1,406 distinct candidate targets

→ 200 public + 800 private sessions + 50,000 frozen products

---

# Slide 9

## Private labels stay private

### PARTICIPANTS CAN SEE

- Frozen catalog fields · anonymous aggregate profile
- Public customer messages · public target parent_asin for local development

### ORGANIZER KEEPS PRIVATE

- 800 target parent_asin labels · hidden intent cards · simulator state
- Raw user IDs · raw histories · reviews · timestamps

### VERIFIED BEFORE RELEASE

- 0 public/private user overlap
- 0 public/private target overlap
- 0 target records in visible history
- 0 intent_card fields in released participant data

The private set is never placed in the participant repository.

---

# Slide 10

## Four customer behaviors test different Agent skills

|  |  |
|---|---|
| 40% BUYING | a hard constraint appears early |
| 40% BROWSING | the request begins vague |
| 15% INTENT OVERRIDE | a preference changes on turn 3 or 4 |
| 5% BOUNDARY | the customer may have no preference |

### OVERRIDE EXAMPLE

- Turn 1: black running shoes
- Turn 3: “Actually, make them casual white sneakers.”

**Weak Agent:** appends contradictory words

**Strong Agent:** replaces black → white and running → casual, then reranks.

---

# Slide 11

## A simple agent contract enables controlled evaluation

### LOCAL SESSION LOOP

- reset → customer message → Agent response → validate → exact match → reply or stop

### OFFICIAL PYTHON INTERFACE

- reset(session_id, user_profile)
- respond(session_id, user_message, turn, top_k)

### RESPONSE FIELDS

- message · ask_attribute · ordered recommendations · optional usage

### VALIDATION RULES

- First 10 unique, catalog-valid parent_asin values are scored.
- Duplicates and invalid IDs are removed; numeric scores are ignored.
- Exact equality is required. Exceptions, invalid output or timeout may count as a miss.
- Maximum 10 turns; Intent Override cannot score before the changed intent appears.

The evaluator imports the submission locally  -  no URL or fixed port.

---

# Slide 12

## Accuracy, rank and speed determine the score

### PER SESSION

- Hit@10 = 1 if the target is in the scored Top 10; otherwise 0.
- Reciprocal Rank = 1 / target rank; miss = 0.
- First-hit turn = 1-10; miss is assigned 11.

### AGGREGATE OVER 800 PRIVATE SESSIONS

- HR@10 = successful sessions / N
- MRR = Σ reciprocal rank / N
- MTTC = Σ first-hit turn / N
- Efficiency = clip((11 − MTTC) / 10, 0, 1)
- TechnicalScore = 0.50 × HR@10 + 0.30 × MRR + 0.20 × Efficiency
- Exact parent_asin matching · breakdowns: Buying · Browsing · Intent Override · Boundary

---

# Slide 13

## Automated results lead; innovation completes the judging

### OFFICIAL EVENT-LEVEL JUDGING

35% Technical Execution

20% Innovation & Problem Insight

20% Impact & Relevance

15% Feasibility & Practicality

10% Presentation & Communication

### TECHNICAL EVIDENCE

Coverage through Hit Rate@K

Ranking precision through MRR and Top-K Hit Rate

Conversational efficiency through MTTC

### MODEL AND COST POLICY

The solution includes LLM Semantic Ranking.

Teams using external services manage their own credentials,

usage limits and costs.

**Final judging follows the published TechJam criteria.**

---

# Slide 14

## The baseline gap leaves a real competition ladder

### PUBLIC-SET SMOKE TEST · 200 SESSIONS

|  | Hit Rate@10 | MRR | MTTC↓ |
|---|---:|---:|---:|
| Weak BM25, no state | 12.5% | 0.068034 | 9.81 |

- The weak starter provides an accessible, reproducible starting point.
- It uses no LLM and no conversation state.
- The low score leaves substantial room for retrieval, state and clarification improvements.

These are public-set reference results, not leaderboard targets.
