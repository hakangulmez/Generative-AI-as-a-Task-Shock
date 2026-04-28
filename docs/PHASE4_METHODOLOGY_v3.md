# Phase 4 Methodology v3 — Software-Adapted Replicability Framework (Final)

**Date:** 2026-04-27
**Status:** APPROVED — implementation in progress
**Author:** Hakan Zeki Gülmez (thesis author)
**Supersedes:** PHASE4_METHODOLOGY_v2.md (added Eloundou limitation argument; verified Eisfeldt citation source)

---

## Section 0 — How to Use This Document

This document is the **authoritative reference** for Phase 4 methodology. It supersedes all prior methodology documents:
- `PHASE4_METHODOLOGY_PIVOT.md` (v0, deprecated 2026-04-25)
- `PHASE4_METHODOLOGY_v1.md` (deprecated 2026-04-27 after empirical sanity test)
- `PHASE4_METHODOLOGY_v2.md` (superseded 2026-04-27 with foundation refinements; conceptually identical to v3)

Every Claude instance starting work on this thesis MUST read this document in full before producing any spec, code, or analysis.

If a Claude proposes:
- Returning to ONET embedding match as the primary scoring methodology (it failed; see Section 3.2)
- Letting LLM apply Eloundou's worker-level E0/E1/E2 rubric directly to product tasks (this was the original Phase 4 approach; failed for determinism reasons; see Section 3.1)
- Single-run scoring without multi-run averaging
- Modifying the R0/R1/R2 rubric without explicit author authorization

→ The proposal is methodologically wrong and must be refused with reference to the relevant section.

---

## Section 1 — Executive Summary

This thesis measures B2B software product exposure to LLM substitution following the ChatGPT shock (2022Q4). The treatment variable ρ_i quantifies the **functional replicability of a firm's product tasks by LLMs**.

**Methodology in one paragraph:**

For each firm, we extract product tasks from pre-shock 10-K Item 1 (LLM-mediated extraction). For each product task, the LLM classifies it as **R0** (outside LLM scope), **R1** (direct LLM substitute), or **R2** (LLM + tools substitute) using a software-adapted rubric inspired by Eloundou et al. (2024). We aggregate using Eloundou's β formula: ρ_i = (R1_count + 0.5 × R2_count) / n_tasks × 99 + 1, scaled to [1, 100]. Each firm is scored 3 times; final ρ_i is the mean. Reliability is validated via ICC(3,1) ≥ 0.90 on a 15-firm subsample.

**Conceptual claim:** B2B software products historically served as toolkits for customer firm employees. The LLM shock threatens products by enabling LLMs to replicate the **functions the products perform** — not the workers' tasks (Eisfeldt-Schubert-Zhang 2023 covers that), but the software's tasks themselves. Our measurement captures this threat at the **functional replicability** dimension. Adoption barriers (scale, latency, integration depth, switching costs, regulatory friction) are captured separately in the demand-side variable δ_i.

**Why this approach (not ONET-embedding):** We attempted to inherit Eloundou's published worker-level labels via embedding similarity (v1 methodology). Empirical sanity testing (2026-04-27) demonstrated this approach cannot differentiate firms whose products serve cognitive workers — both ZS-type security infrastructure and EGAN-type chatbot platforms have customers performing cognitive tasks that ONET uniformly labels E1/E2. Eloundou's corpus measures worker-level replicability, not software-level. This is consistent with limitations Eloundou et al. (2024) themselves acknowledge (annotator pool not occupationally diverse; rubric subjective). We pivot to a software-adapted rubric (R0/R1/R2) applied via LLM classification, with multi-run averaging for noise control. This is conceptually closer to Eisfeldt-Schubert-Zhang's detailed-rubric approach than to Eloundou's task-corpus approach.

**Why we accept this trade-off:** The methodologically pure path (inherit published labels) failed empirically. The next-best path (adapt rubric to the right unit of analysis, manage noise via multi-run) is what Eisfeldt-Schubert-Zhang did within the worker domain. We do the same within the software domain. The trade-off (mild researcher discretion in rubric adaptation, in exchange for valid measurement) is necessary and defensible.

---

## Section 2 — Conceptual Framework

### 2.1 The Pre-LLM World — Worker+Software Complementary

Before 2022, the B2B software industry operated under a coherent economic logic:

> Customer firm employees (workers) performed business tasks. B2B software products served as **toolkits** that augmented worker capabilities, enabling tasks to be performed at greater scale, lower cost, or higher quality than workers alone could achieve.

This is the standard task substitution framework (Acemoglu & Restrepo 2018, 2022) applied to enterprise software: software automated routine cognitive tasks, complementing remaining human judgment, customer interaction, and exception handling.

**Key implication:** B2B software products derive their value from **enabling worker task execution**, not from existing independently of workers. A CRM exists because sales teams need it. A security platform exists because IT teams need it. A spreadsheet exists because analysts need it.

### 2.2 The LLM Shock — Threat Vectors

The release of ChatGPT (2022-11-30) introduced a new technological capability: an LLM that could itself perform many cognitive tasks previously requiring human workers. This creates three distinct threat scenarios for any product:

**Scenario S1 — Worker substitution, software preserved**
LLM replaces the worker for a given task; the software product remains necessary for execution.

Example: HubSpot's marketing automation. A copywriter (worker) historically wrote email copy; HubSpot scheduled and tracked the campaigns. LLM can now write the copy, replacing the worker. HubSpot still schedules and tracks. **HubSpot retains value but loses some "user" complexity.**

**Scenario S2 — Software substitution, worker preserved**
Worker remains necessary; LLM replaces the software's contribution to the workflow.

Example: A legal team historically used a specialized contract review tool (software) to highlight risk clauses, with attorneys (workers) making final decisions. With LLM, attorneys may use ChatGPT directly for the same review, bypassing the specialized tool. **Software loses, worker retained.**

**Scenario S3 — Joint substitution**
Neither worker nor specialized software needed; LLM (with possible additional tooling) handles the entire task autonomously.

Example: Tier-1 customer support. Historically a human agent + ticketing software handled inquiries. An autonomous LLM agent now handles the entire flow. **Both worker and existing software displaced.**

**Implication for measurement:** Our treatment variable ρ_i must capture the threat to **the software product**, which corresponds to S2 and S3 scenarios (where software is substitutable). S1 affects the worker, not the product, and is the focus of Eisfeldt-Schubert-Zhang (2023). Our extension shifts the measurement target one level up the value chain.

### 2.3 What ρ_i Measures and Does Not Measure

ρ_i measures **functional replicability**: can the software task be performed by an LLM (or LLM + standard tools) with equivalent quality?

ρ_i does NOT measure:
- **Scale or latency feasibility:** Whether the LLM can perform the task at the throughput the product handles. Real-time security inspection at 250B requests/day is functionally text-classifiable but operationally infeasible for current LLMs at that scale. This dimension is captured in the rubric: tasks requiring such scale are R0 (outside LLM operational scope). Even so, residual scale-related friction shifts to δ_switch (technical switching cost).
- **Integration depth:** Whether the LLM can replicate the product's deep integration with customer systems. This shifts to δ_switch.
- **Regulatory acceptability:** Whether the LLM's output is acceptable in regulated decision contexts. This shifts to δ_error.
- **Data dependency:** Whether the LLM has access to the proprietary data the product accumulates. This shifts to δ_data.

**Effective threat = ρ × (1 - δ).** A firm with high functional replicability (high ρ) but high adoption friction (high δ) faces low effective threat. The thesis's central empirical question is whether realized revenue effects align with functional replicability or with effective threat.

This separation is methodologically deliberate. ρ captures the **technological possibility** of substitution; δ captures the **economic feasibility** of substitution. Conflating them obscures the channels through which AI shocks transmit to firm performance.

---

## Section 3 — The Methodology Journey: Why Three Approaches Were Tried

This section documents the methodology evolution because the **rejection of two alternatives** is itself defensible argumentation for the final choice. The thesis methodology chapter will reference this evolution explicitly.

### 3.1 Approach 1: LLM-Driven Direct Classification (Deprecated 2026-04-25)

**What it was:** Single LLM call extracts product tasks AND classifies them as E0/E1/E2 using Eloundou's rubric in-context.

**Why it failed:**

*Empirical reason — determinism:*
Mini smoke test (2026-04-25):
- ZS run 1: 9 tasks, 4E0/3E1/2E2, ρ = 45.0
- ZS run 2 (same prompt, same text, temperature=0): 8 tasks, 5E0/2E1/1E2, ρ = 31.9
- **Δρ = 13.1 points across two runs of the same firm.**

This level of single-firm noise is methodologically unacceptable for a treatment variable in difference-in-differences identification. Treatment-side noise causes attenuation bias in interaction coefficients—the central inference of the thesis.

*Methodological reason — research integrity:*
Author's principle (verbatim):
> "Eğer kendim bir product task datası yaratırsam, ki benim ana treatment'ımı etkileyen bir şey, tezde güvenilirlik azalır."
>
> *("If I generate a product task dataset myself, which is something that affects my main treatment, the credibility of the thesis decreases.")*

Letting the LLM apply Eloundou's rubric in-context means the labels are "Claude's interpretation of Eloundou," not Eloundou's published labels. Defense becomes circular.

**Lesson learned:** Single-call extraction + classification produces noise too large for treatment-grade measurement. Either (a) decouple extraction from classification, or (b) use multi-run averaging for classification.

### 3.2 Approach 2: ONET Embedding Match (Deprecated 2026-04-27)

**What it was:** LLM extracts product tasks and identifies the worker task each is "used in." Embed the worker task description, match against Eloundou's 19,265-task ONET corpus via cosine similarity, inherit the matched task's published gpt4_exposure label.

**Conceptual appeal:**
- Methodologically pure: labels come from Eloundou's published corpus, no researcher discretion in classification
- Eisfeldt-Schubert-Zhang (2023) extension: their approach + product-level aggregation
- Deterministic labeling step

**Why it failed:**

*Sanity test (2026-04-27, $0 cost, ~1 minute):*

Tested four hand-crafted worker task descriptions for ZS (expected: low ρ) and EGAN (expected: high ρ):

| Firm | Worker task | Top-1 ONET match | Label | Similarity |
|------|-------------|------------------|-------|------------|
| ZS | Network admins monitor systems for threats | "Develop or recommend network security measures" | E1 | 0.638 |
| ZS | Security analysts investigate incidents | "Investigate computer security incidents" | E1 | 0.662 |
| ZS | Security architects design access control | (network security analysis) | E2 | 0.658 |
| ZS | IT operations engineers configure firewalls | (security policy implementation) | E2 | 0.651 |
| EGAN | Customer service reps respond via chat | (customer service interaction) | E2 | 0.593 |
| EGAN | Support agents classify and route tickets | (customer support routing) | E2 | 0.590 |
| EGAN | Content authors write knowledge base articles | (knowledge management) | E1 | 0.490 |
| EGAN | CX analysts analyze conversation transcripts | (customer feedback analysis) | E1 | 0.529 |

**Both firms produced identical R-distributions: 2 E1 + 3 E2, predicted ρ ≈ 70.**

*Why ONET cannot differentiate them:*
Eloundou's corpus measures **worker-level** replicability. ZS's customers are network administrators (cognitive workers performing log analysis, threat investigation, policy writing). EGAN's customers are customer service representatives (cognitive workers performing chat handling, ticket classification, KB curation). **Both groups perform cognitive work**, and Eloundou uniformly labels cognitive work as E1 or E2. E0 in Eloundou's corpus corresponds to physical work, equipment operation, board-level oversight—not knowledge worker tasks at all.

The worker-level corpus has no granularity to distinguish "security infrastructure for cognitive workers" from "chatbot platform for cognitive workers." The differentiation we need exists at the **software-task level**, not the worker-task level.

*Consistency with Eloundou's own acknowledged limitations:*
Eloundou et al. (2024) explicitly acknowledge their annotator pool is not occupationally diverse:

> "A fundamental limitation of our approach lies in the subjectivity of the labeling. In our study, we employ annotators who are familiar with the GPT models' capabilities. However, this group is not occupationally diverse, potentially leading to biased judgments regarding GPTs' reliability and effectiveness in performing tasks within unfamiliar occupations."

The annotator pool consisted of GPT-familiar individuals (5 named annotators in the acknowledgments). For knowledge-worker occupations, GPT-familiar annotators systematically perceive cognitive tasks as LLM-replicable, producing the dense E1/E2 labeling that compresses our sample. Eloundou themselves note: "Although we use multiple annotation sources, none is considered the definitive ground truth."

Our sanity test result is therefore not a surprise — it is the empirical manifestation of a limitation the original authors document. The ONET corpus is well-suited to its design purpose (general workforce exposure measurement) but not to ours (B2B software product differentiation).

**Lesson learned:** Eloundou's corpus is the wrong unit of analysis for our research question. We measure software replicability; Eloundou measures worker replicability. They are conceptually distinct, and the empirical mismatch confirms the conceptual mismatch.

### 3.3 Approach 3 (Final): Software-Adapted Rubric + Multi-Run Classification

**What it is:** LLM extracts product tasks and classifies them using a **software-adapted rubric** (R0/R1/R2) defined explicitly for software-task replicability. Each firm is scored 3 times; final ρ_i is the mean. Reliability is validated via ICC(3,1) ≥ 0.90 on a 15-firm subsample.

**Why this is the right path:**

1. **Conceptual alignment.** Software-adapted rubric measures what we need to measure (software-level functional replicability). Eloundou-style worker rubric does not.

2. **Eisfeldt-Schubert-Zhang (2023) precedent.** Their methodology uses Eloundou's rubric **as detailed by themselves** in their own paper appendix. They expand Eloundou's E1 definition with concrete software-related examples ("writing and transforming text and code, providing edits, translating text, summarizing documents..."). This is rubric adaptation, not rubric inheritance. We do the same — adapt the rubric to our unit of analysis (software tasks instead of worker tasks). Eisfeldt explicitly characterize this as ChatGPT acting as "a research assistant mapping task statements into existing categories" (Internet Appendix C).

3. **Determinism handled by multi-run averaging.** The Approach 1 problem (13-point single-run noise) is mitigated by averaging 3 runs. Expected residual noise: ~13/√3 ≈ 7.5 points per firm. ICC reliability test verifies adequate consistency.

4. **Phase 6 robustness includes Approach 2.** ONET embedding match is preserved as an alternative specification in robustness analysis, providing an external validity check using Eloundou's published labels (even though those labels are conceptually downstream of our research question).

**Trade-off acknowledged:** This approach reintroduces some researcher discretion in the rubric definition. The author's principle ("don't generate the dataset that drives your treatment") is partially compromised — we don't generate the labels per firm, but we do generate the rubric used by the labeling LLM. We accept this trade-off because:
- The pure-inheritance alternative (Approach 2) failed empirically
- Rubric adaptation has direct precedent in Eisfeldt-Schubert-Zhang (2023)
- Eloundou themselves acknowledge their rubric is subjective and their annotator pool not occupationally diverse — pure inheritance was never as discretion-free as it appeared
- Multi-run reliability protocol manages stochasticity
- Phase 6 robustness includes the pure-inheritance alternative as comparator

---

## Section 4 — The R0/R1/R2 Rubric (Software-Adapted)

### 4.1 Conceptual Definition

The rubric mirrors Eloundou's E0/E1/E2 conceptual structure but applies to **software tasks** rather than **worker tasks**:

| Eloundou (worker) | This thesis (software) |
|-------------------|------------------------|
| E0: Worker task LLM cannot replicate | R0: Software task LLM cannot replicate |
| E1: Worker task LLM can replicate alone | R1: Software task LLM can replicate alone |
| E2: Worker task LLM + tools can replicate | R2: Software task LLM + tools can replicate |

The shift in unit of analysis (worker → software) requires reframing what "replicate" means:

- For Eloundou: Replication = LLM performing the same task as the worker would, with equivalent quality, in ≤50% of the time
- For us: Replication = LLM (with or without standard tools) performing the same function as the software does, with equivalent business outcome, accessible to the customer firm without requiring the original software

### 4.2 Detailed Rubric Definitions

**R0 — Outside LLM scope**

The software task cannot be replicated by LLM technology (vanilla or tool-augmented) due to fundamental capability gaps. Typical reasons (any one is sufficient for R0):

- **Real-time / sub-second latency:** Task must complete in milliseconds; LLM inference is too slow regardless of model size
- **Massive scale / parallel processing:** Task volume (millions of operations per second, billions per day) exceeds LLM throughput at any economically feasible cost
- **Direct hardware/network/system control:** Task requires kernel-level access, packet-level processing, or physical system control that LLMs cannot perform
- **Strict deterministic computation:** Task must produce exact, reproducible outputs (regulatory calculations, fiduciary computations, cryptographic operations)
- **Continuous live data streams:** Task processes streaming inputs (sensor data, market ticks, network packets) at rates incompatible with LLM batch inference

**Examples (R0):**
- Process 250 billion network packets per second for security inspection (latency + scale)
- Execute high-frequency trading orders with microsecond response time (latency + determinism)
- Render 3D graphics in real-time for video games (compute scale + latency)
- Maintain continuous database replication across geographic regions (continuous + scale)
- Calculate index constituents per regulatory schema (deterministic + compliance)

**R1 — Direct LLM substitute**

The software task can be replicated by vanilla LLM (ChatGPT, Claude, etc.) accessed directly through a standard interface, with equivalent quality and accessibility for the customer. The LLM alone is sufficient — no additional software, no tool integration, no custom build needed. The customer can substitute the product with the LLM directly.

**Examples (R1):**
- Generate marketing email content from a brief
- Translate customer support messages between languages
- Summarize meeting notes into action items
- Write product descriptions for catalog listings
- Generate first-draft contract clauses
- Provide chatbot responses to customer inquiries
- Draft routine business correspondence

**R2 — LLM + tools substitute**

Vanilla LLM is insufficient, but the software task can be replicated by LLM combined with standard, off-the-shelf tooling that already exists in 2024 (RAG applications, code interpreter, browser plugins, database connectors, image generation models). The customer would need to assemble or use existing LLM-tool combinations rather than the product, but the building blocks all exist and are accessible.

**Examples (R2):**
- Generate financial reports from database queries (LLM + SQL connector)
- Categorize and route customer tickets with reference to knowledge base (LLM + RAG)
- Score lead qualifications based on CRM data (LLM + database access)
- Identify suspicious transactions in payment records (LLM + transaction database + fraud rule library)
- Generate code based on technical requirements (LLM + code interpreter + linter)
- Produce localized marketing content with brand assets (LLM + image generation + brand guidelines)

### 4.3 Boundary Cases — Critical for Reliability (Worked Examples)

The R0/R2 boundary is where classification noise concentrates. Tasks involving "real-time" or "near real-time" requirements can plausibly fall into either category depending on interpretation. The prompt MUST include explicit boundary examples to stabilize classification:

| Task description | Correct label | Reasoning |
|------------------|---------------|-----------|
| Real-time alerting that must fire within 500ms | R0 | Latency requirement is fundamental — LLM inference cannot meet 500ms SLA |
| Hourly anomaly summary report | R2 | Batch generation acceptable; LLM + log database can produce summaries |
| Streaming dashboard refresh every 5 seconds | R0 | Continuous refresh at 5s intervals exceeds LLM inference economics at scale |
| Daily security incident digest | R1 | Text summarization, no scale constraint — vanilla LLM sufficient |
| Real-time fraud detection on each transaction | R0 | Per-transaction sub-second decision required; LLM cannot meet latency |
| Weekly fraud pattern report | R2 | Batch analysis with database access, R2 territory |
| Live video stream content moderation | R0 | Continuous video analysis exceeds LLM throughput economics |
| Post-event content moderation review | R1 | Async text+image classification, LLM-capable |

These boundary examples are operationalized in the prompt with the heuristic: "If the task requires sub-second per-operation latency or continuous streaming, → R0. If batch-mode acceptable with hourly or longer cycles, → R1 or R2 depending on tool requirement."

### 4.4 Decision Heuristics for the LLM

When classifying a software task, the LLM is instructed to ask three questions in order:

**Q1: Can vanilla LLM produce the same business outcome at equivalent quality, accessible to the customer through a standard chat interface?**
→ If YES → R1
→ If NO → continue to Q2

**Q2: Can LLM + standard, currently-available tooling (RAG, code interpreter, database connectors, image gen) produce the same business outcome?**
→ If YES → R2
→ If NO → continue to Q3

**Q3: Is the task outside LLM technology's reach due to scale, latency, hardware control, deterministic requirements, or continuous streaming?**
→ If YES → R0
→ If NO → reconsider Q1 and Q2 (no task should fail all three; this indicates the software task is not well-defined)

The LLM provides reasoning per task, allowing audit of classification logic.

### 4.5 Scope of LLM Discretion in This Rubric

The LLM exercises judgment in:
- Identifying which of the three categories a task falls into
- Recognizing whether currently-available tools (R2) suffice for replication
- Distinguishing operationally infeasible (R0) from technically possible-but-niche

The LLM does NOT exercise judgment in:
- The category boundaries themselves (defined explicitly in the prompt with worked boundary examples per Section 4.3)
- The aggregation formula (Eloundou's β, applied mechanically)
- The list of tasks (extracted in the same call, per the prompt)

This reduces but does not eliminate researcher discretion. Reliability is validated via the multi-run protocol (Section 6).

---

## Section 5 — Methodology Pipeline

### 5.1 Overview

```
[10-K Item 1 text] (321 firms, pre-shock, < 2022-11-01)
        ↓
   For iteration k ∈ {1, 2, 3}:
        ↓
        Step A: LLM extracts product tasks (8-10 per firm)
        Step B: LLM classifies each task as R0 / R1 / R2 with reasoning
                (combined with Step A in single LLM call)
        ↓
        Output: tasks_k = [(task_text, R-label, reasoning), ...]
        ↓
        Step C: Aggregate per iteration
            ρ_k = (R1_count + 0.5 × R2_count) / n_tasks × 99 + 1
        ↓
   End for
        ↓
   Step D: Average across iterations
        ρ_i = mean(ρ_1, ρ_2, ρ_3)
        ↓
   Step E: Reliability metrics (subsample only)
        ICC(3,1) computed across 15-firm anchor sample
```

### 5.2 Step A — Product Task Extraction

**Input:** 10-K Item 1 text from `text_data/10k_extracts/{TICKER}.txt`

**LLM call:** Claude Haiku 4.5 (hard-pinned: claude-haiku-4-5-20251001), temperature=0, with structured output via Pydantic schema.

**Prompt instructs the LLM to:**
- Read the Item 1 business description
- Identify 8-10 distinct **software tasks** — that is, functions the product itself performs (not customer firm activities, not industry context)
- Granularity guidance (critical for cross-firm comparability): not so atomic that they are sub-features ("toggle X on"), not so high-level that they are entire service categories ("provide CRM"). Aim for the level: "Generate personalized email content," "Process network traffic in real-time," "Calculate credit risk scores."
- Each task expressed in 1-2 sentences

**Granularity calibration:** The 8-10 task target (narrowed from 6-12) reduces granularity-induced ρ variance. While β formula is granularity-invariant per task (n_tasks divides into both numerator and denominator), edge cases at extremes (a 4-task firm vs a 14-task firm) can produce inconsistent rubric application. The 8-10 range is the empirical sweet spot for B2B software 10-Ks.

### 5.3 Step B — R-Classification

**Combined with Step A in the same LLM call** (single prompt, single response, single tool_use schema).

**For each extracted product task, the LLM classifies:**
- R-label (R0, R1, or R2) per the rubric in Section 4
- Reasoning (1-2 sentences explaining the classification, referencing the decision heuristics)

**Output schema (Pydantic):**

```python
class ProductTaskWithRubric(BaseModel):
    task: str              # 30-300 chars, software task description
    r_label: Literal["R0", "R1", "R2"]
    reasoning: str         # 30-200 chars, brief justification

class SupplyScore(BaseModel):
    ticker: str
    tasks: list[ProductTaskWithRubric]  # 8-10 items
    overall_reasoning: str              # 100-500 chars, firm-level summary
    
    # Derived fields (computed by validator):
    n_tasks: int
    r0_count: int
    r1_count: int
    r2_count: int
    raw_exposure: float       # (r1 + 0.5*r2) / n
    normalized_score: float   # raw_exposure * 99 + 1
```

### 5.4 Step C — Per-Iteration Aggregation

```python
ρ_k = (r1_count + 0.5 * r2_count) / n_tasks * 99 + 1
```

Where `k` is the iteration index (1, 2, or 3).

### 5.5 Step D — Multi-Run Averaging

```python
ρ_i = mean(ρ_1, ρ_2, ρ_3)
```

**Why averaging, not median or mode:**
- Mean is the unbiased estimator under symmetric noise
- Median is more robust to outliers but less efficient with N=3
- Mode is undefined for continuous values

If reliability concerns surface (ICC < 0.90), trimmed mean or Bayesian shrinkage estimators are robustness alternatives in Phase 6.

**Output recorded per firm:**
- ρ_i (mean)
- ρ_1, ρ_2, ρ_3 (per-iteration values, for ICC and audit)
- Per-iteration task counts and R-distributions
- Per-iteration reasoning (concatenated for audit)
- ICC(3,1) — computed at sample level, reported per firm tier

### 5.6 Step E — Reliability Validation (15-firm subsample)

The reliability subsample is the 15 anchor firms from `config/anchor_firms.yaml`.

**ICC(3,1) computation:**

For 15 firms × 3 iterations, compute single-rater intraclass correlation:

```
ICC(3,1) = (BMS - EMS) / (BMS + (k-1) * EMS)
```

Where BMS = between-subjects mean square, EMS = error mean square, k = 3 (iterations).

**Decision thresholds (Koo & Li 2016 cutoffs):**
- ICC ≥ 0.90 (excellent) → reliability acceptable, proceed with single-iteration scoring for full sample (321 firms × 1 call)
- 0.75 ≤ ICC < 0.90 (good) → marginal; proceed with multi-iteration full sample (321 × 3 = 963 calls)
- ICC < 0.75 → reject; methodology requires further refinement (rubric clarification, prompt revision, or model upgrade)

**Cost implications:**
- ICC ≥ 0.90: ~$5 supply scoring (321 calls) + $5 demand = $10
- 0.75 ≤ ICC < 0.90: ~$15 supply scoring (963 calls) + $15 demand = $30
- ICC < 0.75: pause and reconsider

---

## Section 6 — Mental Walkthrough Sanity Check (Predictions)

The mental walkthrough was originally performed during methodology design (2026-04-26 to 2026-04-27) using R-rubric mental application — the same rubric this v3 document formalizes. The predictions remain valid because they were constructed using the framework that v3 codifies.

### 6.1 Predicted ρ_β by Firm

These are **a priori predictions** made before any LLM scoring. The "Observed" column is added post-hoc as smoke results come in (commits `dbcefa9`, `d7836e5`, smoke v2/v3, 2026-04-28).

| Firm | Sector | Predicted R-distribution | Predicted ρ_β | Observed ρ_mean | Notes |
|------|--------|--------------------------|---------------|-----------------|-------|
| EGAN (eGain) | Customer engagement / chatbot | 0 R0, 8 R1, 0 R2 | **100** | **75.4** (std 6.96) | Model finds 1-2 R2 (knowledge-base RAG component) and occasional R0; pure-R1 prediction was optimistic — see revision below |
| LPSN (LivePerson) | Conversational AI | 0 R0, 8 R1, 0 R2 | **100** | not yet scored | Likely similar to EGAN — revise downward in step 6b results |
| HUBS (HubSpot) | Marketing CRM | 0 R0, 4 R1, 4 R2 | **75** | **76.0** (std 3.91) | ✓ Prediction matched empirically |
| NOW (ServiceNow) | ITSM workflow | 2 R0, 3 R1, 3 R2 | **57** | not yet scored (not in anchor yaml) | Predicted only; deferred for full sample |
| ZS (Zscaler) | Network security | 4 R0, 3 R1, 2 R2 | **45** | **42.2** (std 7.58) | ✓ Prediction matched; observed R-mix slightly different (4 R0, 1 R1, 5 R2 mean) but rho consistent |
| DDOG (Datadog) | Observability infrastructure | 3 R0, 3 R1, 2 R2 | **50** | not yet scored | Predicted only |
| SPGI (S&P Global) | Knowledge / ratings | 2 R0, 1 R1, 5 R2 | **44** | not yet scored | Predicted only |

### 6.1.1 Empirical Calibration After Smoke Test (2026-04-28)

The Step 6a smoke test on 3 archetype firms (ZS, EGAN, HUBS) revealed one prediction that requires revision:

**EGAN observed (75.4) vs. predicted (100).** The pure-R1 prediction assumed every customer-engagement task is direct LLM substitute. Empirically the model classified 1-2 tasks as R2 (knowledge-base retrieval = RAG-style integration) and occasionally 1 task as R0. This is methodologically defensible: knowledge-base lookup over the firm's own customer-data corpus is RAG (R2 by our rubric), not pure vanilla LLM (R1). The pure-R1 prediction reflected an over-simplified mental model of EGAN's product surface.

**Revision rule:** Predictions in this section are **descriptive heuristics** for the author's mental walkthrough, not normative targets. Empirical scores from Step 6a/6b supersede a priori predictions. The methodology defense in the thesis writeup will state: "Predicted ranges in Section 6.1 represent the author's pre-execution intuitions; observed ranges in Section 6.1 (post-smoke) reflect what the LLM-as-classifier produces from each firm's actual 10-K Item 1 text. Where the two diverge, the empirical score takes precedence."

**Revised predicted ρ ranges (for advisory comparison only, not decision gates):**
- EGAN: [70, 90] (was [85, 100])
- LPSN: [70, 90] (parallel revision; same product category)
- HUBS: [65, 85] (unchanged — empirically validated)
- ZS: [30, 55] (was [40, 55] — slightly widened on the low end based on iteration variance)
- DDOG, SPGI, NOW: predictions stand pending Step 6b

These ranges are now used only for soft anchor-pattern checks in `08_score_supply_rho.py` console output. They do NOT constitute a fail condition; ICC reliability (Section 5.6) is the authoritative decision gate for proceeding to full-sample scoring.


### 6.2 Why R-Rubric Differentiates Where ONET Matching Did Not

ZS prediction breakdown:
- "Process 250B packets/sec" → **R0** (latency + scale outside LLM scope)
- "Detect threats in real-time streams" → **R0** (latency)
- "Route traffic globally with sub-ms" → **R0** (latency + distributed scale)
- "Inline content inspection at line rate" → **R0** (latency + scale)
- "Sandbox detonation reports" → **R1** (text generation, LLM can write)
- "DLP classification using OCR" → **R1** (text classification, LLM can do)
- "Cloud app categorization" → **R2** (LLM + reference data)
- "Auto-remediate misconfigurations" → **R2** (LLM + cloud APIs)
- "Generate access policies" → **R1** (text generation, LLM can do)

→ 4 R0, 3 R1, 2 R2 → ρ = (3 + 0.5×2)/9 × 99 + 1 = **45**

EGAN prediction breakdown:
- "Provide chatbot responses" → **R1** (canonical LLM task)
- "Search KB for relevant articles" → **R1** (LLM with retrieval; or even basic LLM)
- "Suggest knowledge articles to agents" → **R1** (LLM ranking)
- "Categorize incoming cases" → **R1** (text classification)
- "Route cases to departments" → **R1** (text classification)
- "Self-service portal content" → **R1** (text generation)
- "Sentiment analysis" → **R1** (canonical LLM task)
- "Translate customer messages" → **R1** (canonical LLM task)

→ 0 R0, 8 R1, 0 R2 → ρ = (8 + 0)/8 × 99 + 1 = **100**

**The R-rubric naturally separates ZS (latency-protected) from EGAN (pure LLM competition).** The ONET corpus could not because both firms' customer-workers perform cognitive tasks. The R-rubric measures the software function itself, not the worker function it serves.

### 6.3 What the Walkthrough Confirms

**Pattern 1:** Pure LLM competitors at the top (EGAN, LPSN at 100). Direct chatbot competition with low δ.

**Pattern 2:** Infrastructure with latency/scale at the bottom of the software tier (ZS at 45, DDOG at 50). R0 protection from operational requirements.

**Pattern 3:** Mid-range firms in the middle (HUBS at 75, NOW at 57). Mix of R1 and R2 tasks.

**Pattern 4:** Knowledge tier intermediate (SPGI at 44). Mix of R0 (deterministic computation, real-time feeds) and R2 (data + LLM).

**Pattern 5:** ρ × (1-δ) effective threat reveals different ranking than ρ alone. Infrastructure firms protected by switching cost / data moats; pure LLM competitors highly threatened.

### 6.4 What the Walkthrough Does Not Confirm

**Empirical validation comes from:**
- Anchor smoke test (3 firms × 3 iterations = 9 calls, ~$0.06): real LLM extraction and R-classification with multi-run
- Anchor full test (15 firms × 3 iterations = 45 calls, ~$0.54): ICC computation and broader pattern check
- Full supply scoring (321 firms × 1 or 3 calls = $5-15): the actual treatment data

The mental walkthrough informs design; it does not constitute validation.

---

## Section 7 — Methodological Foundations

### 7.1 Eloundou et al. (2024) — Inspiration with Acknowledged Limitations

**Citation:** Eloundou, T., Manning, S., Mishkin, P., Rock, D. (2024). "GPTs are GPTs: Labor market impact potential of LLMs." *Science* 384(6702), 1306-1308.

**Methodology of the original paper (verified via web search 2026-04-27):**

Eloundou's labels were produced through a dual-track annotation:
1. **Human annotators:** 5 named individuals (Muhammad Ahmed Saeed, Bongane Zitha, Merve Özen Şenen, J.J., Peter Hoeschele), described as "OpenAI workers familiar with the rubric"
2. **GPT-4 annotator:** Prompt "tuned for agreement with a sample of labels from the authors"

The rubrics for human and GPT-4 annotators were "slightly adjusted" between the two tracks. Eloundou explicitly states: "Although we use multiple annotation sources, none is considered the definitive ground truth." The 50% time-reduction threshold for E1 classification is acknowledged as "arbitrary, [...] selected for ease of interpretation by annotators."

**What we use:**
- The conceptual structure (E0/E1/E2 categories) — **adapted to R0/R1/R2** for software-task domain
- The α/β/γ aggregation formulas (β as main spec)
- The rubric definitions as a starting point — **with software-specific examples and decision heuristics added**

**What we do NOT use:**
- The 19,265-task corpus directly for primary scoring (we use Eloundou's labels as Phase 6 robustness comparator only)
- Their occupation-level aggregation (we aggregate at firm-product level)
- Their employment-share weighting (we use product-task counts)

**Why we don't inherit labels directly (defense statement):**

> "Our rubric structure (R0/R1/R2) parallels Eloundou et al. (2024)'s E0/E1/E2 by design, applied to software tasks rather than worker tasks. We retain Eloundou's β aggregation formula. We do not directly inherit Eloundou's published task labels because their corpus measures worker-level replicability, which empirically does not differentiate B2B software firms whose customers are uniformly cognitive workers (see Section 3.2). This is consistent with Eloundou's own acknowledged limitations: their annotator pool is not occupationally diverse, and they explicitly state that no annotation source is the definitive ground truth. We adopt Eloundou's framework as **inspiration for rubric structure**, while shifting the unit of analysis to the software-task level — analogous to Eisfeldt-Schubert-Zhang (2023)'s extension of the Eloundou framework with software-specific rubric detail."

### 7.2 Eisfeldt, Schubert & Zhang (2023) — Methodological Precedent for Rubric Adaptation

**Citation:** Eisfeldt, A.L., Schubert, G., Zhang, M.B. (2023). "Generative AI and Firm Values." NBER Working Paper 31222 (May 2023, last revised Jan 2026, forthcoming Journal of Finance). SSRN: 4440717. Slide deck author list adds Taska, B. (SkyHive) for the firm-occupation matching component.

**Verified citations (via web search 2026-04-27):**

From the Eisfeldt-Schubert-Zhang research presentation slides (URL: https://www.miaobenzhang.com/GenAI_Slides.pdf, retrieved 2026-04-27):

> "Important clarification: this is **not** asking the LLM about its capabilities. **No special LLM wisdom is required.** We provide detailed rubrics and the LLM just maps a statement into the pre-defined matrix of capabilities."

This is the cardinal methodological principle: the LLM is a **classifier**, not a self-assessor. We adopt the same principle in our prompt design (Section 9 and the system-prompt drafted in `prompts/supply_rho_system.txt`).

From the same slide deck, Eisfeldt-Schubert-Zhang's verbatim E1 rubric definition (which we adapt to R1 in our work):

> "Label tasks E1 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground alone can reduce the time it takes to complete the task with equivalent quality by at least half. This includes tasks that can be reduced to:
>   – Writing and transforming text and code according to complex instructions
>   – Providing edits to existing text or code following specifications
>   – Writing code that can help perform a task that used to be done by hand
>   – Translating text between languages
>   – Summarizing medium-length documents
>   – Providing feedback on documents
>   – Answering questions about a document
>   – Generating questions a user might want to ask about a document
>   – Writing questions for an interview or assessment
>   – Writing and responding to emails, including ones that involve refuting information or engaging in a negotiation (but only if the negotiation is via written correspondence)
>   – Maintain records of written data
>   – Prepare training materials based on general knowledge
>   – Inform anyone of any information via any written or spoken medium."

Eisfeldt-Schubert-Zhang submit `rubric + 2 examples + (occupation, task statement) pair` to GPT-3.5 Turbo and obtain `score (0, 0.5, or 1) + explanation`. This per-task design with explicit reasoning trace is the structure we partially adopt (we use combined extraction+classification per firm rather than per task, see Section 6.2 for noise budget).

From the Eisfeldt-Schubert-Zhang paper (Internet Appendix C, classification procedure documentation):

> "Note that this classification method should not be interpreted as requiring ChatGPT to have any kind of correct 'knowledge of its own capabilities.' Instead, the categories for what kinds of capabilities state-of-the-art LLMs have are given by us to the model, as they were pre-defined by researchers in collaboration with OpenAI in Eloundou et al. (2023). That is, the right way to think about the role of ChatGPT here is as a research assistant mapping task statements into existing categories, which relies on its ability to interpret language, understand occupational contexts, and reason about which known LLM capabilities would [apply]."

This is the **exact methodological framing we adopt**: a detailed rubric as input, LLM as a classifier mapping into pre-defined categories. Our extension is the rubric domain (software tasks) rather than the rubric mechanism.

**What we extend:**
- LLM-as-classifier methodology with detailed rubric (their explicit framing above)
- Use of LLM as research assistant for task labeling
- Multi-run reliability validation (their Internet Appendix details GPT consistency across runs)
- Three-step aggregation: task-level → ... → firm-level

**Where we differ:**
- Eisfeldt aggregates to firm-WORKFORCE level (occupational employment shares from LinkedIn data via Revelio)
- We aggregate to firm-PRODUCT level (product task counts from 10-K Item 1)
- Eisfeldt's rubric remains at worker-task level; ours adapts to software-task level

**Conceptual relationship:**

| | Eisfeldt-Schubert-Zhang (2023) | This thesis |
|---|---|---|
| Aggregation target | Firm workforce | Firm product |
| Input data | LinkedIn occupational employment shares | 10-K Item 1 product description |
| Rubric | Eloundou's E1/E2 with software-related examples added | R0/R1/R2 reframed for software-task domain |
| Question | "How exposed is this firm's labor force to LLM substitution?" | "How exposed is this firm's product to LLM substitution?" |
| Position in value chain | One level: worker exposure | One level up: software exposure (workers' tools) |

**Defense statement:**

> "We extend Eisfeldt-Schubert-Zhang (2023)'s firm-workforce-level methodology to the firm-product level. Both studies use detailed-rubric LLM classification with multi-run reliability validation. Eisfeldt explicitly characterize their approach as the LLM serving as 'a research assistant mapping task statements into existing categories' (Internet Appendix C). The rubric adaptation in our work parallels their elaboration of Eloundou's E1 definition with software-specific examples — we apply the same pattern, adapted to our unit of analysis (software tasks instead of worker tasks). The conceptual extension follows Acemoglu and Restrepo (2018, 2022)'s task substitution framework applied one level up the value chain: B2B software products are themselves a form of capital that automates worker tasks; LLMs threaten that capital where the underlying software function is LLM-replicable."

### 7.3 Labaschin, Eloundou, Manning, Mishkin & Rock (2025) — Firm-Workforce Aggregation Precedent

**Citation:** Labaschin, B., Eloundou, T., Manning, S., Mishkin, P., Rock, D. (2025). "Extending GPTs Are GPTs to Firms." AEA Papers and Proceedings 115, 51–55. DOI: 10.1257/pandp.20251045.

**What this paper does:** Aggregates Eloundou et al. (2024)'s pre-computed task-level GPT-4 exposure scores (E0/E1/E2) to firm level using employment-weighted averages. They use Revelio Labs employment counts by SOC code:

$$E_{it} = \sum_j \frac{FTE_{ijt}}{FTE_{it}} \times E_j$$

where firms are indexed by $i$, occupations by $j$, $t$ is time (June 2024), and $E_j$ is the occupation-level β score (E1 + 0.5×E2) inherited from Eloundou et al. (2024).

**What this paper does NOT do:** It does **not** generate new LLM classifications. It reuses Eloundou's published labels and aggregates them. There is no new prompt, no new rubric, no new LLM call.

**Why this is relevant to our methodology:** This paper is the canonical example of the **worker-channel firm aggregation** approach that our thesis explicitly does NOT replicate. Labaschin et al. measure ρ_firm via the worker-substitution channel S1: firm exposure equals the employment-weighted average of how much each worker's tasks are LLM-replicable.

**Our positioning:** Our work measures ρ_firm via the **product-substitution channel S2 / joint channel S3**: firm exposure equals the average LLM-replicability of the software product's own functional tasks. The two measures are **complementary**, not competing:

- S1 (Labaschin 2025): "How exposed is this firm's workforce to LLMs?"
- S2/S3 (this thesis): "How exposed is this firm's product to LLMs?"

**Phase 6 robustness opportunity:** A natural future-work angle (out of scope for this thesis) is to compute both ρ_S1 (Labaschin-style worker aggregation) and ρ_S2/S3 (our product aggregation) for our 321 firms, and examine their correlation. We expect partial overlap (firms whose products serve highly replicable workers will likely have replicable products too) but meaningful separation (a security infrastructure firm like ZS may have low ρ_S2/S3 even if its customers are high-ρ_S1 cognitive workers).

**Defense statement:**

> "Our work is conceptually distinct from Labaschin et al. (2025)'s firm-workforce aggregation. Labaschin et al. measure firm exposure via the LLM-replicability of the firm's **employees'** tasks (worker channel, S1). We measure firm exposure via the LLM-replicability of the firm's **product's** tasks (product channel, S2/S3). The two measures are complementary; they capture different threat channels in the unified Worker+Software framework presented in Section 2."

### 7.4 Acemoglu & Restrepo Task Substitution Framework

**Citations:**
- Acemoglu, D., Restrepo, P. (2018). "The race between man and machine." American Economic Review 108(6), 1488-1542.
- Acemoglu, D., Autor, D., Hazell, J., Restrepo, P. (2022). "Artificial intelligence and jobs." Journal of Labor Economics 40(S1), S293-S340.

**Core framework:** Tasks can be performed by labor or by capital (machines). Technological change affects which tasks become more efficient under each mode. New technologies create reallocation, displacement, and reinstatement effects.

**Our extension:** The framework historically applies to the worker-vs-machine boundary. We apply it to the **software-vs-LLM** boundary. B2B software products are a form of capital that automated worker tasks in the 1990s-2010s. LLMs are a new form of capital that may automate the tasks the software products themselves perform—displacing software, not just labor.

**This framing is the thesis's primary conceptual contribution.** The methodology operationalizes it via the R-rubric applied to software tasks.

### 7.5 Koo & Li (2016) — Reliability Thresholds

**Citation:** Koo, T.K., Li, M.Y. (2016). "A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research." *Journal of Chiropractic Medicine* 15(2), 155-163.

**What we use:** ICC interpretation thresholds for reliability assessment (Section 5.6).

The standard cutoffs:
- ICC < 0.50: poor reliability
- 0.50 ≤ ICC < 0.75: moderate reliability
- 0.75 ≤ ICC < 0.90: good reliability
- ICC ≥ 0.90: excellent reliability

These cutoffs guide our decision protocol for whether single-run or multi-run scoring is required for the full 321-firm sample.

### 7.6 Capability Dynamics and Instrument Validity

A natural objection to LLM-based scoring is that the measurement instrument is itself the technology being measured, and that technology improves over time. We address this through three layers of defense, presented here as a unified methodology statement and operationalized in Phase 6 as an empirical robustness check.

**The two distinct problems.**

The temporal-capability concern conflates two distinct issues that deserve separate treatment:

1. **Time-varying treatment intensity (econometric).** LLM capabilities have improved monotonically since the Q4 2022 ChatGPT shock. A firm whose product has ρ = 70 against the GPT-4 Q4 2022 frontier may face a higher effective replicability against the GPT-5 / Claude Opus 4.7 frontier of Q2 2026. This is a feature of the world, not a measurement artifact.

2. **Instrument-stability (measurement validity).** We score firms using `claude-haiku-4-5-20251001`. If we re-score the same firms with a different LLM (Claude Opus 4.7, GPT-4.1, Gemini 2.5), we may obtain different ρ values. The question is whether our reported ρ_i represents a stable construct or an instrument-dependent artifact.

These problems require separate methodological responses.

**Response to Problem 1: Static treatment, dynamic outcome.**

We adopt Eloundou et al. (2024)'s single-snapshot approach: ρ_i is measured once at a fixed instrument vintage (Claude Haiku 4.5, prompted to evaluate against ChatGPT-class capability as of Q4 2022). The dynamic effects of subsequent capability improvements appear in the event-study coefficients γ_h for each post-shock period h, not in a time-varying ρ_i,t. The treatment intensity at the moment of the shock is the relevant exposure; subsequent capability growth multiplies that exposure into outcomes through firm-specific dynamics that the regression captures.

This approach is the dominant convention in the labor-economics-of-AI literature (Eloundou 2024, Eisfeldt-Schubert-Zhang 2023, Felten et al. 2023). It is replicable, transparent, and avoids the panel-construction explosion that time-varying ρ_i,t would entail.

**Response to Problem 2 (Layer 1): Frozen instrument vintage.**

The scoring run is conducted at a single point in time using a hard-pinned model identifier (`claude-haiku-4-5-20251001`). The system prompt is fixed; its SHA-256 hash is recorded in the run logs. Any future researcher with the same model ID and prompt hash can replicate our scoring procedure exactly (subject to the model's deterministic-mode behavior at temperature 0). This is the standard replicability discipline for LLM-based research instruments.

**Response to Problem 2 (Layer 2): Capability-anchored prompt.**

The prompt explicitly instructs the LLM to evaluate against ChatGPT-class Q4 2022 capability, not its own present capability. This is the cardinal methodological cue from Eisfeldt-Schubert-Zhang (Section 7.2): the LLM's role is "research assistant mapping task statements into existing categories," not capability self-assessor. We further reinforce this with the Section 1 disclaimer in `supply_rho_system.txt`: "This is NOT a self-assessment of your own capabilities."

**Response to Problem 2 (Layer 3): Cross-model robustness check (Phase 6).**

In Phase 6, we re-score the 14-15 firm anchor sample using a second LLM (provisionally GPT-4.1 via OpenAI API or Gemini 2.5 via Google AI Studio). Cost: ~$0.15-0.30 for the anchor subset. We compute across-model intraclass correlation:

- ICC_across ≥ 0.75 → measurement instrument-invariant; primary results stand.
- ICC_across < 0.75 → instrument-dependent; report as a methodological caveat, but main results remain valid because instrument vintage is fixed for our research question.

The cross-model ICC test is published in Phase 6 robustness section regardless of outcome; transparency on instrument validity is itself a methodological contribution.

**Defense paragraph for thesis writeup.**

> Our supply-side replicability score ρ is measured at a fixed instrument vintage: Claude Haiku 4.5 (model ID `claude-haiku-4-5-20251001`), prompted to evaluate against ChatGPT-class capability as of Q4 2022. The score reflects the firm's product exposure to LLMs at the moment of the ChatGPT shock, not the present capability frontier. Subsequent capability improvements affect outcomes through the event-study dynamics, not through the baseline ρ_i. To assess instrument validity, we re-score the 14-firm anchor sample with [second model] and report across-model ICC of [X.XX], indicating [strong/moderate] instrument invariance for relative rankings. Absolute ρ levels are calibrated to the primary instrument vintage.

---

## Section 8 — Phase 4 Implementation Plan

This methodology document drives the following implementation steps. Each step is a separate Claude Code session (compaction prevention).

### 8.1 Steps Already Completed (commits)

- **9c06cd9:** Pydantic schemas (SupplyScore, DemandScore) — DemandScore preserved, SupplyScore will be rewritten in Step 4 below.
- **c3b1681:** llm_client.py — infrastructure, schema-agnostic, kept.
- **33b1c4e:** Cache verification fix — kept.
- **9e526b6:** sentence-transformers dependency — kept (for Phase 6 robustness alternative).
- **b662006:** Eloundou ONET corpus build script and parquet — kept (for Phase 6 robustness alternative).
- **e807f98:** task_matching.py — embedding match utility — kept (for Phase 6 robustness alternative).

### 8.2 Steps Pending

**Step 4 (next): Schema rewrite + R-rubric prompt rewrite**

This step has two sub-priorities flagged by Claude Code in v2 review:
- **Priority A:** R0/R2 boundary worked examples per Section 4.3 (real-time alerting → R0 vs hourly batch → R2, etc.). These are critical for inter-run reliability.
- **Priority B:** 8-10 task granularity guidance with consistent-granularity worked examples in prompt.

Tasks:
- Rewrite `prompts/supply_rho_system.txt`:
  - Section 1: Methodological context (brief reference to thesis purpose)
  - Section 2: Software-task definition and granularity guidance (8-10 task target)
  - Section 3: Full R0/R1/R2 rubric with definitions, decision heuristics, and **boundary worked examples** (Section 4.3)
  - Section 4: 8-12 worked examples of typical task classification (mix of R0, R1, R2 across different industries)
  - Section 5: Output format and reasoning requirements
  - Word count target: ~5000-6000 tokens (above Haiku 4.5's 4096-token cache minimum)
- Rewrite `scripts/utils/schemas.py` SupplyScore:
  - `tasks: list[ProductTaskWithRubric]` where each item has `task`, `r_label`, `reasoning`
  - Validators: 8-10 tasks (relaxed to 6-12 for edge tolerance), length constraints, R-label literal type
  - Computed fields: r0/r1/r2 counts, raw_exposure, normalized_score
- Self-tests pass

**Step 5: 08 script rewrite for multi-run scoring**
- Rewrite `scripts/08_score_supply_rho.py`:
  - Add multi-run mode: `--iterations 3` flag (default 1, anchor uses 3)
  - Per-firm output: separate row per iteration in scratch CSV, then aggregated row in final CSV
  - Final CSV columns: ticker, n_tasks_iter1/2/3, r0_count_iter1/2/3, r1_count, r2_count, rho_iter1/2/3, rho_mean, rho_std, tasks_json_iter1/2/3 (audit), reasoning_iter1/2/3, llm_call_cost_total, scored_at
  - Anchor pattern checks revised for R-rubric
- Verification tests pass

**Step 6: Anchor smoke + reliability test**
- Smoke: 3 firms × 3 iterations = 9 calls, ~$0.06
- Anchor full + reliability: 15 firms × 3 iterations = 45 calls, ~$0.54
- ICC(3,1) computation script: `scripts/utils/icc.py`
- Decision gate per Section 5.6

**Step 7: Full supply scoring**
- If ICC ≥ 0.90: 321 firms × 1 call = $5
- If 0.75 ≤ ICC < 0.90: 321 firms × 3 calls = $15
- Output: `data/processed/supply_rho.csv`

**Step 8: Demand pipeline**
- Demand prompt minor refinement (δ_switch scope to include scale/latency-related switching cost explicitly)
- Demand schema preserved
- Anchor test
- Full demand scoring: 321 firms × 1 or 3 calls, $5-15

### 8.3 Cost Summary

| Phase 4 step | Calls | Cost |
|--------------|-------|------|
| Smoke (3 × 3) | 9 | $0.06 |
| Anchor + reliability (15 × 3) | 45 | $0.54 |
| Full supply (321 × 1 or 3) | 321-963 | $5-15 |
| Full demand (321 × 1 or 3) | 321-963 | $5-15 |
| **Total Phase 4** | **696-1,932** | **$10.60-30.60** |

Within original Phase 4 budget envelope (~$30).

---

## Section 9 — Critical Guardrails for Future Claude Instances

**This section is permanent rules. Any Claude instance reading this document must respect these without modification.**

### Rule 1: R-Rubric Stability
The R0/R1/R2 rubric definitions in Section 4 are stable. Modifications (re-defining what constitutes R0 vs R2, adding R3, changing decision heuristics) require explicit author authorization documented in Section 11 history.

### Rule 2: Aggregation Formula Fidelity
The β formula is `(R1_count + 0.5 × R2_count) / n_tasks × 99 + 1`. This mirrors Eloundou's β scaled to [1, 100]. Modifications to the weights require explicit author authorization.

### Rule 3: Multi-Run Protocol Mandatory for Anchor
Anchor smoke and anchor test always use 3 iterations per firm. This is non-negotiable. ICC(3,1) must be computed and reported for the 15-firm anchor sample before proceeding to full scoring.

### Rule 4: Scale/Latency Belong to R0 (when fundamental) or δ (when adoption barrier)
If a software task is operationally outside LLM scope (e.g., 250B packets/sec), it is R0 — measured in ρ. If a task is functionally LLM-replicable but the customer cannot adopt due to integration depth or scale-related switching cost, this is δ_switch — measured separately. The two channels are conceptually distinct and must not be conflated.

### Rule 5: Demand Methodology Preserved
The Phase 3 demand methodology (δ_switch + δ_error + δ_data, equal weighting) is preserved. Demand pivot is not on the table. Minor scope refinements (e.g., explicit inclusion of technical switching cost in δ_switch) are acceptable.

### Rule 6: Pre-Shock 10-K Item 1 Only
Source text is `text_data/10k_extracts/{TICKER}.txt` filtered to Item 1 only, filing date < 2022-11-01. Post-shock 10-Ks contain firm adaptation language and are contaminated. Item 1A (Risk Factors) and Item 7 (MD&A) are excluded.

### Rule 7: Model Hard Pin
`claude-haiku-4-5-20251001`. Never Sonnet, Opus, or any other model for the supply scoring pipeline. The pin is for cost control and reproducibility.

### Rule 8: One Spec Per Claude Code Session
Compaction in Claude Code can corrupt multi-step specs. Each implementation spec must be a single, focused work unit (file rewrite, scoring run, or analysis), with verification tests and a single commit at the end.

### Rule 9: No Return to ONET Embedding for Primary Scoring
The ONET embedding match approach (v1 methodology) is empirically ruled out for primary scoring (Section 3.2). It is preserved as Phase 6 robustness alternative only. Any proposal to revive ONET embedding match as primary methodology must be refused with reference to Section 3.2 evidence.

### Rule 10: This Document Is Authoritative
If this document and another conversation, summary, or memory contradict, this document wins. If the author updates the methodology, this document must be updated explicitly with a dated revision note in Section 11.

### Rule 11: Anchor-Pull Avoidance
Anchor firms (15 in `config/anchor_firms.yaml`) are used for pattern checks and reliability validation, not for prompt anchoring. The supply prompt MUST NOT contain numeric target scores for any anchor firm. The prompt's anchor firm content (if any) is for extraction and classification examples, never for outcome calibration.

---

## Section 10 — Open Questions and Future Work

**Q1: Multi-run cost vs reliability trade-off.**
If ICC ≥ 0.90 on anchor sample, single-run full sample is acceptable. If lower, multi-run full sample required (~$15 vs $5). Phase 4 budget accommodates either.

**Q2: Phase 6 ONET embedding comparator.**
Re-run scoring using v1 methodology (ONET embedding match) on full sample. Compare distributions, rankings, DiD coefficients. Expected: similar relative rankings for clear cases (EGAN high, knowledge tier middle), divergence for infrastructure firms (ZS, DDOG higher in ONET-based ρ due to corpus limitations). This is reported as alternative specification robustness.

**Q3: Eisfeldt-style core/supplemental decomposition.**
ONET task_type metadata (Core vs Supplemental) is available from v1 work. Phase 6 robustness can decompose ρ_alt (ONET-based) into ρ_core_alt and ρ_supp_alt. Less directly applicable to R-rubric (no Core/Supplemental in our framework), but provides additional Eisfeldt-style cross-validation.

**Q4: Embedding model upgrade for Phase 6.**
all-MiniLM-L6-v2 used in v1 corpus. Alternatives (text-embedding-3-small, all-mpnet-base-v2) tested as robustness in Phase 6.

**Q5: Custom demand methodology pivot (deferred indefinitely).**
The demand side may eventually warrant a measurement framework analogous to supply. Currently held in Phase 3 form. Author indicated demand methodology revisit will happen separately, after supply pipeline is fully validated.

**Q6: Worker vs Software task autonomous/augmenting decomposition (deferred).**
A future extension can label each product task as "autonomous" (software does it without worker, S2/S3 scenarios) vs "augmenting" (software helps worker, S1 scenario from product's perspective). This adds analytical depth but is not in Phase 4 scope.

**Q7: Measurement error and attenuation bias reporting.**
Multi-run averaging reduces but does not eliminate single-firm classification noise. Expected residual noise ~7.5 points. This must be reported in thesis methodology chapter as standard measurement error producing **attenuation bias** in DiD interaction coefficients — true effects may be larger than estimated. This conservative framing strengthens defense.

---

## Section 11 — Document History

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2026-04-25 | v0.1 (PHASE4_METHODOLOGY_PIVOT.md) | Initial pivot doc — pure embedding match | Web Claude |
| 2026-04-27 | v1.0 (PHASE4_METHODOLOGY_v1.md) | ONET embedding match with worker-task identification, Eisfeldt extension | Web Claude |
| 2026-04-27 | v2.0 (PHASE4_METHODOLOGY_v2.md) | Software-adapted R-rubric with multi-run averaging, after empirical sanity test ruled out v1 | Web Claude |
| 2026-04-27 | v3.0 (this) | Added Eloundou self-acknowledged limitations to Section 3.2 and 7.1; verified Eisfeldt citation source (slides + Internet Appendix C); added Section 4.3 (R0/R2 boundary worked examples); narrowed task target from 6-12 to 8-10; added Q7 (measurement error reporting) | Web Claude |
| 2026-04-27 | v3.1 (current) | **Citation fix:** Eisfeldt et al. → Eisfeldt-Schubert-Zhang (2023, NBER WP 31222) — earlier "(2024)" was incorrect publication year. Added verbatim E1 rubric quote from Eisfeldt-Schubert-Zhang slides into Section 7.2. **New Section 7.3:** Labaschin et al. (2025) AEA P&P firm-workforce aggregation precedent — earlier session mistakenly conflated this Eloundou-team firm extension paper with Eisfeldt's separate paper. Section now explicitly distinguishes worker-channel S1 (Labaschin) from product-channel S2/S3 (this thesis). Renumbered 7.3→7.4 and 7.4→7.5. | Web Claude |
| 2026-04-28 | v3.2 (current) | **Schema refinement post-smoke (commits `dbcefa9`, `d7836e5`):** Per-task reasoning max_length 200→350 (model output ~250 char natural). Removed model-reported counts from `SupplyScore` — `r0_count/r1_count/r2_count/n_tasks/raw_exposure/normalized_score` now computed deterministically from `tasks` labels via `compute_aggregates()` helper. Eliminates self-consistency failure mode observed in ZS smoke v2 (model produced inconsistent counts across retries). Schema now requires only `ticker/tasks/overall_reasoning` from model. **Section 6.1 empirical update:** Smoke test (ZS, EGAN, HUBS, 3 iter each) produced rho_mean = (42.2, 75.4, 76.0). EGAN [85,100] → [70,90] revised (pure-R1 prediction was optimistic; KB retrieval is naturally R2). Predicted ranges now advisory, not decision gates — ICC (Section 5.6) is authoritative. | Web Claude |
| 2026-04-28 | v3.3 (current) | **New Section 7.6 (Capability Dynamics and Instrument Validity):** Three-layer defense added — (1) static treatment + dynamic outcome via event-study (Eloundou convention), (2) frozen instrument vintage with hash-recorded prompt + hardpinned model ID, (3) cross-model ICC robustness check planned for Phase 6 (~$0.15-0.30, second LLM TBD between GPT-4.1 / Gemini 2.5). Adds defense paragraph for thesis writeup. **Planned prompt revision (next commit):** (a) Section 3 sectoral priors removed (anchor-pull mitigation), (b) Section 7 anchor numeric scores removed, distributions retained (R-structure shown without prediction targets), (c) Section 2 vintage framing tweak ("ChatGPT-class Q4 2022" replacing anachronistic "GPT-4 / Claude-Sonnet"). Phase 6 robustness items recorded: inter-model triangulation, R0 multi-axis decomposition, double-pass extract+classify, granularity drift transparency table. | Web Claude |

---

## Section 12 — Why v3 Is the Final Version (For Now)

Three approaches were tried. The first (LLM-driven direct classification with Eloundou rubric) failed for determinism. The second (ONET embedding match with worker-task identification) failed for unit-of-analysis mismatch. The third (software-adapted rubric with multi-run averaging) is empirically untested but conceptually correct: it measures what we want to measure (software-level replicability) using a methodology with direct precedent in Eisfeldt-Schubert-Zhang (2023) and noise control via multi-run averaging.

If anchor smoke testing reveals further issues — e.g., ICC < 0.75 even with multi-run, or rubric ambiguity producing systematic misclassifications — the methodology will require another revision documented in Section 11. Until then, v3 is the operating framework.

The thesis methodology chapter will document this evolution explicitly. The journey from v0 → v1 → v2 → v3 is itself a contribution: it demonstrates that measuring software-level LLM exposure required us to identify and reject two natural-looking approaches before arriving at the empirically valid one. Documenting these dead ends is good scientific practice and strengthens defense.

**End of Document.**

This document is the authoritative reference for Phase 4 methodology. Save to repo root, upload to Project knowledge, include in every new Claude Code session for verification.
