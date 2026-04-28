# CLAUDE.md — AI Assistant Instructions

This file tells Claude (or any AI assistant) exactly what this repository is and how to work within it correctly. Read it fully before touching any code or data.

Last substantive update: 2026-04-28 (Phase 4 R-rubric pivot, smoke validation complete).

---

## What This Repo Is

**Master's thesis empirical pipeline** for:
> *"Generative AI as a Task Shock: Product-Level LLM Substitution in B2B Software Markets"*
> Hakan Zeki Gülmez · TUM School of Management · Supervisor: Prof. Dr. Helmut Farbmacher · Submission: October 2026

**Research question:** Do B2B software firms whose products are more replicable by LLMs experience worse revenue growth after the ChatGPT shock (2022Q4)? And does demand-side friction moderate that effect?

**Sample scope — three-tier universe (321 firms):**
- **Primary software** (256 firms) — SIC 7370–7379 B2B software and data processing; the treatment group
- **Primary knowledge** (35 firms) — knowledge-intensive services (consulting, education, publishing, info services); test whether any post-2022 shock is AI-specific to software
- **Placebo** (30 firms) — energy, manufacturing, payments, pharma; falsification check with no plausible AI replicability mechanism

---

## Estimating Equations

**Main specification (supply side only):**
```
ln(Revenue_it) = α_i + δ_t + β · (Post_t × ρ_i) + ε_it
```

**Extended specification (supply × demand interaction) — triple interaction form:**
```
ln(Revenue_it) = α_i + δ_t + β1·(ρ_i × Post_t) + β2·(δ_i × Post_t)
                 + β3·(ρ_i × δ_i × Post_t) + ε_it
```

Where:
- `α_i` = firm fixed effects
- `δ_t` = time (quarter) fixed effects
- `ρ_i` = supply-side LLM replicability score ∈ [1, 100] (pre-shock, cross-sectional)
- `δ_i` = demand-side friction score ∈ [0, 1] (pre-shock, cross-sectional)
- `Post_t = 1` for `period_end ≥ 2022-10-01` (2022Q4 onward)

Interpretation:
- `β1 < 0` would confirm H1: higher LLM replicability → larger negative revenue effect
- `β3 > 0` would confirm H2: higher demand friction moderates (buffers) the negative effect

Inference: wild cluster bootstrap (WCB), clustered at firm level. R package: `fwildclusterboot`.

Alternative "effective exposure" specification and full robustness battery documented in `docs/phase6_notes.md`.

---

## The Two Treatment Variables

### Supply Side: ρ_i (LLM Replicability Score)

**Prompt:** `prompts/supply_rho_system.txt` (de-anchored 2026-04-28, commit c1acabf)
**Script:** `scripts/08_score_supply_rho.py` (multi-iter scoring, commit b485cf1)
**Output:** `data/processed/supply_rho.csv` (renamed from `lit_scores.csv` in Step 5)

Measures the fraction of a firm's product task bundle that an LLM can replicate at near-zero marginal cost. Applied at the product level — one level up from Eloundou et al. (2024) who classify worker tasks. The R-rubric is software-adapted from Eisfeldt-Schubert-Zhang (2023) and follows Acemoglu-Restrepo (2018, 2022) task-substitution logic.

**Framework (R0 / R1 / R2):**
- **R1** — direct LLM substitute (vanilla LLM via chat/API): text generation, classification, summarization, translation, drafting communications, structured extraction
- **R2** — LLM + standard tools: RAG over document corpus, SQL queries + LLM synthesis, code execution sandbox, simple multi-step workflows
- **R0** — outside LLM scope: hard latency constraints, hardware control, massive scale economics, deterministic computation, deeply proprietary integration, regulated final attestation

**Score formula (pure Eloundou β aggregation, retained from Phase 3):**
```
raw_exposure     = (R1_count + 0.5 × R2_count) / n_tasks
normalized_score = round(raw_exposure × 99 + 1, 1)   ∈ [1, 100]
```

This is the Eloundou et al. (2024) β coefficient applied verbatim, scaled to a 1–100 interval. The score emerges deterministically from `compute_aggregates()` which derives counts from validated task labels — the model is never asked to self-report counts. This eliminated the self-consistency failure mode observed in Step 6a smoke v2 (commit `d7836e5`).

**Methodological constraint (prompt-level):** The prompt explicitly forbids the model from targeting any expected score. Five anchor archetypes (EGAN, ZS, HUBS, SPGI, DDOG) embedded in the prompt illustrate task decomposition patterns (which tasks belong to R0/R1/R2 for a given product type) but are NOT numeric calibration targets. Numeric ρ scores were removed from anchor examples in commit `c1acabf` after smoke evidence showed residual anchor-pull leakage was empirically zero but methodologically risky.

**Multi-iter reliability validation:** The `--iterations N` flag (default 1, max 3) runs N independent scorings per firm. Step 6b will use 3 iterations on the 14-15 firm anchor sample to compute ICC(3,1). Decision gate per `docs/PHASE4_METHODOLOGY_v3.md` Section 5.6: ICC ≥ 0.90 → single-run full sample (321 × 1 = ~$3); 0.75-0.90 → multi-run (321 × 3 = ~$10); < 0.75 → reject and revise prompt.

**Methodology document:** Full methodology is in `docs/PHASE4_METHODOLOGY_v3.md` (current version v3.3, 860 lines, commit `c1acabf`). The document contains the conceptual lineage (Eloundou 2024, Eisfeldt-Schubert-Zhang 2023, Labaschin et al. 2025), full R-rubric specification, ICC threshold rationale (Koo & Li 2016), and Section 7.6 on Capability Dynamics & Instrument Validity (three-layer defense for the temporal-capability concern).

**Task decomposition examples — 15 firms in the prompt (9 software + 5 knowledge + 1 placebo):**

| Ticker | Tier | Profile |
|--------|------|---------|
| ZS | software | Zero-trust network security — real-time packet inspection, policy enforcement, threat detection from live streams |
| DDOG | software | Infrastructure observability — real-time metrics ingestion, anomaly detection, distributed tracing |
| CRWD | software | Endpoint security — kernel-level agents, behavioral telemetry, real-time threat detection |
| VEEV | software | Life sciences cloud — pharma CRM, regulatory submissions, clinical data, validated system documentation |
| PAYC | software | HCM/payroll — payroll compliance, employee records, benefits admin, time tracking |
| HUBS | software | CRM/marketing — email campaigns, pipeline management, contact records, marketing automation |
| LPSN | software | Conversational AI — customer chat, intent classification, conversation routing |
| EGAN | software | Knowledge management — knowledge base authoring, customer self-service, agent assist |
| ZIP | software | Online job marketplace — job-candidate matching, resume screening, employer-candidate messaging |
| MCO | knowledge | Credit ratings + analytical products — NRSRO issuance, research publishing, data feeds |
| SPGI | knowledge | Ratings + indices + market intelligence — credit ratings, S&P/Dow Jones indices, commodities pricing |
| MSCI | knowledge | Investment indices + ESG analytics — index calculation, ESG scoring, portfolio analytics |
| COUR | knowledge | Online education marketplace — video lecture delivery, learner-instructor matching, credentials |
| CHGG | knowledge | Homework help + textbook services — Q&A platform, writing assistance, tutoring marketplace |
| PFE | placebo | Pharmaceutical manufacturer — drug discovery, clinical trials, manufacturing, FDA approval |

Note: these firms appear in the prompt as **reasoning examples**, not calibration targets. The prompt now lists 5 archetypes (EGAN, ZS, HUBS, SPGI, DDOG) — one per business-model archetype. The full anchor sample of 14 firms in `config/anchor_firms.yaml` is used for the Step 6b reliability run (3 iterations each, ICC(3,1) computation). PFE is appended programmatically by `08_score_supply_rho.py` as a placebo when its 10-K text is present.

**Scoring source:** Must use pre-shock 10-K Item 1 text from `text_data/10k_extracts/`. Never score from web research or post-shock filings.

---

### Demand Side: δ_i (Demand Friction Score)

**Prompt:** `prompts/demand_delta_system.txt` (revised 2026-04-24, commit 6e23712)
**Script:** `scripts/09_score_demand_delta.py` (Phase 4 — not yet created)
**Output:** `data/processed/demand_friction.csv`

Measures how insulated a firm's customers are from switching to LLM-based alternatives, even when those alternatives could replicate the product's core tasks. δ_i is the moderator: high friction means replicability translates less into revenue loss.

**Three sub-scores (each ∈ {0.0, 0.1, ..., 1.0}):**

**δ_switch — Switching Cost** (Farrell & Klemperer 2007)
Four components: contractual (multi-year agreements, termination penalties), technical integration (custom APIs, embedded connectors to ERP/CRM/clinical systems, proprietary data formats), organizational (staff training/certification, change management), data portability (business data stored in hard-to-migrate formats). Score reflects how long and costly a replacement project would be for a typical enterprise customer.

**δ_error — Error Cost** (Agrawal, Gans & Goldfarb 2018 — Prediction Machines)
AI reduces prediction costs but not judgment costs. Error cost arises when AI mistakes carry irreversible, high-magnitude, or legally regulated consequences — regulatory penalties, patient harm, fiduciary liability, safety incidents. High error cost creates mandatory human oversight requirements that slow AI substitution regardless of capability.

**δ_data — Data/Network Moat** (Katz & Shapiro 1985; Rochet & Tirole 2003)
Proprietary, uniquely accumulated data or network effects that a new AI-native competitor cannot replicate from public data alone. Score reflects exclusivity, volume, and feedback loop strength. Public data, third-party feeds, and customer-provided data do not qualify.

**Composite:**
```
δ_composite = (δ_switch + δ_error + δ_data) / 3   ∈ [0, 1]
```

**Methodological constraint (prompt-level):** Each sub-score must be computed deterministically from the 6-level rubric (0.0, 0.2, 0.4, 0.6, 0.8, 1.0). Each level's rubric descriptor includes 4–6 categorical cross-sector examples (e.g., "payroll/HCM platforms with tax jurisdiction configurations", "clinical data platforms embedded in FDA-regulated trial workflows", "actuarial databases with insurer pooling") — these are sector-level guidance, NOT firm-specific anchor targets. The prompt explicitly forbids targeting any expected score. This was a deliberate Phase 3 change (commit 6e23712) to eliminate researcher discretion.

**Sub-score reasoning examples — 13 firms in the prompt (7 software + 5 knowledge + 1 placebo):**

Each anchor entry retains its narrative profile (which switching-cost mechanisms dominate, what drives error cost, how the data moat arises) but the numeric sub-score line was deliberately removed. Firms covered: ZS, DDOG, VEEV, PAYC, HUBS, LPSN, ZIP, MCO, SPGI, MSCI, CHGG, COUR, PFE.

**Knowledge-intensive services note (in prompt):** Rating agencies, index providers, financial data firms, and consulting firms have different switching-cost mechanisms than software firms — their "integration" is regulatory or institutional rather than technical. Error cost can be extreme even without software integration (e.g., a rating downgrade has legal and market consequences comparable to FDA pharma errors). The prompt's rubric and 6-level descriptors apply unchanged, but reasoning paths differ. Consumer-facing education platforms (homework help, MOOCs) represent the opposite profile — near-zero switching friction, direct LLM substitution.

---

## Pipeline State

| Step | Script | Output | Status |
|------|--------|--------|--------|
| 1. Firm universe | `01_build_firm_universe.py` | `data/raw/firm_universe.csv` | **DONE** — 321 firms (3 tiers) |
| 2. 10-K Item 1 extraction | `02_collect_10k_text.py` | `text_data/10k_extracts/*.txt` | **DONE** — 321/321 in-universe firms |
| 3. Financial panel | `03_build_financial_panel.py` | `data/processed/financial_panel.csv` | **DONE** — 321 firms, 61,857 obs |
| 4. RPO panel | `04_build_rpo_quarterly.py` | `data/processed/rpo_quarterly.csv` | **DONE** — 279 ok / 12 no_rpo / 30 placebo skipped |
| 5. Billings panel | `05_build_billings_panel.py` | `data/processed/billings_panel.csv` | **DONE** — 321 firms, 8,207 rows |
| 6. Margin panel | `06_build_margin_panel.py` | `data/processed/margin_panel.csv` | **DONE** — 321 firms, 41,960 rows |
| 7. AI mention panel | `07_build_ai_mention_panel.py` | `data/processed/ai_mention_panel.csv` | **DONE** — 321 firms, 8,437 filings |
| Phase 3 — prompts revised | `prompts/*.txt` | — | **DONE** (2026-04-24, commits aa2f687 + 6e23712) |
| Phase 6 — robustness notes | `docs/phase6_notes.md` | — | **DONE** (2026-04-24, commit 75fab03) |
| Repo cleanup | — | — | **DONE** (2026-04-24, commits 5381eee + 3018b61) |
| 8. Supply scoring (ρ_i) | `08_score_supply_rho.py` | `data/processed/supply_rho.csv` | **IN PROGRESS** — smoke validated (commit `c1acabf`); anchor reliability run (Step 6b) pending |
| 9. Demand scoring (δ_i) | `09_score_demand_delta.py` (not yet created) | `data/processed/demand_delta.csv` | After Step 7 (full supply run) |
| 10. DiD regressions | `analysis/did_v3.R` | — | After steps 8–9 |

---

## Phase 2 Data State (Verified 2026-04-22)

All Phase 2 panels confirmed against filesystem:

| Panel | File | Rows | Firms | Notes |
|-------|------|------|-------|-------|
| Financial | `financial_panel.csv` | 61,857 | 321 | 10 metrics; revenue rows = 8,207 |
| RPO | `rpo_quarterly.csv` | 5,293 | 279 | 12 no_rpo; 30 placebo skipped |
| Billings | `billings_panel.csv` | 8,207 | 321 | gap-aware rpo_delta (120-day threshold) |
| Margin | `margin_panel.csv` | 41,960 | 321 | 5 ratios; P1/P99 winsorized |
| AI Mention | `ai_mention_panel.csv` | 8,437 | 321 | 22-pattern lexicon; 49.5× post/pre ratio |

Text corpus:
- `text_data/10k_extracts/`: 321 files covering all in-universe firms (ACN + NWSA resolved on 2026-04-23)
- `text_data/10k10q_extracts/`: 8,437 files (one per filing row in ai_mention_panel)
- Total: 2.1 GB

**`data/processed/panel.csv` is stale:** 223 firms, 13 cols, old wide-format schema from before the three-tier expansion. Do not use for analysis — use `financial_panel.csv` + `rpo_quarterly.csv` instead.

---

## Phase 3 Complete (Prompt Engineering)

Phase 3 ran from 2026-04-23 to 2026-04-24. Both scoring prompts were rewritten to eliminate researcher discretion from the measurement process, and thesis-writing phase notes were captured while methodology thinking was fresh.

**Commits (linear sequence on main):**
- `aa2f687` — Supply prompt: anchor score targets removed, integration depth penalty removed, pure Eloundou β aggregation
- `6e23712` — Demand prompt: anchor numeric sub-score targets removed, all 18 rubric descriptors (3 sub-scores × 6 levels each) strengthened with cross-sector categorical examples, knowledge-intensive services note rewritten to remove numerical guidance
- `75fab03` — `docs/phase6_notes.md` created with robustness battery and methodology defense plan
- `5381eee` — Repo cleanup: 6 deprecated scripts deleted, stale Claude Code worktree removed
- `3018b61` — Typo fix in `scripts/utils/xbrl.py` docstring (stray duplicate prefix)

**Core methodological principle established in Phase 3:**
The supply and demand scoring scores emerge *deterministically* from rubric matching and aggregation formulas. There is no author-picked calibration target embedded anywhere in the scoring pipeline. Firm examples in prompts teach classification logic ("zero-trust security products have real-time enforcement tasks which classify as E0") rather than numeric targets ("ZS should score ≈10"). This distinction is the central defense against researcher-degrees-of-freedom critique.

**Current prompt sizes:**
- `prompts/supply_rho_system.txt`: 345 lines, 19,891 chars (~4,200 tokens)
- `prompts/demand_delta_system.txt`: 386 lines, 23,059 chars (~4,900 tokens)

Supply (~4,200 tokens) is just above the 4,096-token Haiku 4.5 cache minimum; demand (~4,900 tokens) clears it comfortably. The Sonnet minimum is 1,024 tokens but Sonnet is not used in this pipeline. Cache hit cost per firm: ~$0.0004 (supply) + ~$0.0005 (demand).

---

## Phase 4 Status (Scoring Validation In Progress)

Phase 4 is the LLM scoring stage. Steps 4-7 form the supply-scoring pipeline; Step 8 will be the demand-scoring pipeline (not yet started).

### Steps Completed

| Step | Commit | What |
|------|--------|------|
| 4 — Schema + R-rubric prompt rewrite | `90deade` | E0/E1/E2 → R0/R1/R2; 5 anchor archetypes; per-task reasoning |
| 5 — Multi-iter infra + ICC utility | `b485cf1` | `--iterations N` flag; new wide CSV; `scripts/utils/icc.py` |
| 6a-patch — Reasoning length fix | `dbcefa9` | `max_length` 200→350 (model writes ~250 char naturally) |
| 6a-patch2 — Remove model-reported counts | `d7836e5` | Schema returns only ticker/tasks/overall_reasoning; `compute_aggregates()` derives counts |
| 6a-patch3 — De-anchor prompt + add methodology | `c1acabf` | Removed Section 3 sectoral priors and Section 7 numeric scores; methodology v3.3 added at `docs/PHASE4_METHODOLOGY_v3.md` |

### Smoke Test Results (post-c1acabf)

| Firm | Predicted Range (advisory) | Observed ρ_mean | ρ_std | Verdict |
|------|----------------------------|-----------------|-------|---------|
| ZS | [40, 55] | 42.2 | 7.58 | IN RANGE |
| EGAN | [70, 90] (revised) | 75.4 | 6.96 | IN RANGE |
| HUBS | [65, 85] | 76.0 | 3.91 | IN RANGE |

EGAN range was revised from [85, 100] to [70, 90] in methodology v3.2 after the model consistently identified knowledge-base retrieval as R2 (RAG, not vanilla R1). All three smoke firms now produce clean multi-iteration output. ZS smoke v3 → v4 shift was 0.0 points (de-anchoring had zero empirical effect).

### Next Steps

**Step 6b (next):** 14-15 firm anchor reliability run. `python3 scripts/08_score_supply_rho.py --test` (defaults to `--iterations 3` in test mode). Output: `data/processed/supply_rho_anchor.csv`. Cost: ~$0.45. Then `python3 scripts/utils/icc.py` to compute ICC(3,1) on `rho_iter1/2/3` columns.

Decision gate (from `docs/PHASE4_METHODOLOGY_v3.md` Section 5.6):
- ICC ≥ 0.90 → proceed to Step 7 single-run full sample (321 × 1 = ~$3.21)
- 0.75 ≤ ICC < 0.90 → proceed to Step 7 multi-run full sample (321 × 3 = ~$9.63)
- ICC < 0.75 → reject; methodology requires further refinement

**Step 7:** Full sample scoring (321 firms × N iterations). Output: `data/processed/supply_rho.csv` (different file from anchor reliability output). Iteration count determined by Step 6b ICC result.

**Step 8 (Phase 4 demand pipeline):** Demand scoring for 321 firms. Requires `scripts/09_score_demand_delta.py` (not yet created), uses preserved Phase 3 prompt at `prompts/demand_delta_system.txt`. Spec to be issued after supply pipeline completes.

### Phase 4 Cumulative Cost Tracker

| Stage | Cost |
|-------|------|
| Smoke v1 (failed) | $0.22 |
| Smoke v2 (HUBS+EGAN clean) | $0.07 |
| Smoke v3 (ZS clean post-patch2) | $0.06 |
| Smoke v4 (ZS post-de-anchor verification) | $0.06 |
| **Phase 4 to date** | **$0.41** |
| Step 6b anchor reliability (14 × 3) | +$0.45 |
| Step 7 full sample (single-iter or multi-iter) | +$3.21 to +$9.63 |

Methodology v3 envelope: $5-15 total Phase 4. We are tracking inside.

---

## Phase 5+ Pending (External Validation, Assembly, Regressions)

1. **External validators (robustness — see `docs/phase6_notes.md`):**
   - Cross-reference ρ against Eloundou et al. (2024) `occ_level.csv` firm-level aggregate at primary SIC (via OEWS weights)
   - Anthropic Economic Index task usage (release_2026_01_15) for realized-usage comparison
   - Microsoft Copilot applicability scores (Tomlinson et al. 2025)
   - Felten LM-AIOE and genAI-AIOE (2023 updates)

2. **Master panel assembly:** Merge all panels (firm_universe + financial + RPO + billings + margin + AI mention + supply_rho + demand_delta) into `master_panel.parquet`. This is the input to R.

3. **R econometrics (`analysis/did_v3.R`):**
   - `did_baseline` — fixest two-way FE with Post×ρ
   - `continuous_dose` — contdid package for continuous treatment DiD
   - `event_study` — dynamic treatment effects by quarter around 2022Q4
   - `staggered_shocks` — robustness to alternative shock dates
   - `outcomes_cascade` — cascade from revenue → billings → margins
   - `honest_did_sensitivity` — Rambachan-Roth bounds for parallel trends
   - Inference: WCR31 wild cluster bootstrap + CR2 Bell-McCaffrey SE

4. **Robustness batteries (supply + demand):** See `docs/phase6_notes.md` for the full robustness plan — α/β/γ Eloundou aggregations, PCA of demand sub-scores, equal-weighting defense as maximum entropy prior, weighted alternative composites, single sub-score regressions.

5. **Methodology defense coverage (in thesis):**
   - AAHR exclusion problem — reframe ρ_i as product-level substitution rather than worker-level displacement (all 321 firms sit in NAICS 51/54 which AAHR excluded as AI-producer sectors)
   - Variance compression within SIC 7370-7379 — report both raw-scale and within-sector-demeaned correlations against external validators
   - Three-tier identification — identifying variation is both within-software AND across tiers

6. **Thesis documentation:** Methodology section draft, diagnostics appendix, figure notebook finalization.

---

## Current File Structure

```
scripts/
  01_build_firm_universe.py     # DONE — do not re-run (SEC rate limits)
  02_collect_10k_text.py        # DONE — do not re-run (SEC rate limits)
  03_build_financial_panel.py   # DONE — do not run without --tickers
  04_build_rpo_quarterly.py     # DONE — builds RPO quarterly from EDGAR companyfacts
  05_build_billings_panel.py    # DONE — billings = revenue + gap-aware rpo_delta
  06_build_margin_panel.py      # DONE — 5 ratios from financial_panel, P1/P99 winsorized
  07_build_ai_mention_panel.py  # DONE — AI mention counts from 10-K/10-Q full text
  08_score_supply_rho.py        # DONE (commits b485cf1 + dbcefa9 + d7836e5) — multi-iter scoring with R-rubric schema
  09_score_demand_delta.py      # NOT YET CREATED — Phase 4 Step 8 (after supply pipeline complete)
  build_eloundou_corpus.py      # KEPT for Phase 6 robustness alternative (ONET embedding match) — not used in primary scoring

  utils/
    __init__.py
    edgar.py                    # EDGAR client: submissions, companyfacts, filing text
    xbrl.py                     # XBRL extraction, Q4 formula, RPO fallback chain
    text_sections.py            # 10-K section parser (Item 1 extraction + iXBRL)
    logging_setup.py            # Structured logging
    llm_client.py               # DONE — schema-agnostic API client with cache verification + retry logic
    schemas.py                  # SupplyScore + DemandScore + ProductTaskWithRubric + compute_aggregates()
    icc.py                      # ICC(3,1) computation utility (Koo & Li 2016) — Phase 4 Step 5 commit b485cf1
    task_matching.py            # KEPT for Phase 6 robustness alternative — not used in primary scoring

analysis/
  did_v3.R                      # Two-way FE DiD + WCB; update after scoring

prompts/
  supply_rho_system.txt         # Phase 4 R-rubric (R0/R1/R2), de-anchored archetypes (commit c1acabf, ~20.3KB)
  demand_delta_system.txt       # Phase 3 (preserved) — δ_switch/δ_error/δ_data, 6-level rubric (~23.1KB)

docs/
  PHASE4_METHODOLOGY_v3.md      # Full Phase 4 methodology v3.3 — R-rubric, ICC, capability dynamics defense
  phase6_notes.md               # Thesis writing phase notes — robustness battery + methodology defense

config/
  ai_mention_lexicon.yaml       # 22-pattern strict post-ChatGPT lexicon (3 categories)
  anchor_firms.yaml             # Still contains old numeric targets — Phase 4 Step 1 decision pending
  shock_dates.yaml              # Shock date config (2022-10-01)
  universe_filters.yaml         # SIC, exchange, coverage thresholds
  universe_tickers.yaml         # Explicit ticker lists per tier

data/
  external/
    eloundou_full_labelset.tsv         # ONET task corpus from Eloundou et al. (2024) — Phase 6 robustness only
    eloundou_task_embeddings.parquet   # Pre-computed task embeddings for embedding-match alternative — Phase 6 only
  raw/
    firm_universe.csv                  # 321 firms: ticker, CIK, SIC, exchange, tier, sector_code
    edgar_cache/                       # SEC EDGAR API response cache — gitignored, populated on demand
  processed/
    financial_panel.csv         # 321 firms, 61,857 obs (10 metrics, 2019Q1–2025Q4)
    rpo_quarterly.csv           # 279 firms, 5,293 rows (quarterly RPO snapshots)
    billings_panel.csv          # 321 firms, 8,207 rows (revenue + gap-aware rpo_delta)
    margin_panel.csv            # 321 firms, 41,960 rows (5 ratios, P1/P99 winsorized)
    ai_mention_panel.csv        # 321 firms, 8,437 filings (22-pattern lexicon, 3 categories)
    supply_rho_smoke_v2.csv     # Phase 4 Step 6a smoke v2 (HUBS+EGAN clean) — diagnostic record, kept
    supply_rho_smoke_v3.csv     # Phase 4 Step 6a-patch2 smoke (ZS clean) — diagnostic record, kept
    supply_rho_smoke_v4.csv     # Phase 4 Step 6a-patch3 smoke (ZS post-de-anchor) — diagnostic record, kept
    supply_rho_errors.jsonl     # Error log from smoke iterations (4 entries, all resolved)
    # supply_rho_anchor.csv       — Step 6b output (not yet generated)
    # supply_rho.csv               — Step 7 output (not yet generated)
    billings_panel_qa.json      # QA log: billings coverage per firm
    margin_panel_qa.json        # QA log: margin warnings per firm
    ai_mention_panel_qa.json    # QA log: mention counts per firm
    rpo_quarterly_qa.json       # QA log: RPO tag coverage per firm
    financial_panel_qa.json     # QA log: financial metric coverage per firm
    extraction_qa.json          # QA log: 10-K extraction quality

text_data/
  10k_extracts/                 # Pre-shock 10-K Item 1 text (gitignored) — 321 files
  10k10q_extracts/              # Full filing text cache for AI mention panel (gitignored) — 8,437 files

notebooks/
  thesis_notebook.ipynb         # ALL figures go here, nowhere else
```

---

## Repository Rules (Always Follow)

### Scoring Rules
- **Primary scoring model:** `claude-haiku-4-5-20251001` (~$0.016/firm). Budget is ~$10 total for supply + demand across 321 firms.
- **NEVER use Sonnet, Opus, or any larger model** — they would cost $30–100+. Hard pin in `llm_client.py`.
- **Fallback model:** Gemini 2.5 Flash free tier (`gemini-2.5-flash`), used only on anchor validation failure, never for cost reasons.
- **Source text:** Always score from `text_data/10k_extracts/{TICKER}.txt` — never from web research, company websites, or post-shock filings.
- **Pre-shock only:** 10-K filings dated < 2022-11-01. Post-shock text is contaminated by firms' own AI adaptations.
- **Supply score formula:** pure Eloundou β: `(E1 + 0.5×E2) / n_tasks` then `× 99 + 1`. No integration penalty, no `max(0, …)` clipping, no researcher-calibrated adjustment. Emerges deterministically from the task classification.
- **Demand score formula:** `(δ_switch + δ_error + δ_data) / 3`. Each sub-score must be in `{0.0, 0.1, ..., 1.0}`. Rounded to 3 decimal places.
- **Deterministic computation rule:** Both prompts explicitly instruct the model not to adjust scores manually. If the pipeline detects a model-reported score diverging from the formula-computed score by > 0.01, log a warning.

### Financial Panel Rules
- **IIIV excluded** — post-privatization accounting restatements (FY2022: −40%, FY2023: −49%) make revenue data unreliable. Hardcoded exclusion in `03_build_financial_panel.py`.
- **Q4 revenue** = `Annual_FY − (Q1 + Q2 + Q3)`. Negative values are discarded (tag mismatch / amended filing artifact).
- **Multiple annual filings** for same fiscal year: higher value retained (original preferred over post-restatement amendment).
- **Do not run `03_build_financial_panel.py` without `--tickers`** — it overwrites the full panel.

### RPO Rules
- RPO is a **balance-sheet stock** (point-in-time snapshot at period end), not a flow metric — no cumulative summing.
- Tag hierarchy: `RevenueRemainingPerformanceObligation` → `ContractWithCustomerLiability` → `CWCL_Current + CWCL_Noncurrent` → `DeferredRevenue + DeferredRevenueNoncurrent`
- For duplicate `period_end` dates (amended filings), keep latest filed.
- 12/291 attempted firms genuinely have no reportable RPO data — this is expected. 30 placebo firms are skipped by design.
- **RPO/Revenue ratio:** P99-winsorized; use pre-shock firm-level average as moderator if needed.

### Billings Panel Rules
- **Formula:** `billings_t = revenue_t + (rpo_t − rpo_{t-1})`
- **Gap-aware rpo_delta:** NaN when consecutive RPO observations are >120 days apart. First observation per firm is always NaN.
- **No coverage filter at build time** — billings=NaN is valid. Apply thresholds at analysis time in R.
- **Negative billings are valid** — reflect customer churn or contract restructuring.

### 10-K Extraction Rules
- **Source:** SEC EDGAR only (`data.sec.gov/submissions/` API)
- **Cutoff:** Filing date strictly < 2022-11-01
- **Section:** Item 1 Business Description — extract in full, no truncation
- **No fallbacks:** No Wayback Machine, no product pages, no company websites

### What Goes Where
| Item | Location | Notes |
|---|---|---|
| All figures | `notebooks/thesis_notebook.ipynb` | ONLY here — no standalone .py figure scripts |
| Firm universe | `scripts/01_build_firm_universe.py` | SEC EDGAR SIC 7370-7379 + knowledge + placebo tiers |
| 10-K extraction | `scripts/02_collect_10k_text.py` | Pre-shock only |
| Financial panel | `scripts/03_build_financial_panel.py` | XBRL quarterly revenue + financials |
| RPO panel | `scripts/04_build_rpo_quarterly.py` | XBRL balance sheet stock |
| Billings panel | `scripts/05_build_billings_panel.py` | revenue + gap-aware rpo_delta |
| Margin panel | `scripts/06_build_margin_panel.py` | 5 ratios from financial_panel |
| AI mention panel | `scripts/07_build_ai_mention_panel.py` | 22-pattern lexicon, 10-K/10-Q full text |
| Supply scoring | `scripts/08_score_supply_rho.py` + `prompts/supply_rho_system.txt` | Phase 4 |
| Demand scoring | `scripts/09_score_demand_delta.py` + `prompts/demand_delta_system.txt` | Phase 4 |
| LLM client | `scripts/utils/llm_client.py` | Shared; not yet created (Phase 4) |
| R regressions | `analysis/did_v3.R` | fixest + fwildclusterboot |
| LaTeX manuscript | Overleaf only | `thesis.tex` is gitignored — NEVER commit it |
| Methodology notes | `docs/phase6_notes.md` | Robustness battery + defense plan |

### Git Rules
- Commits authored solely by Hakan — no co-author attributions
- Never commit: `thesis.tex`, `data/raw/`, `text_data/`, `logs/`, API keys
- Always `git pull` before starting a new work session

---

## Key Numbers

| Parameter | Value |
|---|---|
| Firms in universe | 321 (256 primary_software + 35 primary_knowledge + 30 placebo) |
| 10-K extractions | 321/321 in-universe firms (ACN + NWSA resolved 2026-04-23) |
| Firms in financial panel | 321 (IIIV and ONTF absent by design) |
| Financial panel observations | 61,857 (10 metrics) · 8,207 revenue rows |
| Firms with RPO data | 279/291 attempted (30 placebo skipped; 12 no_rpo) |
| Billings panel | 8,207 rows · 229 computable / 21 sparse / 71 no_billings |
| Margin panel | 41,960 rows · 5 ratios · P1/P99 winsorized |
| AI mention panel | 8,437 filings · 321 firms · 49.5× post/pre ratio |
| Panel period | 2019Q1–2025Q4 |
| Shock date | 2022Q4 (`period_end ≥ 2022-10-01`) |
| Extraction cutoff | < 2022-11-01 (pre-shock) |
| Scoring model (hard pin) | `claude-haiku-4-5-20251001` |
| Scoring budget | ~$10 total (Haiku 4.5 @ ~$0.016/firm × 321 × 2 sides) |
| Supply prompt | 345 lines, 19,891 chars, ~4,200 tokens (cache-friendly) |
| Demand prompt | 386 lines, 23,059 chars, ~4,900 tokens (cache-friendly) |

---

## Common Mistakes to Avoid

1. **Don't overwrite `financial_panel.csv`** without `--tickers` — rebuilding takes time and risks introducing errors
2. **Don't score from web research** — treatment variable must come from pre-shock 10-K text only
3. **Don't use post-2022Q4 10-K filings** for scoring — post-shock adaptation contaminates the treatment
4. **Don't re-add IIIV** — excluded intentionally; its FY2022–2023 revenue data is accounting-restated garbage
5. **Don't put figure code in .py scripts** — everything in `thesis_notebook.ipynb`
6. **Don't commit `thesis.tex`** — Overleaf only
7. **Don't add web/Wayback fallback** to 10-K extraction — SEC EDGAR only
8. **Don't run scoring scripts without setting ANTHROPIC_API_KEY** — they fail silently after rate-limit delay
9. **Don't treat billings=NaN as imputable** — it means rpo_delta was non-computable; filter at analysis time, never fill
10. **Don't use `data/processed/panel.csv`** — stale 223-firm file; use `financial_panel.csv` + `rpo_quarterly.csv`
11. **Don't use Sonnet, Opus, or any larger model for scoring** — `claude-haiku-4-5-20251001` is the only model hard-pinned in `llm_client.py` and the only one within budget; larger models would cost 10-30× more.
12. **Don't leak API keys to CLAUDE.md, commit messages, or remote chat** — `.env` only, never in shared context
13. **Don't skip the anchor validation gate** — unvalidated scores are noise; budget spent on a bad full run cannot be recovered
14. **Don't re-introduce the integration depth penalty** — it was deliberately removed in commit aa2f687 to eliminate researcher discretion. Pure Eloundou β is the methodology anchor.
15. **Don't re-introduce numeric anchor targets to prompts** — the Phase 3 design principle is that scores emerge from rubric matching, not from firm-specific calibration. Adding "expected score ≈ X" lines to either prompt reintroduces anchor-pull.
16. **Don't re-create deleted scripts** — `score_literature_rubric.py`, `score_demand_friction.py`, and the four unnumbered pipeline scripts were deleted in commit 5381eee. Phase 4 scripts must be written fresh to match current methodology (no penalty, correct prompt paths, Haiku-pinned).
17. **Don't trust model-reported aggregate counts** — the `SupplyScore` schema deliberately removed `r0_count`, `r1_count`, `r2_count`, `n_tasks`, `raw_exposure`, and `normalized_score` from the model output (commit `d7836e5`). All aggregates are computed from `tasks` labels via `compute_aggregates()`. If you find yourself adding count fields back, you are recreating the self-consistency failure mode that took down the ZS smoke v2 run.

---

## Running the Pipeline

```bash
# Steps 1–7 already done. Do not re-run without good reason (SEC rate limits).
# Phase 3 complete (commits aa2f687 + 6e23712 + 75fab03 + 5381eee + 3018b61).
# Phase 4 in progress: schema + prompt + multi-iter infra + de-anchored prompt
# committed at 90deade → b485cf1 → dbcefa9 → d7836e5 → c1acabf.

# Phase 4 Step 6b (next): anchor reliability test
# Requires ANTHROPIC_API_KEY in .env (auto-loaded via python-dotenv)
python3 scripts/08_score_supply_rho.py --test \
    --output data/processed/supply_rho_anchor.csv     # 14-15 firms × 3 iter, ~$0.45

# Compute ICC(3,1) on the anchor reliability output
python3 scripts/utils/icc.py    # self-test only; full computation requires custom call

# Phase 4 Step 7 (after ICC gate): full sample scoring
# Iteration count determined by ICC band:
#   ICC ≥ 0.90 → single-iter (~$3.21)
#   0.75-0.90 → multi-iter (~$9.63)
python3 scripts/08_score_supply_rho.py --skip-existing \
    --output data/processed/supply_rho.csv             # 321 firms

# Phase 4 Step 8: demand scoring (requires writing 09_score_demand_delta.py first)
python3 scripts/09_score_demand_delta.py --skip-existing \
    --output data/processed/demand_delta.csv           # 321 firms

# Phase 5: regressions
Rscript analysis/did_v3.R
```

---

## Theoretical Context

**Second-order displacement:** AI displacing the software products that had themselves displaced workers from cognitive tasks in the 1990s–2000s. The Acemoglu–Restrepo (2022) task-substitution framework applies one level up the value chain: B2B software firms are *task externalizers* who face product obsolescence when LLMs can perform the same tasks at near-zero marginal cost.

**Supply side (ρ_i):** Eloundou et al. (2024) "GPTs are GPTs" task exposure framework, applied to product-level tasks instead of worker tasks. A product whose core task bundle is E1/E2-exposed faces the same substitution pressure as a worker in an E1/E2 occupation. The β aggregation (E1 + 0.5·E2 / n_tasks) is the middle of Eloundou's three aggregations (α = E1 only; β = E1 + 0.5·E2; γ = E1 + E2).

**Demand side (δ_i):** Even when a product is highly replicable, customers may not switch because of: (1) switching costs (Farrell & Klemperer 2007) — contractual, technical, organizational, and data portability barriers; (2) error costs (Agrawal, Gans & Goldfarb 2018) — regulatory, legal, and irreversibility consequences when AI makes mistakes; (3) data/network moats (Katz & Shapiro 1985; Rochet & Tirole 2003) — proprietary data accumulation and network effects that alternatives cannot replicate.

**Interaction hypothesis:** `β3 > 0` in the triple-interaction specification would mean higher demand friction dampens the negative effect of high LLM replicability. Equivalently, replicability only translates into revenue loss when demand friction is low — capability × (absence of friction) is the effective exposure channel.

**Three-tier design:** Primary software firms are the treatment group. Knowledge-intensive controls test whether any post-2022 revenue effect is AI-specific to software or affects all knowledge industries broadly. Placebo firms provide a falsification check — no plausible mechanism for LLM replicability to affect energy, manufacturing, or pharma revenues.

**Core identification insight (see `docs/phase6_notes.md`):** The thesis resolves a tension in the AI exposure literature. Occupation-level exposure scores (Eloundou 2024, Felten-Raj-Seamans 2023) uniformly predict high impact for knowledge workers, yet observed firm-level revenue responses are highly heterogeneous. The ρ × δ decomposition explains this heterogeneity by separating what LLMs CAN do (supply) from what customers WILL accept (demand). High capability does not imply high impact; impact emerges when replicability coincides with low friction.

---

## Thesis Writing Notes

See `docs/phase6_notes.md` for:
- Main DiD specification alternatives (triple interaction + effective exposure)
- Core identification framing (capability × friction decomposition)
- Demand composite robustness battery (weighted alternatives, PCA, single sub-score tests)
- Supply robustness (Eloundou α/β/γ, external validators: Eloundou occ_level, AEI, Tomlinson Microsoft, Felten)
- Methodology defense requirements (AAHR exclusion, variance compression, three-tier identification)
