# CLAUDE.md — AI Assistant Instructions

This file tells Claude (or any AI assistant) exactly what this repository is and how to work within it correctly. Read it fully before touching any code or data.

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

**Extended specification (supply × demand interaction):**
```
ln(Revenue_it) = α_i + δ_t + β₁·(Post_t × ρ_i) + β₂·(Post_t × ρ_i × δ_i) + ε_it
```

Where:
- `α_i` = firm fixed effects
- `δ_t` = time (quarter) fixed effects
- `ρ_i` = supply-side LLM replicability score ∈ [1, 100] (pre-shock, cross-sectional)
- `δ_i` = demand-side friction score ∈ [0, 1] (pre-shock, cross-sectional)
- `Post_t = 1` for `period_end ≥ 2022-10-01` (2022Q4 onward)

Inference: wild cluster bootstrap (WCB), clustered at firm level. R package: `fwildclusterboot`.

---

## The Two Treatment Variables

### Supply Side: ρ_i (LLM Replicability Score)

**Prompt:** `prompts/supply_rho_system.txt`
**Script:** `scripts/08_score_supply_rho.py`
**Output:** `data/processed/lit_scores.csv`

Measures the fraction of a firm's product task bundle that an LLM can replicate at near-zero marginal cost. Applied at the product level — one level up from Eloundou et al. (2024) who classify worker tasks.

**Framework (E0 / E1 / E2):**
- **E1** — direct LLM exposure: output is primarily text, document processing, communication, matching/ranking/routing on text content, structured queries, template generation
- **E2** — LLM + standard tools: requires database read/write or API calls before generating language output; a developer could build this in months using off-the-shelf components
- **E0** — no meaningful LLM exposure: continuously updating real-time data streams, physical hardware interaction, sub-second latency SLA as the product, deep proprietary system integration as the core moat

**Integration Depth Penalty** (captures structural switching-cost moats that reduce effective substitutability even for E1/E2 tasks):
- `0` — no significant moat; customer could switch to an LLM-based substitute with minimal friction
- `-1` — moderate moat; embedded in enterprise workflows, 6–18 month switch horizon (payroll/HCM, mid-market CRM, ITSM)
- `-2` — strong moat; the product IS the integration — deep proprietary connectors, years of customer-specific data, regulatory certification (core banking, semiconductor EDA, pharma regulatory cloud)

**Score formula:**
```
raw_exposure      = (E1_count + 0.5 × E2_count) / n_tasks
adjusted_exposure = max(0.0, raw_exposure + integration_depth_penalty / 10)
normalized_score  = round(adjusted_exposure × 99 + 1, 1)   ∈ [1, 100]
```

**Calibration anchors — 14 firms in `config/anchor_firms.yaml` (9 software + 5 knowledge; placebo anchor pending):**
| Ticker | Score | Tier | Profile |
|--------|-------|------|---------|
| ZS | ≈10 | software | Real-time zero-trust network security, all E0, moat=-2 |
| DDOG | ≈16 | software | Infrastructure observability, mostly E0, moat=-1 |
| CRWD | ≈22 | software | Endpoint security, real-time behavioral telemetry, moat=-2 |
| VEEV | ≈38 | software | Pharma CRM + regulatory submissions, mixed, moat=-2 |
| PAYC | ≈45 | software | HCM/payroll, text-heavy but compliance integration, moat=-1 |
| HUBS | ≈64 | software | CRM/marketing, core is written communication, moat=-1 |
| LPSN | ≈72 | software | Conversational AI platform, mostly E1, moat=0 |
| EGAN | ≈80 | software | Knowledge management, all E1, moat=0 |
| ZIP | ≈88 | software | Job marketplace, all text matching/routing, moat=0 |
| MCO | ≈30 | knowledge | Credit ratings; text outputs but regulatory moat |
| SPGI | ≈35 | knowledge | Ratings + indices + market intelligence |
| MSCI | ≈40 | knowledge | Indices + ESG analytics + portfolio tools |
| COUR | ≈60 | knowledge | Online courses, content ecosystem |
| CHGG | ≈75 | knowledge | Homework help, text-heavy, directly exposed |

**Scoring source:** Must use pre-shock 10-K Item 1 text from `text_data/10k_extracts/`. Never score from web research or post-shock filings.

---

### Demand Side: δ_i (Demand Friction Score)

**Prompt:** `prompts/demand_delta_system.txt`
**Script:** `scripts/09_score_demand_delta.py`
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

**Calibration anchors — 12 firms in `config/anchor_firms.yaml` (7 software + 5 knowledge; placebo anchor pending):**
| Ticker | δ_switch | δ_error | δ_data | δ_composite |
|--------|----------|---------|--------|-------------|
| ZS | 0.8 | 0.8 | 0.2 | 0.600 |
| DDOG | 0.6 | 0.4 | 0.2 | 0.400 |
| VEEV | 1.0 | 1.0 | 0.6 | 0.867 |
| PAYC | 0.6 | 0.8 | 0.1 | 0.500 |
| HUBS | 0.4 | 0.2 | 0.3 | 0.300 |
| LPSN | 0.3 | 0.1 | 0.2 | 0.200 |
| ZIP | 0.1 | 0.1 | 0.7 | 0.300 |
| MCO | 0.9 | 1.0 | 0.9 | 0.933 |
| SPGI | 0.8 | 0.9 | 0.9 | 0.867 |
| MSCI | 0.7 | 0.7 | 0.8 | 0.733 |
| CHGG | 0.1 | 0.1 | 0.1 | 0.100 |
| COUR | 0.4 | 0.3 | 0.2 | 0.300 |

---

## Pipeline State

| Step | Script | Output | Status |
|------|--------|--------|--------|
| 1. Firm universe | `01_build_firm_universe.py` | `data/raw/firm_universe.csv` | **DONE** — 321 firms (3 tiers) |
| 2. 10-K Item 1 extraction | `02_collect_10k_text.py` | `text_data/10k_extracts/*.txt` | **DONE** — 319/321 in-universe; ACN + NWSA absent (see note) |
| 3. Financial panel | `03_build_financial_panel.py` | `data/processed/financial_panel.csv` | **DONE** — 321 firms, 61,857 obs (10 metrics; 8,207 revenue rows) |
| 4. RPO panel | `04_build_rpo_quarterly.py` | `data/processed/rpo_quarterly.csv` | **DONE** — 279 ok / 12 no_rpo / 30 placebo skipped |
| 5. Billings panel | `05_build_billings_panel.py` | `data/processed/billings_panel.csv` | **DONE** — 321 firms, 8,207 rows; 229 computable / 21 sparse / 71 no_billings |
| 6. Margin panel | `06_build_margin_panel.py` | `data/processed/margin_panel.csv` | **DONE** — 321 firms, 41,960 rows; 5 ratios, P1/P99 winsorized |
| 7. AI mention panel | `07_build_ai_mention_panel.py` | `data/processed/ai_mention_panel.csv` | **DONE** — 321 firms, 8,437 filings; 49.5× post/pre mention ratio |
| 8. Supply scoring (ρ_i) | `08_score_supply_rho.py` | `data/processed/lit_scores.csv` | **NEXT** — Phase 3 prompt rewrite first; ACN + NWSA need resolution |
| 9. Demand scoring (δ_i) | `09_score_demand_delta.py` | `data/processed/demand_friction.csv` | After step 8 |
| 10. DiD regressions | `analysis/did_v3.R` | — | After steps 8–9 |

Note: `lit_scores.csv` already contains 247 entries from prior batch scoring (old prompt, software-only universe). These must be re-scored with the updated Phase 3 prompt before use.

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
- `text_data/10k_extracts/`: 321 files (includes IIIV + ONTF which are not in universe; ACN + NWSA absent)
- `text_data/10k10q_extracts/`: 8,437 files (one per filing row in ai_mention_panel)
- Total: 2.1 GB

**ACN and NWSA — pre-shock 10-K extract absent:** Both are in the universe (primary_knowledge tier) and have complete 10k10q caches (28 files each), so the AI mention panel covers them correctly. However, `text_data/10k_extracts/ACN.txt` and `text_data/10k_extracts/NWSA.txt` are absent — Phase 2 Step 2 did not complete for these two firms. Before Phase 3 supply and demand scoring, resolve by re-running `02_collect_10k_text.py --tickers ACN NWSA`, or exclude and document.

**`data/processed/panel.csv` is stale:** 223 firms, 13 cols, old wide-format schema from before the three-tier expansion. Do not use for analysis — use `financial_panel.csv` + `rpo_quarterly.csv` instead.

---

## Phase 3 Pending (Prompt Engineering — Before Scoring)

Before running `08_score_supply_rho.py` or `09_score_demand_delta.py`:

1. **ACN + NWSA pre-shock text:** Resolve missing `text_data/10k_extracts/ACN.txt` and `NWSA.txt`. Re-run `02_collect_10k_text.py --tickers ACN NWSA`, or exclude and document.

2. **`prompts/supply_rho_system.txt` rewrite:** The existing prompt was written for software-only scoring. Full rewrite required:
   - Eloundou E0/E1/E2 framework VERBATIM from Appendix A of Eloundou et al. 2024 "GPTs are GPTs" (arXiv:2303.10130)
   - O*NET v27.x mapping referenced for task identification
   - `business_model_types` expanded from 5 categories (a–e) to 11 (a–k) to handle knowledge + placebo tiers:
     - a: saas_software
     - b: enterprise_on_premises_software
     - c: developer_tools_infrastructure
     - d: security_compliance
     - e: information_services_data_provider
     - f: publishing_content
     - g: education_training
     - h: consulting_professional_services
     - i: physical_product
     - j: infrastructure_network
     - k: mixed_or_other
   - JSON output schema (strict format for parsing by `08_score_supply_rho.py`)
   - The 14 supply anchors already in `config/anchor_firms.yaml` (9 software + 5 knowledge) should be embedded in the prompt as calibration examples; verify the rewritten prompt's expected outputs match the file's `rho` values

3. **`prompts/demand_delta_system.txt` rewrite:**
   - The 12 demand anchors already in `config/anchor_firms.yaml` (7 software + 5 knowledge) should be embedded as calibration examples; verify expected scores match the file
   - Add a knowledge-product note: consulting/education/publishing have different switching cost mechanisms than software (engagement contracts vs software integration depth)
   - Demand friction composite definition = mean of three sub-scores (unchanged)

4. **`config/anchor_firms.yaml` — add placebo anchor:** The file already has 14 supply + 12 demand anchors covering software and knowledge tiers. The only missing piece is at least one placebo firm in both sections to validate the low end of the score distribution (expected ρ ≈ 5, δ_composite ≈ 0.1–0.2 for an industrial/energy firm). Add one of: CAT, DE, PFE, JNJ, or XOM.

---

## Phase 4 Pending (Scoring Validation and Execution — After Phase 3)

1. **`scripts/utils/llm_client.py`** — NOT YET CREATED. Write this before any scoring script. Must:
   - Support `anthropic` (primary) and `openai`-compatible endpoint (Gemini 2.5 Flash free tier fallback)
   - Enable prompt caching (system prompts are >5,000 tokens; caching cuts per-firm cost ~4×)
   - Reject placeholder API keys: must fail loudly if `ANTHROPIC_API_KEY` starts with `'REPLACE-ME'` or is empty string

2. **Supply scoring test:** `python3 scripts/08_score_supply_rho.py --test`
   - Scores the 14 anchor firms only (~$0.50 budget)
   - Decision gate — ALL must pass before full run:
     - Spearman rank correlation with expected anchors ≥ 0.80
     - Mean absolute difference ≤ 10 score points
     - Count of firms with |diff| > 15 must be ≤ 2
   - If gate fails: iterate on `supply_rho_system.txt` prompt, or switch to Gemini 2.5 Flash fallback

3. **Demand scoring test:** `python3 scripts/09_score_demand_delta.py --test`
   - Scores the 12 anchor firms
   - Same decision gate criteria as supply (Spearman ≥ 0.80, MAD ≤ 10, ≤ 2 outliers)
   - If gate fails: iterate on `demand_delta_system.txt` prompt, or switch to Gemini 2.5 Flash fallback

4. **Reliability block (test-retest):**
   - 30-firm subsample (5 from each of 6 `sector_code` values)
   - Score at T=0 three times (same prompt, same inputs, temperature=0)
   - Compute ICC(3,1) intraclass correlation
   - Decision gate: ICC ≥ 0.90 for both supply and demand
   - If fails: add prompt rigor (e.g., structured reasoning chains before numeric output)

5. **Full scoring run (after anchor + reliability gates pass):**
   - `python3 scripts/08_score_supply_rho.py --skip-existing` (321 firms, ~$5)
   - `python3 scripts/09_score_demand_delta.py --skip-existing` (321 firms, ~$5)

6. **Cross-LLM validation (conditional — free tier, post-production):**
   Required only if anchor Spearman on primary (Haiku) is in [0.80, 0.90). If anchor Spearman ≥ 0.90, skip this step — strong anchor correlation is sufficient validation.
   If required:
   - Score same 321 firms with Gemini 2.5 Flash free tier (no marginal cost)
   - Compare Spearman correlation Haiku vs Gemini
   - If Spearman(Haiku, Gemini) ≥ 0.80: primary scores validated for thesis use
   - If < 0.80: manual spot-check on 20 firms with largest disagreement to identify systematic bias

---

## Phase 5+ Pending (After Scoring)

1. **External validators (robustness):**
   - Cross-reference ρ against Eloundou et al. 2024 occupational exposure at firm's primary SIC
   - AEI AI economy dataset (if updated through 2024)
   - Tomlinson & Felten LM+GenAI occupational exposure score
   - BLS OEWS May 2022 occupational employment
   - O*NET v27.x tasks

2. **Master panel assembly:** Merge all panels (firm_universe + financial + RPO + billings + margin + AI mention + lit_scores + demand_friction) into `master_panel.parquet`. This is the input to R.

3. **R econometrics (`analysis/did_v3.R`):**
   - `did_baseline` — fixest two-way FE with Post×ρ
   - `continuous_dose` — contdid package for continuous treatment DiD
   - `event_study` — dynamic treatment effects by quarter around 2022Q4
   - `staggered_shocks` — robustness to alternative shock dates
   - `outcomes_cascade` — cascade from revenue → billings → margins
   - `honest_did_sensitivity` — Rambachan-Roth bounds for parallel trends
   - Inference: WCR31 wild cluster bootstrap + CR2 Bell-McCaffrey SE

4. **Three robustness checks:** TBD once main results are in.

5. **Thesis documentation:** Methodology section draft, diagnostics appendix, figure notebook finalization.

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
  08_score_supply_rho.py        # NEXT (Phase 4) — requires llm_client + Phase 3 prompt rewrite
  09_score_demand_delta.py      # After supply scoring

  utils/
    __init__.py
    edgar.py                    # EDGAR client: submissions, companyfacts, filing text
    xbrl.py                     # XBRL extraction, Q4 formula, RPO fallback chain
    text_sections.py            # 10-K section parser (Item 1 extraction + iXBRL)
    logging_setup.py            # Structured logging
    llm_client.py               # NOT YET CREATED — Phase 4 prep; anthropic/gemini, prompt caching

analysis/
  did_v3.R                      # Two-way FE DiD + WCB; update after scoring

prompts/
  supply_rho_system.txt         # Supply-side system prompt (E0/E1/E2 framework) — Phase 3 rewrite needed
  demand_delta_system.txt       # Demand-side system prompt (δ_switch/δ_error/δ_data) — Phase 3 rewrite needed

config/
  ai_mention_lexicon.yaml       # 22-pattern strict post-ChatGPT lexicon (3 categories)
  anchor_firms.yaml             # 14 supply + 12 demand anchors (software + knowledge); placebo anchor pending
  shock_dates.yaml              # Shock date config (2022-10-01)
  universe_filters.yaml         # SIC, exchange, coverage thresholds
  universe_tickers.yaml         # Explicit ticker lists per tier

data/
  raw/
    firm_universe.csv           # 321 firms: ticker, CIK, SIC, exchange, tier, sector_code
  processed/
    financial_panel.csv         # 321 firms, 61,857 obs (10 metrics, 2019Q1–2025Q4)
    rpo_quarterly.csv           # 279 firms, 5,293 rows (quarterly RPO snapshots)
    billings_panel.csv          # 321 firms, 8,207 rows (revenue + gap-aware rpo_delta)
    margin_panel.csv            # 321 firms, 41,960 rows (5 ratios, P1/P99 winsorized)
    ai_mention_panel.csv        # 321 firms, 8,437 filings (22-pattern lexicon, 3 categories)
    lit_scores.csv              # Supply scores (ρ_i) — 247 entries from old batch; needs full re-score
    billings_panel_qa.json      # QA log: billings coverage per firm
    margin_panel_qa.json        # QA log: margin warnings per firm
    ai_mention_panel_qa.json    # QA log: mention counts per firm
    rpo_quarterly_qa.json       # QA log: RPO tag coverage per firm
    financial_panel_qa.json     # QA log: financial metric coverage per firm
    extraction_qa.json          # QA log: 10-K extraction quality
    panel.csv                   # STALE — 223 firms, old schema; do not use

text_data/
  10k_extracts/                 # Pre-shock 10-K Item 1 text (gitignored) — 321 files (ACN + NWSA absent)
  10k10q_extracts/              # Full filing text cache for AI mention panel (gitignored) — 8,437 files

notebooks/
  thesis_notebook.ipynb         # ALL figures go here, nowhere else
```

---

## Repository Rules (Always Follow)

### Scoring Rules
- **Primary scoring model:** claude-haiku-4-5-20251001 (~$0.016/firm). Budget is ~$10 total for supply + demand across 321 firms. Fallback model on anchor validation failure: Gemini 2.5 Flash free tier — NEVER auto-escalate to Sonnet or Opus, which would blow the budget.
- **Source text:** Always score from `text_data/10k_extracts/{TICKER}.txt` — never from web research, company websites, or post-shock filings
- **Pre-shock only:** 10-K filings dated < 2022-11-01; post-shock text is contaminated by firms' own AI adaptations
- **Supply score formula:** recomputed from task counts in pipeline (see above) — model's own `normalized_score` is for comparison only; pipeline's computed value is authoritative
- **Demand score:** each sub-score must be in `{0.0, 0.1, ..., 1.0}`; composite is arithmetic mean

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
- **RPO/Revenue ratio:** P99-winsorized; use pre-shock firm-level average as moderator if needed. Raw RPO is correlated with firm size (r≈0.6); ratio is size-independent (r≈−0.007).

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
| Supply scoring | `scripts/08_score_supply_rho.py` + `prompts/supply_rho_system.txt` | |
| Demand scoring | `scripts/09_score_demand_delta.py` + `prompts/demand_delta_system.txt` | |
| LLM client | `scripts/utils/llm_client.py` | Shared by both scoring scripts; not yet created |
| R regressions | `analysis/did_v3.R` | fixest + fwildclusterboot |
| LaTeX manuscript | Overleaf only | `thesis.tex` is gitignored — NEVER commit it |

### Git Rules
- Commits authored solely by Hakan — no co-author attributions
- Never commit: `thesis.tex`, `data/raw/`, `text_data/`, `logs/`, API keys

---

## Key Numbers

| Parameter | Value |
|---|---|
| Firms in universe | 321 (256 primary_software + 35 primary_knowledge + 30 placebo) |
| 10-K extractions | 319/321 in-universe (ACN + NWSA absent; IIIV + ONTF present but excluded from universe) |
| Firms in financial panel | 321 (IIIV and ONTF absent by design) |
| Financial panel observations | 61,857 (10 metrics) · 8,207 revenue rows |
| Firms with RPO data | 279/291 attempted (30 placebo skipped; 12 no_rpo) |
| Billings panel | 8,207 rows · 229 computable / 21 sparse / 71 no_billings |
| Margin panel | 41,960 rows · 5 ratios · P1/P99 winsorized |
| AI mention panel | 8,437 filings · 321 firms · 49.5× post/pre ratio |
| Panel period | 2019Q1–2025Q4 |
| Shock date | 2022Q4 (`period_end ≥ 2022-10-01`) |
| Extraction cutoff | < 2022-11-01 (pre-shock) |
| Scoring budget | ~$10 total (Haiku 4.5 @ ~$0.016/firm × 321 × 2 sides) |

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
9. **Don't score ACN or NWSA** until their `text_data/10k_extracts/` files are resolved
10. **Don't treat billings=NaN as imputable** — it means rpo_delta was non-computable; filter at analysis time, never fill
11. **Don't use `data/processed/panel.csv`** — stale 223-firm file; use `financial_panel.csv` + `rpo_quarterly.csv`
12. **Don't use claude-sonnet-4-6 or any larger model for scoring** — Haiku 4.5 is the only model within budget; Sonnet/Opus would cost $30–100
13. **Don't leak API keys to CLAUDE.md, commit messages, or remote chat** — `.env` only, never in shared context
14. **Don't skip the anchor validation gate** — unvalidated scores are noise; the budget spent on a bad full run cannot be recovered

---

## Running the Pipeline

```bash
# Steps 1–7 already done. Do not re-run steps 1–4 without good reason (SEC rate limits).

# Pre-Phase 3: resolve ACN + NWSA missing pre-shock extracts
python3 scripts/02_collect_10k_text.py --tickers ACN NWSA

# Phase 3: prompt rewrites + placebo anchor addition (manual work — no script)

# Phase 4, Step 1: build llm_client utility
# (write scripts/utils/llm_client.py — not yet created)

# Phase 4, Step 2: supply scoring validation
export ANTHROPIC_API_KEY=sk-ant-...
python3 scripts/08_score_supply_rho.py --test           # 14 anchors, ~$0.50; must pass gate
python3 scripts/08_score_supply_rho.py --skip-existing  # full run, ~$5

# Phase 4, Step 3: demand scoring validation
python3 scripts/09_score_demand_delta.py --test         # 12 anchors; must pass gate
python3 scripts/09_score_demand_delta.py --skip-existing  # full run, ~$5

# Phase 5: regressions
Rscript analysis/did_v3.R
```

---

## Theoretical Context

**Second-order displacement:** AI displacing the software products that had themselves displaced workers from cognitive tasks in the 1990s–2000s. The Acemoglu–Restrepo (2022) task-substitution framework applies one level up the value chain: B2B software firms are *task externalizers* who face product obsolescence when LLMs can perform the same tasks at near-zero marginal cost.

**Supply side (ρ_i):** Eloundou et al. (2024) "GPTs are GPTs" task exposure framework, applied to product-level tasks instead of worker tasks. A product whose core task bundle is E1/E2-exposed faces the same substitution pressure as a worker in an E1/E2 occupation.

**Demand side (δ_i):** Even when a product is highly replicable, customers may not switch because of: (1) switching costs (Farrell & Klemperer 2007) — contractual, technical, organizational, and data portability barriers; (2) error costs (Agrawal, Gans & Goldfarb 2018) — regulatory, legal, and irreversibility consequences when AI makes mistakes; (3) data/network moats (Katz & Shapiro 1985; Rochet & Tirole 2003) — proprietary data accumulation and network effects that alternatives cannot replicate.

**Interaction hypothesis:** β₂ < 0 would mean high-replicability firms with low demand friction suffer most. High demand friction (δ_i → 1) buffers against the negative replicability effect.

**Three-tier design:** Primary software firms are the treatment group. Knowledge-intensive controls test whether any post-2022 revenue effect is AI-specific to software or affects all knowledge industries broadly. Placebo firms provide a falsification check — no plausible mechanism for LLM replicability to affect energy, manufacturing, or pharma revenues.
