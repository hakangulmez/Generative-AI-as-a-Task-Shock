# CLAUDE.md — AI Assistant Instructions

This file tells Claude (or any AI assistant) exactly what this repository is and how to work within it correctly. Read it fully before touching any code or data.

---

## What This Repo Is

**Master's thesis empirical pipeline** for:
> *"Generative AI as a Task Shock: Product-Level LLM Substitution in B2B Software Markets"*
> Hakan Zeki Gülmez · TUM School of Management · Supervisor: Prof. Dr. Helmut Farbmacher · Submission: October 2026

**Research question:** Do B2B software firms whose products are more replicable by LLMs experience worse revenue growth after the ChatGPT shock (2022Q4)? And does demand-side friction moderate that effect?

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

**Prompt:** `prompts/literature_rubric_system.txt`
**Script:** `scripts/score_literature_rubric.py`
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

**Calibration anchors (ground truth):**
| Ticker | Score | Profile |
|--------|-------|---------|
| ZS | ≈10 | Real-time zero-trust network security, all E0, moat=-2 |
| DDOG | ≈16 | Infrastructure observability, mostly E0, moat=-1 |
| CRWD | ≈22 | Endpoint security, real-time behavioral telemetry, moat=-2 |
| VEEV | ≈38 | Pharma CRM + regulatory submissions, mixed, moat=-2 |
| PAYC | ≈45 | HCM/payroll, text-heavy but compliance integration, moat=-1 |
| HUBS | ≈64 | CRM/marketing, core is written communication, moat=-1 |
| LPSN | ≈72 | Conversational AI platform, mostly E1, moat=0 |
| EGAN | ≈80 | Knowledge management, all E1, moat=0 |
| ZIP | ≈88 | Job marketplace, all text matching/routing, moat=0 |

**Scoring source:** Must use pre-shock 10-K Item 1 text from `text_data/10k_extracts/`. Never score from web research or post-shock filings.

---

### Demand Side: δ_i (Demand Friction Score)

**Prompt:** `prompts/demand_friction_system.txt`
**Script:** `scripts/score_demand_friction.py`
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

**Calibration anchors:**
| Ticker | δ_switch | δ_error | δ_data | δ_composite |
|--------|----------|---------|--------|-------------|
| ZS | 0.8 | 0.8 | 0.2 | 0.60 |
| DDOG | 0.6 | 0.4 | 0.2 | 0.40 |
| VEEV | 1.0 | 1.0 | 0.6 | 0.87 |
| PAYC | 0.6 | 0.8 | 0.1 | 0.50 |
| HUBS | 0.4 | 0.2 | 0.3 | 0.30 |
| LPSN | 0.3 | 0.1 | 0.2 | 0.20 |
| ZIP | 0.1 | 0.1 | 0.7 | 0.30 |

---

## Pipeline State

| Step | Script | Output | Status |
|------|--------|--------|--------|
| 1. Firm universe | `build_firm_universe.py` | `data/raw/firm_universe.csv` | **DONE** — 248 firms |
| 2. 10-K Item 1 extraction | `collect_10k_text.py` | `text_data/10k_extracts/*.txt` | **DONE** — 239/248 firms |
| 3. Financial panel | `build_financial_panel.py` | `data/processed/financial_panel.csv` | **DONE** — 223 firms, 5,493 obs |
| 4. RPO panel | `build_rpo_quarterly.py` | `data/processed/rpo_quarterly.csv` | **DONE** — 197/223 firms |
| 5. Base panel merge | (manual merge in build script) | `data/processed/panel.csv` | **DONE** — 223 firms, 5,493 obs |
| 6. Supply scoring (ρ_i) | `score_literature_rubric.py` | `data/processed/lit_scores.csv` | **NEXT — needs API key** |
| 7. Demand scoring (δ_i) | `score_demand_friction.py` | `data/processed/demand_friction.csv` | After step 6 |
| 8. DiD regressions | `analysis/did_v3.R` | — | After steps 6–7 |

---

## Current File Structure

```
scripts/
  build_firm_universe.py      # DONE — do not re-run (SEC rate limits)
  collect_10k_text.py         # DONE — do not re-run (SEC rate limits)
  build_financial_panel.py    # DONE — do not run without --tickers
  build_rpo_quarterly.py      # DONE — builds RPO quarterly from EDGAR companyfacts
  score_literature_rubric.py  # NEXT — run with ANTHROPIC_API_KEY
  score_demand_friction.py    # After supply scoring

analysis/
  did_v3.R                    # Two-way FE DiD + WCB; update after scoring

prompts/
  literature_rubric_system.txt  # Supply-side system prompt (E0/E1/E2 framework)
  demand_friction_system.txt    # Demand-side system prompt (δ_switch/δ_error/δ_data)

data/
  raw/
    firm_universe.csv           # 248 firms: ticker, CIK, SIC, exchange
  processed/
    financial_panel.csv         # 223 firms, 5,493 obs (quarterly, 2019Q1–2025Q4)
    rpo_quarterly.csv           # RPO data: 197/223 firms, hierarchical tag fallback
    panel.csv                   # Base panel: financial_panel + rpo merged
    lit_scores.csv              # Supply scores — MUST BE RE-RUN via score_literature_rubric.py
    extraction_qa.json          # 10-K extraction quality log

text_data/10k_extracts/         # Extracted 10-K texts (gitignored) — 239 .txt files
notebooks/
  thesis_notebook.ipynb         # ALL figures go here, nowhere else
```

---

## Repository Rules (Always Follow)

### Scoring Rules
- **Source text:** Always score from `text_data/10k_extracts/{TICKER}.txt` — never from web research, company websites, or post-shock filings
- **Pre-shock only:** 10-K filings dated < 2022-11-01; post-shock text is contaminated by firms' own AI adaptations
- **Model:** `claude-sonnet-4-6` — do not change without updating both scripts
- **Supply score formula:** recomputed from task counts in pipeline (see above) — model's own `normalized_score` is for comparison only; pipeline's computed value is authoritative
- **Demand score:** each sub-score must be in `{0.0, 0.1, ..., 1.0}`; composite is arithmetic mean

### Financial Panel Rules
- **IIIV excluded** — post-privatization accounting restatements (FY2022: −40%, FY2023: −49%) make revenue data unreliable. Hardcoded exclusion in `build_financial_panel.py`.
- **Q4 revenue** = `Annual_FY − (Q1 + Q2 + Q3)`. Negative values are discarded (tag mismatch / amended filing artifact).
- **Multiple annual filings** for same fiscal year: higher value retained (original preferred over post-restatement amendment).
- **Do not run `build_financial_panel.py` without `--tickers`** — it overwrites the full panel.

### RPO Rules
- RPO is a **balance-sheet stock** (point-in-time snapshot at period end), not a flow metric — no cumulative summing.
- Tag hierarchy: `RevenueRemainingPerformanceObligation` → `ContractWithCustomerLiability` → `CWCL_Current + CWCL_Noncurrent` → `DeferredRevenue + DeferredRevenueNoncurrent`
- For duplicate `period_end` dates (amended filings), keep latest filed.
- 26/223 firms genuinely have no reportable RPO data (pre-ASC 606 reporters, point-in-time revenue recognizers) — this is expected, not a bug.
- **RPO/Revenue ratio:** P99-winsorized; use pre-shock firm-level average as moderator if needed. Raw RPO is correlated with firm size (r≈0.6); ratio is size-independent (r≈-0.007).

### 10-K Extraction Rules
- **Source:** SEC EDGAR only (`data.sec.gov/submissions/` API)
- **Cutoff:** Filing date strictly < 2022-11-01
- **Section:** Item 1 Business Description — extract in full, no truncation
- **No fallbacks:** No Wayback Machine, no product pages, no company websites

### What Goes Where
| Item | Location | Notes |
|---|---|---|
| All figures | `notebooks/thesis_notebook.ipynb` | ONLY here — no standalone .py figure scripts |
| Firm universe | `scripts/build_firm_universe.py` | SEC EDGAR SIC 7370-7379 |
| 10-K extraction | `scripts/collect_10k_text.py` | Pre-shock only |
| Financial panel | `scripts/build_financial_panel.py` | XBRL quarterly revenue |
| RPO panel | `scripts/build_rpo_quarterly.py` | XBRL balance sheet stock |
| Supply scoring | `scripts/score_literature_rubric.py` + `prompts/literature_rubric_system.txt` | |
| Demand scoring | `scripts/score_demand_friction.py` + `prompts/demand_friction_system.txt` | |
| R regressions | `analysis/did_v3.R` | fixest + fwildclusterboot |
| LaTeX manuscript | Overleaf only | `thesis.tex` is gitignored — NEVER commit it |

### Git Rules
- Commits authored solely by Hakan — no co-author attributions
- Never commit: `thesis.tex`, `data/raw/`, `text_data/`, `logs/`, API keys

---

## Key Numbers

| Parameter | Value |
|---|---|
| Firms in universe | 248 |
| 10-K extractions | 239/248 |
| Firms in financial panel | 223 (after IIIV exclusion + data quality filters) |
| Panel observations | 5,493 |
| Firms with RPO data | 197/223 |
| Panel period | 2019Q1–2025Q4 |
| Shock date | 2022Q4 (`period_end ≥ 2022-10-01`) |
| Extraction cutoff | < 2022-11-01 (pre-shock) |

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

---

## Running the Pipeline

```bash
# Steps 1–5 already done. Do not re-run.

# Step 6: Supply-side scoring (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python3 scripts/score_literature_rubric.py --test        # validate calibration anchors first
python3 scripts/score_literature_rubric.py --skip-existing  # full run

# Step 7: Demand-side scoring (requires ANTHROPIC_API_KEY)
python3 scripts/score_demand_friction.py --test          # validate calibration anchors first
python3 scripts/score_demand_friction.py --skip-existing # full run

# Step 8: Regressions
Rscript analysis/did_v3.R
```

---

## Theoretical Context

**Second-order displacement:** AI displacing the software products that had themselves displaced workers from cognitive tasks in the 1990s–2000s. The Acemoglu–Restrepo (2022) task-substitution framework applies one level up the value chain: B2B software firms are *task externalizers* who face product obsolescence when LLMs can perform the same tasks at near-zero marginal cost.

**Supply side (ρ_i):** Eloundou et al. (2024) "GPTs are GPTs" task exposure framework, applied to product-level tasks instead of worker tasks. A product whose core task bundle is E1/E2-exposed faces the same substitution pressure as a worker in an E1/E2 occupation.

**Demand side (δ_i):** Even when a product is highly replicable, customers may not switch because of: (1) switching costs (Farrell & Klemperer 2007) — contractual, technical, organizational, and data portability barriers; (2) error costs (Agrawal, Gans & Goldfarb 2018) — regulatory, legal, and irreversibility consequences when AI makes mistakes; (3) data/network moats (Katz & Shapiro 1985; Rochet & Tirole 2003) — proprietary data accumulation and network effects that alternatives cannot replicate.

**Interaction hypothesis:** β₂ < 0 would mean high-replicability firms with low demand friction suffer most. High demand friction (δ_i → 1) buffers against the negative replicability effect.
