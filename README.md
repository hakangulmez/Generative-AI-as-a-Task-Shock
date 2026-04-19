# Generative AI as a Task Shock
### Product-Level LLM Substitution in B2B Software Markets

**Master's Thesis** · TUM School of Management · M.Sc. Management & Technology  
**Author:** Hakan Zeki Gülmez · **Supervisor:** Prof. Dr. Helmut Farbmacher · **Submission:** October 2026

---

## Research Question

Do B2B software firms whose products are more replicable by LLMs experience worse revenue growth after the ChatGPT shock (2022Q4)? And does demand-side friction moderate that effect?

**Main estimating equation:**
```
ln(Revenue_it) = α_i + δ_t + β · (Post_t × ρ_i) + ε_it
```

**Extended specification with demand-side moderator:**
```
ln(Revenue_it) = α_i + δ_t + β₁·(Post_t × ρ_i) + β₂·(Post_t × ρ_i × δ_i) + ε_it
```

Two-way fixed effects DiD. `ρ_i` = product-level LLM replicability score ∈ [1, 100]. `δ_i` = demand-side friction score ∈ [0, 1]. `Post_t = 1` for 2022Q4 onward (`period_end ≥ 2022-10-01`). Inference via wild cluster bootstrap (WCB), clustered at firm level.

---

## Pipeline Status

| Step | Script | Output | Status |
|------|--------|--------|--------|
| 1. Firm universe | `build_firm_universe.py` | `data/raw/firm_universe.csv` | Done — 248 firms |
| 2. 10-K Item 1 extraction | `collect_10k_text.py` | `text_data/10k_extracts/*.txt` | Done — 239/248 firms |
| 3. Financial panel | `build_financial_panel.py` | `data/processed/financial_panel.csv` | Done — 223 firms, 5,493 obs |
| 4. RPO panel | `build_rpo_quarterly.py` | `data/processed/rpo_quarterly.csv` | Done — 197/223 firms |
| 5. Base panel merge | — | `data/processed/panel.csv` | Done — 223 firms, 5,493 obs |
| 6. Supply scoring (ρ_i) | `score_literature_rubric.py` | `data/processed/lit_scores.csv` | **Next — needs API key** |
| 7. Demand scoring (δ_i) | `score_demand_friction.py` | `data/processed/demand_friction.csv` | After step 6 |
| 8. DiD regressions | `analysis/did_v3.R` | — | After steps 6–7 |

---

## Repository Structure

```
scripts/
  build_firm_universe.py      # SEC EDGAR firm universe (SIC 7370-7379, NYSE/Nasdaq)
  collect_10k_text.py         # 10-K Item 1 extraction — pre-shock filings only
  build_financial_panel.py    # Quarterly revenue panel from EDGAR XBRL companyfacts
  build_rpo_quarterly.py      # Quarterly RPO panel from EDGAR XBRL (balance-sheet stock)
  score_literature_rubric.py  # Supply-side scoring: E0/E1/E2 task exposure via Claude API
  score_demand_friction.py    # Demand-side scoring: δ_switch/δ_error/δ_data via Claude API

analysis/
  did_v3.R                    # Two-way FE DiD + WCB inference

prompts/
  literature_rubric_system.txt  # Supply-side system prompt (LLM replicability)
  demand_friction_system.txt    # Demand-side system prompt (customer friction)

data/
  raw/
    firm_universe.csv           # 248 firms: ticker, CIK, SIC, exchange
  processed/
    financial_panel.csv         # 223 firms, 5,493 obs (quarterly revenue + financials)
    rpo_quarterly.csv           # 197/223 firms, quarterly RPO snapshots
    panel.csv                   # Base panel: financial_panel + rpo merged
    lit_scores.csv              # Supply scores (ρ_i) — from score_literature_rubric.py
    extraction_qa.json          # 10-K extraction quality log

notebooks/
  thesis_notebook.ipynb         # All figures (only place for figure code)

text_data/10k_extracts/         # Extracted 10-K texts (gitignored)
logs/                           # Pipeline logs (gitignored)
```

---

## Data Sources

- **Firm universe:** SEC EDGAR submissions API — all SIC 7370–7379 firms on NYSE/Nasdaq with ≥6 pre-shock XBRL quarters
- **Financial data:** SEC EDGAR companyfacts XBRL API — quarterly revenue, gross profit, R&D, SG&A
- **RPO data:** SEC EDGAR companyfacts XBRL API — balance-sheet stock of remaining performance obligations
- **10-K text:** SEC EDGAR full-text search API — Item 1 Business Description, pre-shock filings only (< 2022-11-01)
- **No external data providers** — everything reproducible from public SEC APIs

---

## Treatment Variable Construction

### Supply Side: ρ_i (LLM Replicability Score ∈ [1, 100])

Constructed by scoring each firm's pre-shock 10-K Item 1 Business Description using the **Literature Rubric** — an adaptation of Eloundou et al. (2024) applied one level up the value chain (product tasks, not worker tasks).

**Scoring prompt:** `prompts/literature_rubric_system.txt`

1. Decompose the product into 6–10 customer-facing tasks
2. Classify each task:
   - **E1** — direct LLM exposure: text output, document processing, communication, matching/routing/ranking on text, template generation
   - **E2** — LLM + standard tools: requires database/API access before generating language output
   - **E0** — no meaningful LLM exposure: real-time data streams, physical hardware, sub-second latency SLA, deep proprietary integration as core moat
3. Assign integration depth penalty: `0` (no moat) / `−1` (moderate moat) / `−2` (strong moat)
4. Compute:
   ```
   raw_exposure      = (E1_count + 0.5 × E2_count) / n_tasks
   adjusted_exposure = max(0.0, raw_exposure + penalty / 10)
   normalized_score  = round(adjusted_exposure × 99 + 1, 1)
   ```

**Calibration anchors:** ZS≈10 · DDOG≈16 · CRWD≈22 · VEEV≈38 · PAYC≈45 · HUBS≈64 · LPSN≈72 · EGAN≈80 · ZIP≈88

---

### Demand Side: δ_i (Customer Friction Score ∈ [0, 1])

Constructed by scoring each firm's pre-shock 10-K Item 1 text using the **Demand Friction Rubric** — grounded in three bodies of literature.

**Scoring prompt:** `prompts/demand_friction_system.txt`

Three sub-scores (each ∈ {0.0, 0.1, ..., 1.0}):

**δ_switch — Switching Cost** (Farrell & Klemperer 2007)  
How difficult and costly is it for an existing customer to replace this product? Encompasses contractual lock-in (multi-year agreements), technical integration depth (embedded connectors to core systems), organizational switching costs (retraining, change management), and data portability barriers.

**δ_error — Error Cost** (Agrawal, Gans & Goldfarb 2018)  
How costly or catastrophic is it when the product (or an AI replacement) makes a mistake? High when errors carry regulatory penalties, legal liability, patient safety risk, or irreversible financial consequences — creating mandatory human oversight that slows AI substitution regardless of capability.

**δ_data — Data/Network Moat** (Katz & Shapiro 1985; Rochet & Tirole 2003)  
Does this product have a proprietary data asset or network effect that a new AI-native competitor cannot replicate? Qualifies only for exclusive, volume-accumulated data with feedback loops (Verisk's 30B+ insurance records, ZipRecruiter's 50M jobseeker network) — not public data or customer-provided data.

```
δ_composite = (δ_switch + δ_error + δ_data) / 3   ∈ [0, 1]
```

---

## Financial Panel Construction

**Source:** SEC EDGAR companyfacts XBRL — `us-gaap/Revenues` and related tags.

**Quarterly revenue handling:**
- Q1–Q3: directly from XBRL 10-Q filings
- Q4: computed as `Annual_FY − (Q1 + Q2 + Q3)` — negative values discarded (tag mismatch / amended filing artifact)
- When multiple annual filings exist for a fiscal year, the higher value is retained (original filing preferred over post-restatement amendments)

**Sample construction:**
- SIC 7370–7379 firms on NYSE/Nasdaq (software & data processing)
- ≥ 6 pre-shock quarters of XBRL revenue data required
- IIIV excluded — post-privatization accounting restatements (FY2022: −40%, FY2023: −49%) make revenue data unreliable

**Panel period:** 2019Q1–2025Q4 · 223 firms · 5,493 observations

---

## RPO Panel Construction

**Source:** SEC EDGAR companyfacts XBRL — balance-sheet stock at each 10-Q/10-K period end.

**Tag hierarchy (fallback chain):**
1. `RevenueRemainingPerformanceObligation` (primary ASC 606 tag)
2. `ContractWithCustomerLiability` (alternative ASC 606 tag)
3. `ContractWithCustomerLiabilityCurrent` + `ContractWithCustomerLiabilityNoncurrent` (summed)
4. `DeferredRevenue` + `DeferredRevenueNoncurrent` (pre-ASC 606 proxy)

**Coverage:** 197/223 firms (88%). 26 firms genuinely have no reportable RPO — point-in-time revenue recognizers, pre-ASC 606 reporters, or pure transaction businesses.

**Usage as moderator:** Pre-shock RPO/Revenue ratio (P99-winsorized) as cross-sectional switching cost proxy. Raw RPO correlates with firm size (r≈0.6); ratio is size-independent (r≈−0.007 with log revenue).

---

## Running the Pipeline

```bash
# Steps 1–5: Already done. Do not re-run (SEC rate limits, gitignored outputs).

# Step 6: Supply-side scoring (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python3 scripts/score_literature_rubric.py --test          # validate 5 calibration anchors
python3 scripts/score_literature_rubric.py --skip-existing # full run (239 firms)

# Step 7: Demand-side scoring (requires ANTHROPIC_API_KEY)
python3 scripts/score_demand_friction.py --test            # validate 7 calibration anchors
python3 scripts/score_demand_friction.py --skip-existing   # full run (239 firms)

# Step 8: Regressions
Rscript analysis/did_v3.R
```

---

## Key Design Choices

- **Pre-shock scoring only:** 10-K filings before 2022-11-01 ensure treatment is not contaminated by post-shock firm adaptation to AI
- **IIIV excluded:** Post-privatization accounting restatements make historical revenue data unreliable
- **Q4 revenue:** Computed as `Annual_FY − (Q1 + Q2 + Q3)` from XBRL; negative values discarded (tag mismatch)
- **Two prompts, two dimensions:** Supply side (can the product be replicated?) and demand side (will customers switch if it can?) are scored separately to allow independent variation
- **Interaction hypothesis:** β₂ < 0 would confirm that high-friction firms are buffered from the negative replicability effect — customers don't substitute even when they could

---

## Notes

- `thesis.tex` is gitignored — manuscript lives on Overleaf
- `text_data/`, `logs/`, `data/raw/` are gitignored
- All scoring uses pre-shock 10-K Item 1 text — do not re-score with later filings
- Figures are generated in `notebooks/thesis_notebook.ipynb` — no standalone figure scripts
