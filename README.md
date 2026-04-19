# Generative AI as a Task Shock
### Product-Level LLM Substitution in B2B Software Markets

**Master's Thesis** · TUM School of Management · M.Sc. Management & Technology  
**Author:** Hakan Zeki Gülmez · **Supervisor:** Prof. Dr. Helmut Farbmacher · **Submission:** October 2026

---

## Research Question

Do B2B software firms whose products are more replicable by LLMs experience worse revenue growth after the ChatGPT shock (2022Q4)?

**Estimating equation:**
```
ln(Revenue_it) = α_i + δ_t + β · (Post_t × ρ_i^lit) + ε_it
```

Two-way fixed effects DiD. `ρ_i^lit` = product-level LLM replicability score ∈ [1, 100]. `Post_t = 1` for 2022Q4 onward. Inference via wild cluster bootstrap (WCB), clustered at firm level.

---

## Pipeline Status

| Step | Script | Status |
|------|--------|--------|
| 1. Build firm universe | `scripts/build_firm_universe.py` | Done — 248 firms |
| 2. Extract 10-K Item 1 text | `scripts/collect_10k_text.py` | Done — 239/248 firms |
| 3. Build financial panel | `scripts/build_financial_panel.py` | Done — 223 firms, 5,493 obs |
| 4. Score firms (Literature Rubric) | `scripts/score_literature_rubric.py` | **Next step** |
| 5. Build master panel | `scripts/build_master_panel.py` | After scoring |
| 6. DiD regressions | `analysis/did_v3.R` | After master panel |

---

## Repository Structure

```
scripts/
  build_firm_universe.py      # SEC EDGAR firm universe (SIC 7370-7379)
  collect_10k_text.py         # 10-K Item 1 extraction (pre-2022-11-01 filings)
  build_financial_panel.py    # Quarterly revenue panel from EDGAR XBRL
  score_literature_rubric.py  # E0/E1/E2 task exposure scoring via Claude API
  build_master_panel.py       # Merge financials + scores → master panel

analysis/
  did_v3.R                    # Two-way FE DiD + WCB inference

prompts/
  literature_rubric_system.txt  # System prompt for Claude scoring

data/
  raw/
    firm_universe.csv           # 248 firms: ticker, CIK, SIC, exchange
  processed/
    financial_panel.csv         # 223 firms, 5,493 obs (quarterly, 2019Q1–2025Q4)
    extraction_qa.json          # 10-K extraction quality log

text_data/10k_extracts/         # Extracted 10-K texts (gitignored)
logs/                           # Pipeline logs (gitignored)
```

---

## Data Sources

- **Firm universe:** SEC EDGAR submissions API — all SIC 7370–7379 firms on NYSE/Nasdaq with ≥6 pre-shock XBRL quarters
- **Financial data:** SEC EDGAR companyfacts XBRL API — quarterly revenue, gross profit, R&D, SG&A
- **10-K text:** SEC EDGAR full-text search — Item 1 Business Description, pre-shock filings only (< 2022-11-01)
- **No external data providers** — everything reproducible from public SEC APIs

---

## Treatment Variable Construction

`ρ_i^lit` is constructed by scoring each firm's 10-K Item 1 Business Description using the **Literature Rubric** — an adaptation of Eloundou et al. (2024) applied one level up the value chain:

1. Decompose the product into 6–10 customer-facing tasks
2. Classify each task: **E1** (direct LLM exposure), **E2** (LLM + tools), **E0** (no meaningful exposure)
3. Compute: `raw = (E1 + 0.5×E2) / n_tasks`
4. Apply integration depth penalty (0 / −1 / −2) for structural switching-cost moats
5. Normalize: `score = max(1, min(100, raw_adjusted × 99 + 1))`

Theoretical grounding: Eloundou et al. (2024), Acemoglu & Restrepo (2022), Brynjolfsson et al. (2025).

---

## Running the Pipeline

```bash
# Step 1–2: Already done. Do not re-run (SEC rate limits, gitignored outputs).

# Step 3: Rebuild financial panel (if needed)
python3 scripts/build_financial_panel.py

# Step 4: Score firms (requires ANTHROPIC_API_KEY)
python3 scripts/score_literature_rubric.py --skip-existing

# Step 5: Build master panel
python3 scripts/build_master_panel.py

# Step 6: Run regressions
Rscript analysis/did_v3.R
```

---

## Key Design Choices

- **Pre-shock scoring only:** 10-K filings before 2022-11-01 ensure treatment is not contaminated by post-shock firm adaptation
- **IIIV excluded:** Post-privatization accounting restatements (FY2022: −40%, FY2023: −49%) make the firm's historical revenue data unreliable
- **Q4 revenue:** Computed as `Annual_FY − (Q1 + Q2 + Q3)` from XBRL; negative values discarded (tag mismatch / amended filing artifacts)
- **Amended 10-K handling:** When multiple annual filings exist for a fiscal year, the higher value is retained (original filing preferred over post-restatement amendments)

---

## Notes

- `thesis.tex` is gitignored — manuscript lives on Overleaf
- `text_data/`, `logs/`, `data/raw/` are gitignored
- All scoring uses pre-shock 10-K Item 1 text — do not re-score with later filings
