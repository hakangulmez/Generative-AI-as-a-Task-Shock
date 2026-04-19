# CLAUDE.md — AI Assistant Instructions

This file tells Claude (or any AI assistant) exactly what this repository is and how to work within it correctly.

---

## What This Repo Is

**Master's thesis empirical pipeline** for:
> *"Generative AI as a Task Shock: Product-Level LLM Substitution in B2B Software Markets"*
> Hakan Zeki Gülmez · TUM School of Management · Supervisor: Prof. Dr. Helmut Farbmacher · 2026

The thesis measures whether B2B software firms with higher **product-level LLM replicability** experienced worse revenue growth after the ChatGPT shock (2022Q4).

**Current pipeline state:** Financial panel is built and clean. Next step is scoring firms with the Literature Rubric.

---

## Key Concepts (Understand Before Helping)

### The Treatment Variable
`ρ_i^lit` — the **Literature Rubric score** ∈ [1, 100]. This is the PRIMARY treatment variable. It measures the fraction of a firm's product task bundle that an LLM can replicate at near-zero cost. Constructed from pre-shock 10-K filings (before 2022-11-01) using the E0/E1/E2 task exposure framework.

**Score formula:**
```
raw_exposure      = (E1_count + 0.5 × E2_count) / n_tasks
adjusted_exposure = max(0.0, raw_exposure + integration_depth_penalty / 10)
normalized_score  = round(adjusted_exposure × 99 + 1, 1)   ∈ [1, 100]
```

### The Task Exposure Framework
Applied at the product level (one level up from Eloundou et al. 2024 who classify worker tasks):
- **E1** — direct LLM exposure: text output, document processing, communication, structured queries
- **E2** — LLM + standard tools: needs database/API access but output is language-based
- **E0** — no meaningful LLM exposure: real-time data streams, hardware, sub-second latency, deep proprietary integration

### Integration Depth Penalty
- **0** — no moat, customers can switch easily
- **−1** — moderate moat (embedded in workflows, 6–18 month switch)
- **−2** — strong moat (the integration IS the product — core banking, EDA, pharma regulatory cloud)

### The Shock
ChatGPT public release = November 30, 2022. `Post_t = 1` for 2022Q4 onward (period_end ≥ 2022-10-01).

### The Estimating Equation
```
ln(Revenue_it) = α_i + δ_t + β · (Post_t × ρ_i^lit) + ε_it
```
Two-way fixed effects DiD. Inference via wild cluster bootstrap (WCB), clustered at firm level.

---

## Current File Structure

```
scripts/
  build_firm_universe.py      # DONE — do not re-run
  collect_10k_text.py         # DONE — do not re-run
  build_financial_panel.py    # DONE — clean panel, 223 firms
  score_literature_rubric.py  # NEXT STEP — run with ANTHROPIC_API_KEY
  build_master_panel.py       # After scoring completes

analysis/
  did_v3.R                    # Main DiD regressions (run after master panel)

prompts/
  literature_rubric_system.txt  # System prompt for scoring script

data/raw/
  firm_universe.csv           # 248 firms: ticker, CIK, SIC, exchange

data/processed/
  financial_panel.csv         # 223 firms, 5,493 obs — DO NOT OVERWRITE
  extraction_qa.json          # 10-K extraction QA log
```

---

## Repository Rules (Always Follow)

### What Goes Where
| Item | Location | Notes |
|---|---|---|
| All figure code | `notebooks/thesis_notebook.ipynb` | ONLY here, no separate .py scripts for figures |
| Firm universe | `scripts/build_firm_universe.py` | SEC EDGAR SIC 7370-7379, companyfacts API |
| 10-K extraction | `scripts/collect_10k_text.py` | SEC EDGAR only, no Wayback/product pages |
| Financial panel | `scripts/build_financial_panel.py` | Quarterly revenue from companyfacts XBRL |
| Scoring | `scripts/score_literature_rubric.py` | Literature rubric is primary |
| R regressions | `analysis/did_v3.R` | fixest + fwildclusterboot |
| LaTeX manuscript | Overleaf only | `thesis.tex` is gitignored, NEVER commit it |

### Git Rules
- Commits authored solely by Hakan (no co-author attributions)
- Keep commit history clean and minimal
- Never commit: `thesis.tex`, `data/raw/`, `text_data/`, `logs/`, API keys

### 10-K Extraction Rules
- **Source:** SEC EDGAR only (`data.sec.gov/submissions/` API)
- **Cutoff:** Filing date < 2022-11-01 (pre-shock)
- **Section:** Item 1 Business Description — extract in FULL, no truncation
- **No fallbacks:** No Wayback Machine, no product pages, no company websites

### Scoring Rules
- Primary score = **Literature Rubric** (`ρ_i^lit`), E0/E1/E2 framework
- Scoring model = Claude Sonnet (claude-sonnet-4-6) — specified in score_literature_rubric.py
- Score must be **pre-shock**: constructed from pre-2022Q4 filings only
- Do not re-score with post-shock text

### Financial Panel Rules
- **IIIV is excluded** from the panel (post-privatization accounting restatements: FY2022 −40%, FY2023 −49% — data unreliable). Exclusion is hardcoded in `build_financial_panel.py`.
- Q4 revenue = `Annual_FY − (Q1 + Q2 + Q3)`. Negative Q4 values are discarded (tag mismatch / amended filing artifact).
- When multiple annual filings exist for a fiscal year, the **higher value is retained** (original filing preferred over post-restatement amendments).
- **Do not run `build_financial_panel.py` without `--tickers` unless you intend to rebuild the full panel.** It overwrites `financial_panel.csv`.

---

## Key Numbers

| Parameter | Value |
|---|---|
| Firms in universe | 248 |
| Panel observations | 5,493 |
| Firms in financial panel | 223 (after IIIV exclusion + data quality filters) |
| 10-K extractions | 239/248 |
| Shock date | 2022Q4 (2022-10-01 as period_end threshold) |
| Extraction cutoff | 2022-11-01 (pre-shock) |
| Panel period | 2019Q1–2025Q4 |

---

## Common Mistakes to Avoid

1. **Don't overwrite `financial_panel.csv`** without `--tickers` flag — it has 5,493 rows and took time to build correctly
2. **Don't truncate Item 1 extraction** — extract the full section
3. **Don't commit `thesis.tex`** — Overleaf only
4. **Don't put figure code in .py scripts** — everything in `thesis_notebook.ipynb`
5. **Don't add Wayback/product page fallback** to extraction — SEC EDGAR only
6. **Don't use post-2022Q4 10-K filings** for scoring — post-shock text is contaminated
7. **Don't re-add IIIV** — excluded intentionally due to accounting restatements
8. **Don't use alternative scoring approaches as primary** — Literature Rubric (E0/E1/E2) is the primary treatment variable

---

## If Asked to Write New Code

- Python: follow existing style in `scripts/`
- R: use `fixest` for fixed effects, `fwildclusterboot` for WCB
- Figures: add to `thesis_notebook.ipynb` with filename `fig{N}_{description}.png`

---

## Theoretical Context (for Writing Tasks)

**Second-order displacement:** AI displacing the software products that had themselves displaced workers from cognitive tasks in the 1990s–2000s. The Acemoglu–Restrepo (2022) task-substitution framework applies one level up the value chain: B2B software firms are *task externalizers* who face product obsolescence when LLMs can perform the same tasks at near-zero marginal cost.

**Size heterogeneity mechanism:** Switching costs (Farrell & Klemperer 2007). Enterprise customers face prohibitively high switching costs (multi-year contracts, workflow dependencies, compliance), insulating large-vendor relationships. SME customers can substitute point solutions within a contract cycle, making them more price-responsive to LLM alternatives.

**Entry barrier implication (for Discussion):** Post-shock, the entry barrier for competing with high-replicability software products has collapsed. Any developer with API access can replicate what previously required years of engineering.
