# Generative AI as a Task Shock
### Product-Level LLM Substitution in B2B Software Markets

**Master's Thesis** · TUM School of Management · M.Sc. Management & Technology  
**Author:** Hakan Zeki Gülmez · **Supervisor:** Prof. Dr. Helmut Farbmacher · **Submission:** October 2026

---

## Overview

This repository contains the full empirical pipeline for the master's thesis *"Generative AI as a Task Shock: Product-Level LLM Substitution in B2B Software Markets."*

The central research question: did the ChatGPT shock (November 2022) generate differential revenue outcomes across B2B software firms according to their **product-level LLM replicability** — the degree to which a firm's product performs cognitive tasks that a general-purpose LLM can now replicate at near-zero marginal cost?

**Core finding:** Among SME-segment B2B software firms (pre-shock revenue <$200M, n=94), a one-unit increase in normalized replicability is associated with **60 log-point lower revenue growth** after the shock (β = −0.604, WCB p = 0.003). The effect intensifies 1.52× from the early AI era (2022Q4–2023Q4) to the advanced AI era (2024Q1+), consistent with an expanding automation frontier.

---

## Key Results

| Specification | β | WCB p | N |
|---|---|---|---|
| Primary (SME <$200M, Lit Rubric) | −0.604 | 0.003*** | 94 |
| Extended (SME <$500M, Lit Rubric) | −0.541 | 0.002*** | 116 |
| Quartile (Q75 vs Q25) | −0.286 | 0.005*** | 94 |
| Pre-period placebo | +0.031 | 0.157 | 94 |
| Early AI era (2022Q4–2023Q4) | −0.451 | 0.022* | 94 |
| Advanced AI era (2024Q1+) | −0.686 | 0.008** | 94 |

Gross margin effects are uniformly insignificant (p > 0.4), confirming the **quantity channel**: customers substitute away from high-replicability products entirely rather than renegotiating prices.

---

## Repository Structure

```
thesis-ai-task-shock/
│
├── scripts/
│   ├── collect_10k_text.py       # SEC EDGAR Item 1 extraction (pre-shock, cutoff 2022-11-01)
│   ├── score_rubric_contrast.py  # SBERT contrast scoring (Literature Rubric)
│   ├── score_rubric_sentlevel.py # Sentence-level rubric scoring
│   ├── score_onet_similarity.py  # O*NET similarity scoring (robustness)
│   └── build_firm_texts.py       # Combine extracted texts into panel CSV
│
├── analysis/
│   ├── did_main.R                # Primary DiD regressions + event study
│   └── wcb_rubric.R              # Wild cluster bootstrap inference
│
├── notebooks/
│   └── thesis_notebook.ipynb     # All figures (Figures 1–9)
│
├── data/
│   ├── raw/
│   │   └── firm_universe.csv     # 143 firms: ticker, CIK, company_name
│   └── processed/                # Generated outputs (gitignored)
│
├── figures/                      # All thesis figures (PNG)
│
├── literature_rubric.json        # Ten-criterion rubric definition
├── requirements.txt
├── CLAUDE.md                     # AI assistant instructions for this repo
└── README.md
```

> **Note:** `thesis.tex` and the LaTeX manuscript are managed exclusively via Overleaf and are not tracked in this repository.

---

## Data & Measurement Pipeline

### Step 1 — Universe Construction
143 NYSE/Nasdaq-listed B2B software firms (SIC 7370–7379), IPO before 2020Q1, with quarterly SEC EDGAR financials available.

### Step 2 — 10-K Text Extraction
```bash
python3 scripts/collect_10k_text.py
```
Extracts Item 1 Business Description from each firm's most recent 10-K or 20-F filed **before November 1, 2022** (pre-shock cutoff). Source: SEC EDGAR Submissions API (`data.sec.gov/submissions/CIK{cik}.json`). Output: `text_data/10k_extracts/{TICKER}.txt`.

**Critical design choice:** Pre-shock extraction ensures the replicability score reflects pre-existing product architecture, not post-shock AI repositioning. 10-K filings after 2022Q4 increasingly contain strategic language ("AI-powered", "LLM-enabled") that would contaminate the measure.

### Step 3 — Replicability Scoring (Literature Rubric)
The primary treatment variable ρᵢˡⁱᵗ ∈ [1, 100] is constructed using an LLM-as-judge procedure grounded in the task-based automation literature (Acemoglu & Restrepo 2022; Eloundou et al. 2024).

**Ten scoring criteria** (`literature_rubric.json`):

*Displacement dimensions (LLMs substitute):*
- C1: Text generation / summarization (weight 3)
- C2: Customer communication workflows (weight 3)
- C3: Information synthesis / retrieval (weight 2)
- C4: Routine codifiable cognitive tasks (weight 2)
- C5: Classification / categorization (weight 2)

*Resistance dimensions (LLMs cannot substitute):*
- C6: Real-time data feed dependency (weight 3)
- C7: Physical world / hardware integration (weight 3)
- C8: Complex professional judgment (weight 2)
- C9: Specialized tool / system integration (weight 2)
- C10: Real-time adaptive response (weight 2)

Each criterion scored −1 / 0 / +1 by Claude Opus; raw score normalized to [1, 100].

**Construct validity:** Independent holistic LLM judge scores correlate r = 0.895 with the literature rubric (Figure 2).

### Step 4 — Financial Panel
Quarterly revenue and gross margin from SEC EDGAR XBRL inline filings (10-Q, 10-K). XBRL tags: `Revenues` or `RevenueFromContractWithCustomerExcludingAssessedTax` for revenue; `GrossProfit` for gross margin. Panel: 2020Q1–2025Q4, maximum 24 quarterly observations per firm.

### Step 5 — DiD Estimation
```bash
Rscript analysis/did_main.R
Rscript analysis/wcb_rubric.R
```
Two-way fixed effects DiD with wild cluster bootstrap inference (Cameron et al. 2008). Cluster at firm level.

---

## Empirical Strategy

**Estimating equation:**

```
ln(Revenue_it) = α_i + δ_t + β · (Post_t × ρ_i^lit) + ε_it
```

- `α_i` = firm fixed effects (absorb time-invariant firm characteristics)
- `δ_t` = quarter fixed effects (absorb aggregate macro shocks)
- `Post_t` = 1 for 2022Q4 onward (ChatGPT release)
- `ρ_i^lit` = pre-shock literature rubric replicability score
- Standard errors clustered at firm level; inference via wild cluster bootstrap

**Identification:** Parallel pre-trends confirmed via event study across all 12 pre-shock quarters (Figure 3). Pre-period placebo test: β = +0.031, p = 0.157.

---

## Reproducing the Results

### Requirements
```bash
pip install -r requirements.txt   # Python: pandas, sentence-transformers, anthropic
# R packages: fixest, fwildclusterboot, tidyverse, ggplot2
```

### Full pipeline
```bash
# 1. Extract 10-K texts (requires SEC EDGAR access, ~30 min for 143 firms)
python3 scripts/collect_10k_text.py

# 2. Score firms (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=your_key_here
python3 scripts/score_rubric_contrast.py

# 3. Build financial panel (requires SEC EDGAR access)
# See thesis_notebook.ipynb — Section: Financial Panel

# 4. Run regressions
Rscript analysis/did_main.R
Rscript analysis/wcb_rubric.R

# 5. Generate all figures
jupyter nbconvert --to notebook --execute notebooks/thesis_notebook.ipynb
```

### Quick start (pre-computed data)
Pre-computed scores and panel data are available at the [Streamlit dashboard](https://ai-task-replicability.streamlit.app):

```bash
# With pre-computed data in data/processed/:
Rscript analysis/did_main.R      # Reproduces Tables 1–3
Rscript analysis/wcb_rubric.R    # Reproduces WCB p-values
```

---

## Descriptive Evidence

| Firm | Score | Post-shock revenue growth |
|---|---|---|
| ZipRecruiter (ZIP) | 82 | −35% |
| LivePerson (LPSN) | 72 | −62% |
| eGain (EGAN) | 78 | −41% |
| Cloudflare (NET) | 18 | +127% |
| Datadog (DDOG) | 18 | +89% |
| MongoDB (MDB) | 18 | +76% |

---

## Related Resources

- **Live dashboard:** [ai-task-replicability.streamlit.app](https://ai-task-replicability.streamlit.app)
- **Dashboard repo:** [github.com/hakangulmez/ai-task-replicability-explorer](https://github.com/hakangulmez/ai-task-replicability-explorer)
- **SEC EDGAR:** [data.sec.gov](https://data.sec.gov)

---

## Citation

```bibtex
@mastersthesis{gulmez2026,
  author  = {Gülmez, Hakan Zeki},
  title   = {Generative {AI} as a Task Shock: Product-Level {LLM} Substitution in {B2B} Software Markets},
  school  = {TUM School of Management, Technical University of Munich},
  year    = {2026},
  advisor = {Farbmacher, Helmut}
}
```

---

## License

MIT License. Data sourced from public SEC EDGAR filings.
