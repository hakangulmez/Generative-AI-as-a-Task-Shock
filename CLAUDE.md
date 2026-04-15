# CLAUDE.md — AI Assistant Instructions

This file tells Claude (or any AI assistant) exactly what this repository is and how to work within it correctly.

---

## What This Repo Is

**Master's thesis empirical pipeline** for:
> *"Generative AI as a Task Shock: Product-Level LLM Substitution in B2B Software Markets"*
> Hakan Zeki Gülmez · TUM School of Management · Supervisor: Prof. Dr. Helmut Farbmacher · 2026

The thesis measures whether B2B software firms with higher **product-level LLM replicability** experienced worse revenue growth after the ChatGPT shock (2022Q4).

**Core result:** β = −0.604, WCB p = 0.003*** (SME <$200M, Literature Rubric, n=94)

---

## Key Concepts (Understand Before Helping)

### The Treatment Variable
`ρ_i^lit` — the **Literature Rubric score** ∈ [1, 100]. This is the PRIMARY treatment variable. It measures the fraction of a firm's product task bundle that an LLM can replicate at near-zero cost. Constructed pre-shock (10-K filings before 2022-11-01) using Claude Opus scoring a 10-criterion rubric.

**Not to be confused with:**
- `early_ai` / `agentic_ai` — these are the holistic LLM Judge scores (used as robustness check, NOT primary)
- `replicability_score`, `contrast_score` — SBERT-based scores (robustness only)
- O*NET similarity scores — additional robustness measure

### The Shock
ChatGPT public release = November 30, 2022. `Post_t = 1` for 2022Q4 onward.

### The Estimating Equation
```
ln(Revenue_it) = α_i + δ_t + β · (Post_t × ρ_i^lit) + ε_it
```
Two-way fixed effects DiD. Inference via wild cluster bootstrap (WCB), clustered at firm level.

### The Mechanism
**Quantity channel, not pricing channel.** Gross margin effects are insignificant (p > 0.4). Customers substitute away entirely rather than renegotiating prices. Confirmed by R&D intensity test: β = +0.080, p = 0.024 (affected firms increase defensive R&D).

---

## Repository Rules (Always Follow)

### What Goes Where
| Item | Location | Notes |
|---|---|---|
| All figure code | `notebooks/thesis_notebook.ipynb` | ONLY here, no separate .py scripts for figures |
| 10-K extraction | `scripts/collect_10k_text.py` | SEC EDGAR only, no Wayback/product pages |
| Scoring | `scripts/score_rubric_contrast.py` | Literature rubric is primary |
| R regressions | `analysis/did_main.R`, `analysis/wcb_rubric.R` | |
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
- The text thesis says "~3,000 characters" but this refers to the scoring input window, not the extraction limit. Extract the full Item 1.

### Scoring Rules
- Primary score = **Literature Rubric** (`ρ_i^lit`), not LLM Judge holistic
- Scoring model = Claude Opus (not Sonnet, not Haiku)
- Score must be **pre-shock**: constructed from pre-2022Q4 filings only
- Do not re-score with post-shock text

---

## File Index

```
scripts/collect_10k_text.py       → SEC EDGAR 10-K extractor
scripts/score_rubric_contrast.py  → SBERT contrast scoring (Literature Rubric)
scripts/score_rubric_sentlevel.py → Sentence-level rubric scoring
scripts/score_onet_similarity.py  → O*NET similarity (robustness)
scripts/build_firm_texts.py       → Aggregate texts to CSV

analysis/did_main.R               → Main DiD regressions (Tables 1–3)
analysis/wcb_rubric.R             → Wild cluster bootstrap p-values

notebooks/thesis_notebook.ipynb   → ALL figures (Fig 1–9), data QA, descriptives

data/raw/firm_universe.csv        → 143 firms: ticker, CIK, company_name, meets_filters
data/processed/                   → Generated outputs (gitignored)
text_data/10k_extracts/           → Extracted 10-K texts (gitignored)
literature_rubric.json            → 10-criterion rubric definition
```

---

## Key Numbers (Do Not Change Without Good Reason)

| Parameter | Value | Why |
|---|---|---|
| Firms in universe | 143 | Full sample from SIC 7370–7379 |
| Primary sample | n=94 | SME <$200M pre-shock revenue |
| Extended sample | n=116 | SME <$500M |
| Shock date | 2022Q4 (2022-11-30) | ChatGPT release |
| Extraction cutoff | 2022-11-01 | Pre-shock |
| Panel period | 2020Q1–2025Q4 | 24 quarters max |
| Primary β | −0.604 | WCB p=0.003 |
| Construct validity r | 0.895 | LLM Judge vs Lit Rubric |
| Three-period: Early | −0.451 (p=0.022) | 2022Q4–2023Q4 |
| Three-period: Advanced | −0.686 (p=0.008) | 2024Q1+ |
| Intensification | 1.52× | Advanced / Early |

---

## Robustness Table Structure (tab:robust)

Four panels, do not restructure:
- **Panel A:** Alternative samples (full scored, <$500M, >$200M excluded)
- **Panel B:** Alternative treatment measures (LLM Judge, contrast score, O*NET)  
- **Panel C:** Specification checks (no controls, time trends, quarterly FE)
- **Panel D:** Placebo and falsification (pre-period, gross margin)

R&D intensity sits exclusively in **Table 3** (Mechanism Outcomes), not in the robustness table.

---

## Common Mistakes to Avoid

1. **Don't use `early_ai` as primary treatment** — it's holistic LLM Judge, not the literature rubric
2. **Don't truncate Item 1 extraction** — extract the full section
3. **Don't commit `thesis.tex`** — Overleaf only
4. **Don't put figure code in .py scripts** — everything in `thesis_notebook.ipynb`
5. **Don't add Wayback/product page fallback** to extraction — SEC EDGAR only
6. **Don't use post-2022Q4 10-K filings** for scoring — post-shock text is contaminated
7. **Don't confuse quantity channel with pricing channel** — gross margin insignificance is the key result proving quantity channel

---

## If Asked to Write New Code

- Python: follow existing style in `scripts/`
- R: use `fixest` for fixed effects, `fwildclusterboot` for WCB
- Figures: add to `thesis_notebook.ipynb` with filename `fig{N}_{description}.png`
- Naming: `fig1_` through `fig9_` are taken; new figures start at `fig10_`

---

## Theoretical Context (for Writing Tasks)

**Second-order displacement:** AI displacing the software products that had themselves displaced workers from cognitive tasks in the 1990s–2000s. The Acemoglu–Restrepo (2022) task-substitution framework applies one level up the value chain: B2B software firms are *task externalizers* who face product obsolescence when LLMs can perform the same tasks at near-zero marginal cost.

**Size heterogeneity mechanism:** Switching costs (Farrell & Klemperer 2007). Enterprise customers face prohibitively high switching costs (multi-year contracts, workflow dependencies, compliance), insulating large-vendor relationships. SME customers can substitute point solutions within a contract cycle, making them more price-responsive to LLM alternatives.

**Entry barrier implication (for Discussion):** Post-shock, the entry barrier for competing with high-replicability software products has collapsed. Any developer with API access can replicate what previously required years of engineering. This creates a new competitive dynamic distinct from classical horizontal competition between software firms — substitution now comes from general-purpose AI that any entrepreneur can deploy.
