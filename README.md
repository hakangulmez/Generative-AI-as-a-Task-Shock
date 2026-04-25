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

**Extended specification with demand-side moderator (triple interaction):**
```
ln(Revenue_it) = α_i + δ_t + β1·(ρ_i × Post_t) + β2·(δ_i × Post_t)
                 + β3·(ρ_i × δ_i × Post_t) + ε_it
```

Two-way fixed effects DiD. `ρ_i` = product-level LLM replicability score ∈ [1, 100]. `δ_i` = demand-side friction score ∈ [0, 1]. `Post_t = 1` for 2022Q4 onward (`period_end ≥ 2022-10-01`). Inference via wild cluster bootstrap (WCB), clustered at firm level.

**Core identification:** Occupation-level AI exposure measures uniformly predict high impact for knowledge workers, yet observed firm revenue responses are highly heterogeneous. This framework resolves that tension by separating what LLMs *can* do (supply `ρ`) from what customers *will* accept (demand `δ`). High capability does not imply high impact — impact emerges when replicability coincides with low friction. Interaction coefficient `β3 > 0` would confirm friction buffers the negative effect.

**Three-tier sample (321 firms):** 256 primary B2B software firms (SIC 7370–7379) as the treatment group, 35 knowledge-intensive service firms as controls (rating agencies, index providers, education platforms, consulting), and 30 placebo firms (energy, manufacturing, payments, pharma) for falsification.

---

## Pipeline Status

All data panels are built. Phase 3 prompt engineering is complete. Phase 4 (API scoring) is the next active stage.

| Step | Script | Output | Status |
|------|--------|--------|--------|
| 1. Firm universe | `01_build_firm_universe.py` | `data/raw/firm_universe.csv` | Done — 321 firms (3 tiers) |
| 2. 10-K Item 1 extraction | `02_collect_10k_text.py` | `text_data/10k_extracts/*.txt` | Done — 321/321 firms |
| 3. Financial panel | `03_build_financial_panel.py` | `data/processed/financial_panel.csv` | Done — 321 firms, 61,857 obs |
| 4. RPO panel | `04_build_rpo_quarterly.py` | `data/processed/rpo_quarterly.csv` | Done — 279/291 attempted firms |
| 5. Billings panel | `05_build_billings_panel.py` | `data/processed/billings_panel.csv` | Done — 321 firms, 8,207 rows |
| 6. Margin panel | `06_build_margin_panel.py` | `data/processed/margin_panel.csv` | Done — 321 firms, 41,960 rows |
| 7. AI mention panel | `07_build_ai_mention_panel.py` | `data/processed/ai_mention_panel.csv` | Done — 321 firms, 8,437 filings |
| Phase 3: prompt engineering | `prompts/*.txt` | — | Done — Apr 2026 |
| 8. Supply scoring (ρ_i) | `08_score_supply_rho.py` | `data/processed/lit_scores.csv` | Pending — Phase 4 |
| 9. Demand scoring (δ_i) | `09_score_demand_delta.py` | `data/processed/demand_friction.csv` | Pending — after step 8 |
| 10. DiD regressions | `analysis/did_v3.R` | — | Pending — after steps 8–9 |

---

## Repository Structure

```
scripts/
  01_build_firm_universe.py     # SEC EDGAR firm universe — SIC 7370-7379, knowledge, placebo tiers
  02_collect_10k_text.py        # 10-K Item 1 extraction — pre-shock filings only (< 2022-11-01)
  03_build_financial_panel.py   # Quarterly financials from EDGAR XBRL (10 metrics, 2019Q1–2025Q4)
  04_build_rpo_quarterly.py     # Quarterly RPO panel from EDGAR XBRL (balance-sheet stock)
  05_build_billings_panel.py    # Billings = revenue + gap-aware rpo_delta
  06_build_margin_panel.py      # 5 margin/intensity ratios from financial_panel, P1/P99 winsorized
  07_build_ai_mention_panel.py  # AI mention counts from 10-K/10-Q full filing text
  08_score_supply_rho.py        # Supply-side LLM replicability scoring via Claude API (Phase 4)
  09_score_demand_delta.py      # Demand-side friction scoring via Claude API (Phase 4)

  utils/
    edgar.py                    # EDGAR client: submissions, companyfacts, filing text
    xbrl.py                     # XBRL extraction, Q4 formula, RPO fallback chain
    text_sections.py            # 10-K section parser (Item 1 extraction + iXBRL)
    llm_client.py               # LLM client with prompt caching (Phase 4 first deliverable)
    logging_setup.py            # Structured logging

analysis/
  did_v3.R                      # Two-way FE DiD + WCB inference (fixest + fwildclusterboot)

prompts/
  supply_rho_system.txt         # Supply-side scoring prompt (E0/E1/E2 framework, pure Eloundou β)
  demand_delta_system.txt       # Demand-side scoring prompt (δ_switch/δ_error/δ_data, 6-level rubric)

docs/
  phase6_notes.md               # Thesis writing phase notes — robustness + methodology defense

config/
  ai_mention_lexicon.yaml       # 22-pattern post-ChatGPT AI mention lexicon
  anchor_firms.yaml             # Calibration anchors (structure under review for Phase 4)
  shock_dates.yaml              # Shock date configuration (2022-10-01)
  universe_filters.yaml         # SIC codes, exchange filters, coverage thresholds
  universe_tickers.yaml         # Explicit ticker lists per tier

data/
  raw/
    firm_universe.csv           # 321 firms: ticker, CIK, SIC, exchange, tier, sector_code
  processed/
    financial_panel.csv         # 321 firms, 61,857 obs (10 metrics, 2019Q1–2025Q4)
    rpo_quarterly.csv           # 279 firms, 5,293 rows (quarterly RPO snapshots)
    billings_panel.csv          # 321 firms, 8,207 rows (revenue + gap-aware rpo_delta)
    margin_panel.csv            # 321 firms, 41,960 rows (5 ratios, P1/P99 winsorized)
    ai_mention_panel.csv        # 321 firms, 8,437 filings (AI mentions by category)
    lit_scores.csv              # Supply scores (ρ_i) — produced by 08_score_supply_rho.py in Phase 4
    demand_friction.csv         # Demand scores (δ_i) — produced by 09_score_demand_delta.py in Phase 4

notebooks/
  thesis_notebook.ipynb         # All figures (the only place for figure code)

text_data/10k_extracts/         # Pre-shock 10-K Item 1 text (gitignored) — 321 files
text_data/10k10q_extracts/      # Full filing text cache for AI mention panel (gitignored) — 8,437 files
logs/                           # Pipeline logs (gitignored)
```

---

## Data Sources

All data sourced exclusively from public SEC EDGAR APIs — no external data providers, no web scraping, no Wayback Machine.

- **Firm universe:** SEC EDGAR submissions API — SIC 7370–7379 firms on NYSE/Nasdaq with ≥6 pre-shock XBRL quarters, plus curated knowledge and placebo tier additions
- **Financial data:** SEC EDGAR companyfacts XBRL API — quarterly revenue, COGS, R&D, SG&A, SBC, and related metrics
- **RPO data:** SEC EDGAR companyfacts XBRL API — balance-sheet stock of remaining performance obligations
- **10-K text:** SEC EDGAR full-text API — Item 1 Business Description, pre-shock filings only (filing date < 2022-11-01)

---

## Treatment Variable Construction

### Supply Side: ρ_i — LLM Replicability Score ∈ [1, 100]

Each firm's pre-shock 10-K Item 1 Business Description is scored using an adaptation of the Eloundou et al. (2024) "GPTs are GPTs" task-exposure framework, applied one level up the value chain at the product level rather than the worker level.

The firm's product is decomposed into 6–10 customer-facing tasks, each classified as:
- **E1** — direct LLM exposure: tasks whose output is primarily text, documents, communication, or text-based matching/ranking
- **E2** — LLM + standard tools: tasks requiring database or API access before generating language output
- **E0** — no meaningful LLM exposure: real-time data streams, physical hardware, sub-second latency SLA, or deep proprietary system integration as the core moat

The final score is the Eloundou β aggregation applied at the product level:

```
raw_exposure     = (E1_count + 0.5 × E2_count) / n_tasks
normalized_score = round(raw_exposure × 99 + 1, 1)   ∈ [1, 100]
```

Scores emerge deterministically from task classification. The scoring prompt includes a panel of 15 firms covering software, knowledge-intensive services, and placebo sectors as **task decomposition examples** — these teach classification logic (which tasks belong to E0, E1, or E2 for a given product type) but are not numeric calibration targets. The prompt explicitly forbids the model from targeting any expected score.

**Scoring prompt:** `prompts/supply_rho_system.txt`
**Scoring script:** `scripts/08_score_supply_rho.py`

---

### Demand Side: δ_i — Customer Friction Score ∈ [0, 1]

Constructed by scoring each firm's pre-shock 10-K text across three theoretically grounded dimensions, each on a 6-level rubric (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):

**δ_switch — Switching Cost** (Farrell & Klemperer 2007): How difficult and costly is it for an existing customer to replace this product? Encompasses contractual lock-in, technical integration depth, organizational change costs, and data portability barriers.

**δ_error — Error Cost** (Agrawal, Gans & Goldfarb 2018): How costly is it when the product — or an AI replacement — makes a mistake? High when errors carry regulatory penalties, legal liability, patient safety risk, or irreversible financial consequences.

**δ_data — Data/Network Moat** (Katz & Shapiro 1985; Rochet & Tirole 2003): Does this product have a proprietary data asset or network effect that a new AI-native competitor cannot replicate from public sources?

```
δ_composite = (δ_switch + δ_error + δ_data) / 3   ∈ [0, 1]
```

Each rubric level's descriptor includes 4–6 categorical cross-sector examples (e.g., "payroll/HCM platforms with tax jurisdiction configurations", "clinical data platforms embedded in FDA-regulated trial workflows", "actuarial databases with insurer pooling") — sector-level guidance rather than firm-specific anchor targets. A separate methodological note in the prompt addresses knowledge-intensive services, where switching cost manifests through regulatory or institutional channels rather than technical integration. Consumer-facing education platforms represent the opposite profile (near-zero friction).

**Scoring prompt:** `prompts/demand_delta_system.txt`
**Scoring script:** `scripts/09_score_demand_delta.py`

---

## Financial Panel Construction

The financial panel is built from SEC EDGAR XBRL companyfacts, covering ten metrics per firm-quarter: revenue, COGS, R&D, SG&A, SBC, capex, operating income, net income, deferred revenue (current), and deferred revenue (noncurrent). The panel spans 2019Q1–2025Q4, yielding 61,857 observations across 321 firms.

**Q4 revenue** is computed as `Annual_FY − (Q1 + Q2 + Q3)` from XBRL-reported annual and quarterly figures. Negative implied Q4 values are discarded as tag-mismatch or amended-filing artifacts. When multiple annual filings exist for the same fiscal year, the higher value is retained (original filing preferred over post-restatement amendments).

**Sample construction:** The primary software tier covers SIC 7370–7379 firms on NYSE/Nasdaq with at least six pre-shock XBRL revenue quarters. IIIV is excluded due to post-privatization accounting restatements (FY2022: −40%, FY2023: −49%) that make historical revenue data unreliable. Knowledge and placebo tier firms were added from curated ticker lists.

---

## RPO Panel Construction

Remaining Performance Obligations (RPO) represent contracted but not-yet-recognized revenue — a forward-looking demand signal for SaaS firms. RPO is a balance-sheet stock (point-in-time snapshot at each period end) drawn from EDGAR XBRL companyfacts.

**Four-tier fallback hierarchy:**
1. `RevenueRemainingPerformanceObligation` (primary ASC 606 tag)
2. `ContractWithCustomerLiability` (alternative ASC 606 tag)
3. `ContractWithCustomerLiabilityCurrent` + `ContractWithCustomerLiabilityNoncurrent` (summed)
4. `DeferredRevenue` + `DeferredRevenueNoncurrent` (pre-ASC 606 proxy)

**Coverage:** 279 of 291 attempted firms (30 placebo firms are skipped by design; 12 firms have no reportable RPO across any tag — typically point-in-time revenue recognizers or pre-ASC 606 reporters).

---

## Billings Panel Construction

Billings captures total invoiced amounts including newly-contracted future revenue, making it a leading indicator ahead of recognized revenue for SaaS firms:

```
billings_t = revenue_t + (rpo_t − rpo_{t-1})
```

**Gap-aware rpo_delta:** The sequential RPO difference is set to NaN when consecutive observations are more than 120 days apart, preventing multi-quarter RPO swings from being attributed to a single quarter. This affects annual-only RPO reporters (10-K but not 10-Q disclosers) and firms with irregular reporting gaps. Billings is NaN wherever rpo_delta is NaN — this is correct, not a coverage failure.

**Coverage:** 229 of 321 firms have ≥6 computable billings quarters; 21 have sparse coverage (1–5 quarters); 71 have no computable billings (no RPO data or all gaps exceed 120 days). Coverage thresholds for analysis are applied at regression time, not at panel build time.

---

## Margin Panel Construction

Five margin and intensity ratios are computed from the financial panel and stored in long format alongside P1/P99 winsorized values:

| Metric | Formula |
|--------|---------|
| `gross_margin` | (revenue − COGS) / revenue |
| `rd_intensity` | R&D / revenue |
| `sga_intensity` | SG&A / revenue |
| `sbc_intensity` | SBC / revenue |
| `opex_ratio` | (COGS + R&D + SG&A) / revenue |

Zero-revenue quarters are treated as missing (not divided) to avoid infinite values. NaN propagates naturally when any required component is absent. Winsorization is applied per metric across the full non-NaN distribution (P1/P99 clipping), with raw values retained alongside winsorized values.

---

## AI Mention Panel Construction

Each 10-K and 10-Q filing for all 321 firms over 2019Q1–2025Q4 is searched for post-ChatGPT AI terminology using a 22-pattern lexicon defined in `config/ai_mention_lexicon.yaml`. The lexicon is deliberately strict — generic ML terms predating ChatGPT (machine learning, deep learning, neural network, artificial intelligence) are excluded to avoid pre-period contamination that would bias DiD estimates toward zero.

**Three lexicon categories:**
- **Core** (8 patterns): generative AI, LLM, large language model, foundation model, and variants
- **Named products** (8 patterns): ChatGPT, GPT, Claude, Gemini, Copilot, Llama, Mistral
- **Technical** (6 patterns): prompt engineering, retrieval augmented, RAG, AI assistant, AI copilot, AI agent

Overlapping matches are deduplicated by position (longest match wins at each character offset). The panel records total mention counts by category plus filing word count for density normalization.

**Result:** 8,437 filings across 321 firms. Pre-shock (before 2022Q4): 151 total mentions. Post-shock: 7,472 total mentions. Ratio: **49.5×** — a sharp structural break consistent with the ChatGPT shock.

---

## Running the Pipeline

```bash
# Steps 1–7: Already complete. Do not re-run (SEC rate limits).
# Phase 3 (prompt engineering): Already complete.

# Phase 4, Step 1: write scripts/utils/llm_client.py
# (not yet created — first Phase 4 deliverable)

# Phase 4, Step 2: supply-side scoring (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python3 scripts/08_score_supply_rho.py --test           # validate 15 anchor decompositions (~$0.50)
python3 scripts/08_score_supply_rho.py --skip-existing  # full run (~$5)

# Phase 4, Step 3: demand-side scoring
python3 scripts/09_score_demand_delta.py --test         # validate 13 anchor decompositions
python3 scripts/09_score_demand_delta.py --skip-existing  # full run (~$5)

# Phase 5: regressions
Rscript analysis/did_v3.R
```

Scoring uses `claude-haiku-4-5-20251001` with prompt caching enabled. Total scoring budget is ~$10 across 321 firms × 2 sides.

---

## Key Design Choices

- **Pre-shock scoring only:** 10-K filings dated before 2022-11-01 ensure ρ_i and δ_i are not contaminated by post-shock firm adaptation to AI
- **Three-tier design:** Software firms (treatment) + knowledge firms (spillover control) + placebo firms (falsification) allows testing whether any post-2022 revenue effect is specific to LLM replicability or reflects broader knowledge-economy trends
- **Deterministic rubric-based scoring:** Both supply and demand scores emerge from explicit rubric matching formulas applied to classifications made by the model on pre-shock text. No researcher-calibrated adjustment factors. Anchor firm examples in prompts teach classification logic, not numeric targets.
- **IIIV excluded:** Post-privatization accounting restatements make its historical revenue unreliable
- **Q4 revenue via formula:** XBRL quarterly tags are often absent for Q4; computing Q4 as `Annual − (Q1+Q2+Q3)` is standard and avoids a systematic gap in the panel
- **Gap-aware billings:** RPO differences across annual gaps would attribute multi-quarter changes to a single quarter; the 120-day threshold prevents this contamination
- **Strict AI mention lexicon:** Excluding generic pre-ChatGPT ML terms keeps the pre-shock baseline clean and avoids attenuation bias in the AI adoption proxy

---

## Theoretical Context

**Second-order displacement:** This paper studies AI displacing the software products that had themselves displaced workers from cognitive tasks in the 1990s–2000s. The Acemoglu–Restrepo (2022) task-substitution framework applies one level up the value chain: B2B software firms are *task externalizers* who face product obsolescence when LLMs can perform the same tasks at near-zero marginal cost.

**Supply side (ρ_i):** Eloundou et al. (2024) "GPTs are GPTs" task exposure framework, applied to product-level tasks rather than worker tasks. A product whose core task bundle is E1/E2-exposed faces the same substitution pressure as a worker in an E1/E2 occupation. The β aggregation used (`E1 + 0.5·E2 / n_tasks`) is the middle of Eloundou's three aggregations; α (E1 only) and γ (E1 + E2) are reported as robustness.

**Demand side (δ_i):** Even when a product is highly replicable, customers may not switch due to: (1) switching costs (Farrell & Klemperer 2007) — contractual, technical, organizational, and data portability barriers; (2) error costs (Agrawal, Gans & Goldfarb 2018) — regulatory, legal, and irreversibility consequences when AI makes mistakes; (3) data/network moats (Katz & Shapiro 1985; Rochet & Tirole 2003) — proprietary data accumulation and network effects that alternatives cannot replicate.

**Interaction hypothesis:** `β3 > 0` in the triple-interaction DiD specification would confirm that high demand friction buffers high-replicability firms from revenue loss. Equivalently, the effective exposure channel is capability × (absence of friction), not capability alone.

**Methodology notes:** The robustness battery, external validator plan (Eloundou `occ_level.csv`, Anthropic Economic Index, Microsoft Copilot applicability, Felten AIOE), and methodology defense coverage (AAHR exclusion problem, variance compression within SIC 7370-7379, three-tier identification) are documented in `docs/phase6_notes.md`.

---

## Notes

- `thesis.tex` is gitignored — manuscript lives on Overleaf only
- `text_data/`, `logs/`, `data/raw/` are gitignored — pipeline outputs are not version-controlled
- All scoring uses pre-shock 10-K Item 1 text — scores must not be recomputed from later filings
- All figures are generated in `notebooks/thesis_notebook.ipynb` — no standalone figure scripts
- Scoring model is hard-pinned to `claude-haiku-4-5-20251001`; larger models would exceed the project budget
