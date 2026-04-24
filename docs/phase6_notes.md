# Phase 6 Robustness & Econometric Specification Notes

Persistent notes for thesis writing phase. Captured during Phase 3 prompt engineering (Apr 2026) while methodology design is fresh.

## Main DiD Specification Alternatives

**Option 1 — Classic triple interaction (recommended main spec):**

    Y_it = α_i + γ_t + β1·(ρ_i × Post_t) + β2·(δ_i × Post_t)
           + β3·(ρ_i × δ_i × Post_t) + ε_it

Interpretation:
- β1: supply-side marginal effect (LLM replicability)
- β2: demand-side marginal effect (customer friction)
- β3: interaction — does friction moderate exposure effect?
- H1: β1 < 0 (negative revenue impact from exposure)
- H2: β3 > 0 (friction dampens negative impact)

**Option 2 — Effective exposure specification (robustness):**

    Y_it = α_i + γ_t + β1·(ρ_i × (1-δ_i) × Post_t) + ε_it

Where ρ_i × (1-δ_i) = "effective exposure" = capability × (absence of friction). Cleaner single-coefficient test but loses ability to decompose supply vs demand channels.

Report both specifications. Option 1 is main; Option 2 complements with cleaner interpretation.

## Core Identification Insight (for introduction)

Framing paragraph:

"This framework resolves a tension in AI exposure literature: occupation-level exposure scores (Eloundou et al. 2024, Felten-Raj-Seamans 2023) uniformly predict high impact for knowledge workers, yet observed firm-level revenue responses are highly heterogeneous. Our decomposition explains this heterogeneity by separating what LLMs CAN do (supply ρ) from what customers WILL accept (demand δ). High capability does not imply high impact — impact emerges when replicability coincides with low friction."

## Demand Composite Robustness

Main spec: δ = (δ_switch + δ_error + δ_data) / 3

Equal weighting defense (use verbatim):
"In absence of a priori theoretical justification for differential weights, we adopt equal weighting as a maximum entropy prior and test alternative weighting schemes in robustness."

Robustness battery:
1. Weighted alternatives:
   - (0.4, 0.4, 0.2) — emphasize switch+error
   - (0.5, 0.3, 0.2) — emphasize switch
2. Single sub-score DiD: run with each δ_switch, δ_error, δ_data alone — identifies which channel dominates
3. PCA first principal component as alternative composite (tests sub-score correlation concern)
4. Multicollinearity diagnostics (VIF scores, sub-score correlation matrix)

## Supply Robustness

Main: β = (E1 + 0.5·E2) / n

Aggregation alternatives (Eloundou's three variants):
- α = E1 / n (conservative — direct LLM only)
- β = (E1 + 0.5·E2) / n (middle — main)
- γ = (E1 + E2) / n (expansive — tools count fully)

External validators (research report Part 2, Part 4):
- Eloundou occ_level.csv firm-level aggregate via SIC → SOC-6 → NAICS via OEWS employment weights
  Source: github.com/openai/GPTs-are-GPTs
- Anthropic Economic Index task usage data (release_2026_01_15)
  Source: huggingface.co/datasets/Anthropic/EconomicIndex
- Microsoft Copilot applicability scores (Tomlinson et al. 2025)
  Source: github.com/microsoft/working-with-ai
- Felten LM-AIOE and genAI-AIOE (2023 updates)

Expected correlation with ρ_i: 0.5-0.85 on β aggregation.

Variance compression caveat: within-SIC-7370-7379 correlation will be lower; report within-sector residualized correlations.

Cross-LLM reliability: Claude Haiku 4.5 vs Gemini 2.5 Flash — 30-firm sample × 3 iterations, target ICC ≥ 0.90.

## Methodology Defense Section — Required Coverage

### AAHR Exclusion Problem (CRITICAL)

Acemoglu-Autor-Hazell-Restrepo (2022) exclude NAICS 51 (Information) and 54 (Professional/Scientific/Technical) from their AI exposure analysis because those sectors are AI producers rather than users. All 321 firms in the panel live in these excluded sectors.

Required framing:

"We reframe the exposure logic for the producer sector: ρ_i measures replacement of the firm's PRODUCT by LLMs substituting for B2B workflows, not displacement of workers inside the firm. This is a legitimate and complementary angle — while AAHR measures 'AI substitutes for humans in exposed tasks,' we measure 'AI substitutes for the product that performs those tasks for customers.'"

### Variance Compression within SIC 7370-7379

Eloundou β uniformly high for computing occupations. Cross-firm variance within software sector is compressed relative to cross-sector variance.

Required reporting:
- Correlation with external validators on raw scale
- Correlation on within-sector demeaned scale
- Discussion of whether variance compression is a bug or a feature for identification. Our three-tier design (software + knowledge + placebo) addresses this explicitly — the identifying variation is ACROSS tiers, not purely within software.

### Three-Tier Identification

Panel structure — 256 software + 35 knowledge + 30 placebo — provides ex-ante heterogeneity that addresses variance compression. Main identification comes from:
(a) within software: high-ρ firms vs low-ρ firms differential
(b) across tiers: software vs knowledge vs placebo differential
(c) DiD shock interacts with tier-specific ρ distribution

## Current Position (April 2026)

System design completed:
- Supply measurement (ρ): deterministic, literature-grounded, unbiased
- Demand measurement (δ): structured rubric, theory-backed, moderate subjectivity controlled by 10-K grounding
- Combined identification: Effect = f(ρ × (1-δ)) provides clean decomposition

Next stage: Phase 4 (API client + anchor test + full scoring) → Phase 5 (external validation) → Phase 6 (thesis writing).

Scope: Strong master thesis, publishable in mid-tier finance or econometrics journal. PhD extension would require outcome panel expansion (BuiltWith, G2, SimilarWeb, Wayback Machine product/customer dynamics — see research report Part 7).
