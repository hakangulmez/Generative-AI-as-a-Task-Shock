# _archive — Deprecated Scripts

These scripts pre-date the numbered pipeline (01–09) introduced during the
three-tier universe expansion (April 2026). They are preserved here for
reference and are NOT part of the active pipeline.

| Archived file | Replaced by | Reason |
|---|---|---|
| `build_firm_universe.py` | `01_build_firm_universe.py` | Numbered pipeline; three-tier universe (primary_software + primary_knowledge + placebo) |
| `collect_10k_text.py` | `02_collect_10k_text.py` | Numbered pipeline; pre-shock cutoff logic unchanged |
| `build_financial_panel.py` | `03_build_financial_panel.py` | Numbered pipeline; IIIV exclusion hardcoded; three-tier firm list |
| `build_rpo_quarterly.py` | `04_build_rpo_quarterly.py` | Numbered pipeline; 4-tier fallback tag hierarchy; gap-aware diff |
| `score_literature_rubric.py` | `08_score_supply_rho.py` (Phase 4) | Old supply scoring script: wrong model (Sonnet vs Haiku), SIC-7370-7379 hardcoded, customer_type B2C null-escape, references stale prompt path `prompts/literature_rubric_system.txt` |
| `score_demand_friction.py` | `09_score_demand_delta.py` (Phase 4) | Old demand scoring script: wrong model (Sonnet vs Haiku), references stale prompt path `prompts/demand_friction_system.txt` |

## Do not run these scripts

The archived scoring scripts (`score_literature_rubric.py`,
`score_demand_friction.py`) reference prompt paths that no longer exist
and use `claude-sonnet-4-6` which would exceed the scoring budget (~$30–100
vs ~$10 for Haiku 4.5).

The archived pipeline scripts (`build_*.py`, `collect_10k_text.py`) hit
SEC EDGAR and would overwrite production panel files if run without `--tickers`.
