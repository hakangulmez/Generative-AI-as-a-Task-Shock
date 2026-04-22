"""
XBRL fact extraction utilities for SEC EDGAR companyfacts JSON.

All logic reproduced bit-for-bit from legacy scripts:
  build_financial_panel.py  — revenue tags, Q4 formula, metric tags, COGS fallback
  build_rpo_quarterly.py    — RPO tag hierarchy, balance-sheet stock semantics

Numerical-reproducibility constraints (DO NOT CHANGE WITHOUT REVIEW):
  - REVENUE_TAGS order determines which revenue series wins for each firm
  - Q4 formula: Annual_FY − (Q1+Q2+Q3); discard if result ≤ 0
  - Multiple annual filings for same FY: keep HIGHER value (not latest filed)
  - RPO fallback: try each tier; only advance if current tier yields zero rows
  - RPO sum-tiers: require BOTH components at the exact same period_end date
  - Period filters: quarterly ≤ 100 days, annual 340–400 days
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Panel date bounds (must match build_financial_panel.py exactly)
# ---------------------------------------------------------------------------
PANEL_START = "2019-01-01"
PANEL_END   = "2025-12-31"

# ---------------------------------------------------------------------------
# Tag lists (LEGACY ORDER — do not reorder; first match wins)
# ---------------------------------------------------------------------------

# NOTE: legacy order is RevenueFromContract... first, then Revenues.
# The master prompt lists them in a different order; the code below
# matches the ACTUAL legacy build_financial_panel.py exactly.
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomer",
]

METRIC_TAGS: dict[str, list[str]] = {
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "rd_expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcess",
        "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
    ],
    "sga_expense": [
        "SellingGeneralAndAdministrativeExpense",
        # fallback: SellingAndMarketingExpense + GeneralAndAdministrativeExpense
        # handled inside extract_metric_series()
    ],
}

# SGA component fallback tags (summed when combined tag absent)
_SGA_COMPONENTS = (
    "SellingAndMarketingExpense",
    "GeneralAndAdministrativeExpense",
)

# COGS tags for gross_profit fallback: GrossProfit = Revenue − COGS
COGS_TAGS = [
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
]

# New metrics not in legacy (for margin/billings panel extensions)
EXTENDED_METRIC_TAGS: dict[str, list[str]] = {
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ],
    "stock_comp": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
    ],
    "net_income": [
        "NetIncomeLoss",
    ],
}

# RPO fallback chain — tier order is CRITICAL (legacy order, do not change)
# Each entry: (tier_name, [tag1, tag2, ...])
# Single-tag tiers: value taken directly.
# Two-tag tiers: BOTH tags must be present at the same period_end; values summed.
RPO_TIERS = [
    ("rpo",      ["RevenueRemainingPerformanceObligation"]),
    ("cwcl",     ["ContractWithCustomerLiability"]),
    ("cwcl_sum", ["ContractWithCustomerLiabilityCurrent",
                  "ContractWithCustomerLiabilityNoncurrent"]),
    ("deferred", ["DeferredRevenue", "DeferredRevenueNoncurrent"]),
]

_ANNUAL_FORMS    = frozenset({"10-K", "10-K/A", "10-K405"})
_QUARTERLY_FORMS = frozenset({"10-Q", "10-Q/A"})
_VALID_FORMS     = _ANNUAL_FORMS | _QUARTERLY_FORMS


# ---------------------------------------------------------------------------
# Date / quarter utilities
# ---------------------------------------------------------------------------

def quarter_from_date(date_str: str) -> Optional[tuple[int, int]]:
    """
    Parse 'YYYY-MM-DD' → (fiscal_year, fiscal_quarter).

    Returns None on parse error.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        q = (dt.month - 1) // 3 + 1
        return dt.year, q
    except (ValueError, TypeError):
        return None


def period_to_quarter_label(yr: int, q: int) -> str:
    """(2022, 4) → '2022Q4'"""
    return f"{yr}Q{q}"


# ---------------------------------------------------------------------------
# Internal low-level extractors (match legacy function signatures)
# ---------------------------------------------------------------------------

def _extract_quarterly_raw(us_gaap: dict, tag: str) -> dict[str, dict]:
    """
    Extract true quarterly (non-cumulative) entries for a single XBRL tag.

    Returns {quarter_label: {"period_end", "fiscal_year", "fiscal_quarter",
                              "val", "filed"}}

    Filters:
      - fp != "FY"
      - period_end within PANEL_START..PANEL_END
      - period length ≤ 100 days (excludes cumulative YTD entries)
      - USD units only
    For duplicate quarter labels: keep latest filed.
    """
    if tag not in us_gaap:
        return {}

    entries = us_gaap[tag].get("units", {}).get("USD", [])
    result: dict[str, dict] = {}

    for e in entries:
        if e.get("fp", "") == "FY":
            continue
        end = e.get("end", "")
        if not end or not (PANEL_START <= end <= PANEL_END):
            continue

        start = e.get("start", "")
        if start and end:
            try:
                days = (datetime.strptime(end, "%Y-%m-%d") -
                        datetime.strptime(start, "%Y-%m-%d")).days
                if days > 100:
                    continue  # cumulative YTD
            except ValueError:
                pass

        parsed = quarter_from_date(end)
        if parsed is None:
            continue
        yr, q = parsed
        label = period_to_quarter_label(yr, q)
        filed = e.get("filed", "")
        val   = e.get("val")

        if label not in result or filed > result[label]["filed"]:
            result[label] = {
                "period_end":     end,
                "fiscal_year":    yr,
                "fiscal_quarter": q,
                "val":            val,
                "filed":          filed,
            }

    return result


def _extract_annual_raw(us_gaap: dict, tag: str) -> dict[tuple, dict]:
    """
    Extract annual (FY, ~365-day) entries for a single XBRL tag.

    Returns {(year, month): {"period_end", "val", "filed"}}
    Keyed by (calendar_year, month) of period end to handle non-calendar FY.

    Filters:
      - fp == "FY"
      - period length 340–400 days
      - period_end within PANEL_START..PANEL_END
      - USD units only
    For multiple filings at same FY-end: keep HIGHER value.
    (Amended 10-K/A can reduce reported revenue; original is correct denominator.)
    """
    if tag not in us_gaap:
        return {}

    entries = us_gaap[tag].get("units", {}).get("USD", [])
    result: dict[tuple, dict] = {}

    for e in entries:
        if e.get("fp", "") != "FY":
            continue
        end   = e.get("end", "")
        start = e.get("start", "")
        if not end or not start:
            continue
        if not (PANEL_START <= end <= PANEL_END):
            continue

        try:
            days = (datetime.strptime(end, "%Y-%m-%d") -
                    datetime.strptime(start, "%Y-%m-%d")).days
        except ValueError:
            continue

        if days < 340 or days > 400:
            continue

        dt_end = datetime.strptime(end, "%Y-%m-%d")
        filed  = e.get("filed", "")
        val    = e.get("val")

        key = (dt_end.year, dt_end.month)
        existing_val = result[key]["val"] if key in result else None
        if key not in result or (
            val is not None and existing_val is not None and val > existing_val
        ):
            result[key] = {"period_end": end, "val": val, "filed": filed}

    return result


def _compute_q4(
    quarterly: dict[str, dict],
    annual: dict[tuple, dict],
    all_quarterly_labels: Optional[set[str]] = None,
) -> dict[str, dict]:
    """
    Compute Q4 = Annual_FY − (Q1 + Q2 + Q3) for each fiscal year.

    Rules (legacy-exact):
      - A quarter belongs to the FY if its period_end is within 365 days
        before (and including) the annual period_end.
      - Require exactly 3 such quarters; skip the FY otherwise.
      - Discard if Q4_val ≤ 0 (tag mismatch / restatement artifact).
      - Skip if the Q4 quarter label already exists in `quarterly`
        or in the optional `all_quarterly_labels` set.

    Returns {quarter_label: {...}} for Q4 entries only.
    """
    if not annual:
        return {}

    q4_entries: dict[str, dict] = {}

    for (_fy_year, _fy_month), ann in annual.items():
        ann_end = datetime.strptime(ann["period_end"], "%Y-%m-%d")
        ann_val = ann["val"]
        if ann_val is None:
            continue

        fy_start_approx = ann_end - timedelta(days=365)
        q_sum   = 0
        q_count = 0
        for qlabel, qdata in quarterly.items():
            q_end = datetime.strptime(qdata["period_end"], "%Y-%m-%d")
            if fy_start_approx < q_end <= ann_end:
                if qdata["val"] is not None:
                    q_sum   += qdata["val"]
                    q_count += 1

        if q_count != 3:
            continue

        q4_val = ann_val - q_sum
        if q4_val <= 0:
            continue

        q4_end  = ann["period_end"]
        parsed  = quarter_from_date(q4_end)
        if parsed is None:
            continue
        yr, q   = parsed
        label   = period_to_quarter_label(yr, q)

        if label in quarterly:
            continue
        if all_quarterly_labels and label in all_quarterly_labels:
            continue

        q4_entries[label] = {
            "period_end":     q4_end,
            "fiscal_year":    yr,
            "fiscal_quarter": q,
            "val":            q4_val,
            "filed":          ann["filed"],
        }

    return q4_entries


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------

def extract_quarterly_revenue(
    companyfacts_json: dict,
    *,
    best_coverage: bool = False,
) -> list[dict]:
    """
    Extract quarterly revenue series from companyfacts JSON.

    Parameters
    ----------
    best_coverage : bool, default False
        If False (legacy): first tag in REVENUE_TAGS with any data wins.
        If True: all tags tried; tag with most in-panel quarterly entries
        (after Q4 computation) wins. Use True for the three-tier universe
        panel to handle firms that switched XBRL tags mid-panel (ASC 606
        transition pattern). Keep default False for legacy 248-firm
        reproduction.

    Returns list of dicts sorted by period_end:
      {period_end, fiscal_year, fiscal_quarter, revenue, tag_used}
    Returns [] if no revenue data found.
    """
    us_gaap = companyfacts_json.get("facts", {}).get("us-gaap", {})

    def _to_records(quarters: dict, tag: str) -> list[dict]:
        return [
            {
                "period_end":     v["period_end"],
                "fiscal_year":    v["fiscal_year"],
                "fiscal_quarter": v["fiscal_quarter"],
                "revenue":        v["val"],
                "tag_used":       tag,
            }
            for v in sorted(quarters.values(), key=lambda x: x["period_end"])
        ]

    if not best_coverage:
        # Legacy: first tag with any data wins
        for tag in REVENUE_TAGS:
            quarters = _extract_quarterly_raw(us_gaap, tag)
            if not quarters:
                continue
            annual = _extract_annual_raw(us_gaap, tag)
            if annual:
                quarters.update(_compute_q4(quarters, annual))
            return _to_records(quarters, tag)
        return []

    # Best-coverage: try all tags, pick the one with most observations
    best_q: dict = {}
    best_tag: Optional[str] = None
    for tag in REVENUE_TAGS:
        quarters = _extract_quarterly_raw(us_gaap, tag)
        if not quarters:
            continue
        annual = _extract_annual_raw(us_gaap, tag)
        if annual:
            quarters.update(_compute_q4(quarters, annual))
        if len(quarters) > len(best_q):
            best_q = quarters
            best_tag = tag
    if best_tag is None:
        return []
    return _to_records(best_q, best_tag)


def extract_quarterly_series(
    companyfacts_json: dict,
    tag_candidates: list[str],
    *,
    accumulate: bool = False,
) -> dict[str, dict]:
    """
    Extract a quarterly series using a priority-ordered list of candidate tags.

    Parameters
    ----------
    tag_candidates : list of XBRL tag names tried in order; first with data wins.
    accumulate     : if True, SUM values across all tag_candidates at each quarter
                     (used for component fallback like SGA = S&M + G&A).
                     If False, first matching tag wins (standard behaviour).

    Returns {quarter_label: {"period_end", "fiscal_year", "fiscal_quarter", "val"}}
    Empty dict if no data found.
    """
    us_gaap = companyfacts_json.get("facts", {}).get("us-gaap", {})

    if not accumulate:
        for tag in tag_candidates:
            q = _extract_quarterly_raw(us_gaap, tag)
            if not q:
                continue
            ann = _extract_annual_raw(us_gaap, tag)
            if ann:
                q.update(_compute_q4(q, ann))
            return q
        return {}

    # Accumulate: sum across all tags per quarter
    combined: dict[str, dict] = {}
    for tag in tag_candidates:
        q = _extract_quarterly_raw(us_gaap, tag)
        ann = _extract_annual_raw(us_gaap, tag)
        if ann:
            q.update(_compute_q4(q, ann))
        for label, data in q.items():
            if data["val"] is None:
                continue
            if label not in combined:
                combined[label] = {**data, "val": 0}
            combined[label]["val"] += data["val"]
    return combined


def extract_metric_series(
    companyfacts_json: dict,
    col_name: str,
) -> dict[str, dict]:
    """
    Extract a named metric series using the METRIC_TAGS priority list.

    Handles the sga_expense component fallback (S&M + G&A) when the combined
    tag is absent — reproduces legacy extract_metric() logic exactly.

    Returns {quarter_label: {"period_end", "fiscal_year", "fiscal_quarter", "val"}}
    """
    tags = METRIC_TAGS.get(col_name, [])
    result = extract_quarterly_series(companyfacts_json, tags)
    if result:
        return result

    # sga_expense fallback: sum SellingAndMarketing + GeneralAndAdministrative
    if col_name == "sga_expense":
        result = extract_quarterly_series(
            companyfacts_json, list(_SGA_COMPONENTS), accumulate=True
        )

    return result


def extract_cogs_series(companyfacts_json: dict) -> dict[str, dict]:
    """
    Extract COGS series using COGS_TAGS priority list.
    Used for the gross_profit = revenue − COGS fallback.
    """
    return extract_quarterly_series(companyfacts_json, COGS_TAGS)


def extract_rpo_series(companyfacts_json: dict) -> list[dict]:
    """
    Extract RPO (Revenue Remaining Performance Obligation) series.

    RPO is a balance-sheet STOCK (point-in-time snapshot), not a flow.
    One value per period_end date; no cumulative filtering needed.

    Fallback chain (legacy RPO_TIERS order — do not change):
      Tier 1: RevenueRemainingPerformanceObligation  (single tag)
      Tier 2: ContractWithCustomerLiability           (single tag)
      Tier 3: ContractWithCustomerLiabilityCurrent
              + ContractWithCustomerLiabilityNoncurrent  (BOTH required)
      Tier 4: DeferredRevenue
              + DeferredRevenueNoncurrent                (BOTH required)

    For sum-tiers: BOTH component tags must be present at the same period_end;
    entries where only one component is found are discarded.

    For duplicate period_end dates (amended filings): keep latest filed.
    Form filter: 10-K, 10-Q, 10-K/A, 10-Q/A only.

    Returns list of dicts sorted by period_end:
      {period_end, fiscal_year, fiscal_quarter, rpo, rpo_tag}
    Returns [] if no RPO data found at any tier.
    """
    us_gaap = companyfacts_json.get("facts", {}).get("us-gaap", {})

    for tier_name, tag_keys in RPO_TIERS:
        # Build {period_end: {val, filed, count}} across all tags in this tier
        by_end: dict[str, dict] = {}

        for tk in tag_keys:
            entries = us_gaap.get(tk, {}).get("units", {}).get("USD", [])
            for e in entries:
                end = e.get("end", "")
                if not end or not (PANEL_START <= end <= PANEL_END):
                    continue
                if e.get("form", "") not in _VALID_FORMS:
                    continue
                filed = e.get("filed", "")
                val   = e.get("val")
                if val is None:
                    continue

                if end not in by_end:
                    by_end[end] = {"val": 0, "filed": filed, "count": 0}

                # For duplicate period_end within same tier: keep latest filed
                if filed >= by_end[end]["filed"]:
                    by_end[end]["val"]   += val
                    by_end[end]["filed"]  = filed
                    by_end[end]["count"] += 1

        if not by_end:
            continue

        # Sum-tiers: require BOTH components at each period_end
        if len(tag_keys) > 1:
            by_end = {
                d: v for d, v in by_end.items()
                if v["count"] == len(tag_keys)
            }

        if not by_end:
            continue

        # Build quarter-keyed output
        result: dict[str, dict] = {}
        for end_date, info in sorted(by_end.items()):
            parsed = quarter_from_date(end_date)
            if parsed is None:
                continue
            yr, q  = parsed
            label  = period_to_quarter_label(yr, q)
            # For duplicate quarter labels: keep latest period_end within quarter
            if label not in result or end_date > result[label]["period_end"]:
                result[label] = {
                    "period_end":     end_date,
                    "fiscal_year":    yr,
                    "fiscal_quarter": q,
                    "rpo":            info["val"],
                    "rpo_tag":        tier_name,
                }

        if result:
            return sorted(result.values(), key=lambda x: x["period_end"])

    return []


def _extract_instant_raw(us_gaap: dict, tag: str) -> dict[str, dict]:
    """
    Extract point-in-time (balance-sheet) entries for a single XBRL tag.

    Unlike flow metrics there is no start date — these are snapshot values.
    For duplicate quarter labels (amended filings): keep latest filed.
    Form filter: 10-K, 10-Q, 10-K/A, 10-Q/A only.

    Returns {quarter_label: {"period_end", "fiscal_year", "fiscal_quarter",
                              "val", "filed"}}
    """
    if tag not in us_gaap:
        return {}

    entries = us_gaap[tag].get("units", {}).get("USD", [])
    result: dict[str, dict] = {}

    for e in entries:
        end = e.get("end", "")
        if not end or not (PANEL_START <= end <= PANEL_END):
            continue
        if e.get("form", "") not in _VALID_FORMS:
            continue
        filed = e.get("filed", "")
        val   = e.get("val")
        if val is None:
            continue

        parsed = quarter_from_date(end)
        if parsed is None:
            continue
        yr, q = parsed
        label = period_to_quarter_label(yr, q)

        if label not in result or filed > result[label]["filed"]:
            result[label] = {
                "period_end":     end,
                "fiscal_year":    yr,
                "fiscal_quarter": q,
                "val":            val,
                "filed":          filed,
            }

    return result


def extract_quarterly_metric(
    companyfacts_json: dict,
    tag_list: list[str],
    *,
    is_instant: bool = False,
    fallback_sum_tags: Optional[list[str]] = None,
) -> tuple[dict[str, dict], Optional[str]]:
    """
    Generic extractor for a single financial metric.

    Parameters
    ----------
    tag_list         : XBRL tags tried in priority order; first with data wins.
    is_instant       : True for balance-sheet stocks (point-in-time, no Q4 formula).
                       False for income statement / cash flow flows.
    fallback_sum_tags: If provided and primary tags yield no data, sum these
                       component tags (e.g. SGA = S&M + G&A).

    Returns
    -------
    (data, tag_used)
    data     : {quarter_label: {"period_end", "fiscal_year", "fiscal_quarter", "val"}}
    tag_used : winning tag name, "fallback_sum" if component fallback fired, or None.
    """
    us_gaap = companyfacts_json.get("facts", {}).get("us-gaap", {})

    if is_instant:
        best_result: dict = {}
        best_tag: Optional[str] = None
        for tag in tag_list:
            result = _extract_instant_raw(us_gaap, tag)
            if len(result) > len(best_result):
                best_result = result
                best_tag = tag
        if best_tag is not None:
            return best_result, best_tag
        return {}, None

    # Duration / flow metrics: best-coverage wins (most in-panel observations)
    best_q: dict = {}
    best_tag_flow: Optional[str] = None
    for tag in tag_list:
        q = _extract_quarterly_raw(us_gaap, tag)
        if not q:
            continue
        ann = _extract_annual_raw(us_gaap, tag)
        if ann:
            q.update(_compute_q4(q, ann))
        if len(q) > len(best_q):
            best_q = q
            best_tag_flow = tag
    if best_tag_flow is not None:
        return best_q, best_tag_flow

    # Component fallback (SGA = SellingAndMarketing + GeneralAndAdmin)
    if fallback_sum_tags:
        combined = extract_quarterly_series(
            companyfacts_json, fallback_sum_tags, accumulate=True
        )
        if combined:
            return combined, "fallback_sum"

    return {}, None


def extract_deferred_revenue_series(companyfacts_json: dict) -> list[dict]:
    """
    Extract combined deferred revenue (current + noncurrent) at each period_end.

    Separate from the RPO fallback chain — used directly in the financial panel
    as a standalone balance-sheet metric, not as an RPO proxy.

    Both components must be present at the same period_end to produce a value.
    Returns list of dicts sorted by period_end:
      {period_end, fiscal_year, fiscal_quarter, deferred_revenue}
    """
    tags = ["DeferredRevenue", "DeferredRevenueNoncurrent"]
    us_gaap = companyfacts_json.get("facts", {}).get("us-gaap", {})
    by_end: dict[str, dict] = {}

    for tk in tags:
        entries = us_gaap.get(tk, {}).get("units", {}).get("USD", [])
        for e in entries:
            end = e.get("end", "")
            if not end or not (PANEL_START <= end <= PANEL_END):
                continue
            if e.get("form", "") not in _VALID_FORMS:
                continue
            filed = e.get("filed", "")
            val   = e.get("val")
            if val is None:
                continue
            if end not in by_end:
                by_end[end] = {"val": 0, "filed": filed, "count": 0}
            if filed >= by_end[end]["filed"]:
                by_end[end]["val"]   += val
                by_end[end]["filed"]  = filed
                by_end[end]["count"] += 1

    # Both components required
    by_end = {d: v for d, v in by_end.items() if v["count"] == len(tags)}

    result = []
    for end_date, info in sorted(by_end.items()):
        parsed = quarter_from_date(end_date)
        if parsed is None:
            continue
        yr, q = parsed
        result.append({
            "period_end":       end_date,
            "fiscal_year":      yr,
            "fiscal_quarter":   q,
            "deferred_revenue": info["val"],
        })
    return result
