"""
Unified SEC EDGAR client with disk caching and rate-limited, retried requests.

Caches to data/raw/edgar_cache/:
  submissions/{cik}.json
  companyfacts/{cik}.json
  filings/{cik}/{accession_nodash}.html
  company_tickers.json

Rate limit: 0.15 s between calls (max ~6.6/sec, safely below EDGAR's 10/sec cap).
Retries on 429 / 5xx: 3 attempts with exponential back-off (2 s, 8 s, 32 s).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_HEADERS = {
    "User-Agent": "thesis-research hakanzekigulmez@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

_RATE_LIMIT_S = 0.15          # minimum seconds between any two EDGAR requests
_RETRY_DELAYS = (2, 8, 32)    # seconds before 1st, 2nd, 3rd retry

_SUBMISSIONS_URL    = "https://data.sec.gov/submissions/CIK{cik}.json"
_COMPANYFACTS_URL   = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_ARCHIVES_BASE_URL  = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

_CACHE_ROOT = Path("data/raw/edgar_cache")
_CACHE_SUBMISSIONS  = _CACHE_ROOT / "submissions"
_CACHE_COMPANYFACTS = _CACHE_ROOT / "companyfacts"
_CACHE_FILINGS      = _CACHE_ROOT / "filings"
_CACHE_META         = _CACHE_ROOT

_last_request_time: float = 0.0
# NOTE: _last_request_time is module-level state. This is safe for single-process
# sequential use only. Under multiprocessing, each worker gets its own copy of this
# variable (not shared), so N workers could collectively exceed EDGAR's 10 req/sec
# cap. Parallel EDGAR fetches would require multiprocessing.Lock + multiprocessing.Value
# to coordinate _last_request_time across processes.
_session = requests.Session()


# ---------------------------------------------------------------------------
# Low-level HTTP
# ---------------------------------------------------------------------------
def _rate_sleep() -> None:
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    wait = _RATE_LIMIT_S - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


def _get(url: str, *, binary: bool = False, timeout: int = 30) -> bytes | str | None:
    """
    Perform a GET with rate limiting and exponential back-off retries.

    Returns bytes if binary=True, decoded str otherwise, None on persistent failure.
    """
    _rate_sleep()
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            resp = _session.get(url, headers=_HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            if delay is None:
                return None
            time.sleep(delay)
            continue

        if resp.status_code == 200:
            return resp.content if binary else resp.text

        # Retryable errors
        if resp.status_code in (429, 500, 502, 503, 504):
            if delay is None:
                return None
            time.sleep(delay)
            _rate_sleep()
            continue

        # Non-retryable (404, 403, etc.)
        return None

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pad_cik(cik: int | str) -> str:
    return str(int(cik)).zfill(10)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_company_tickers() -> dict:
    """
    Download and cache SEC master company_tickers_exchange.json.

    Returns the raw parsed JSON dict with keys 'fields' and 'data'.
    Cache path: data/raw/edgar_cache/company_tickers.json
    """
    cache_path = _CACHE_META / "company_tickers.json"
    _CACHE_META.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    raw = _get(_COMPANY_TICKERS_URL, timeout=60)
    if raw is None:
        raise RuntimeError("Failed to download company_tickers_exchange.json from SEC EDGAR")

    data = json.loads(raw)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def ticker_to_cik(ticker: str) -> Optional[int]:
    """
    Look up the CIK for a given ticker symbol.

    Uses the cached company_tickers_exchange.json; returns None if not found.
    """
    data = get_company_tickers()
    fields = data["fields"]
    ticker_idx = fields.index("ticker")
    cik_idx    = fields.index("cik")

    ticker_upper = ticker.upper()
    for row in data["data"]:
        if str(row[ticker_idx]).upper() == ticker_upper:
            return int(row[cik_idx])
    return None


def get_submissions(cik: int | str) -> dict:
    """
    Fetch and cache the submissions JSON for a given CIK.

    Cache path: data/raw/edgar_cache/submissions/{padded_cik}.json

    Automatically fetches older filing history pages listed in
    filings.files and merges them into the recent filings block so
    callers always see a complete filing list.
    """
    padded = pad_cik(cik)
    cache_path = _CACHE_SUBMISSIONS / f"{padded}.json"
    _CACHE_SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = _SUBMISSIONS_URL.format(cik=padded)
    raw = _get(url)
    if raw is None:
        raise RuntimeError(f"Failed to fetch submissions for CIK {padded}")

    data = json.loads(raw)

    # Merge older filing pages into data['filings']['recent']
    older_pages = data.get("filings", {}).get("files", [])
    for page in older_pages:
        filename = page.get("name", "")
        if not filename:
            continue
        page_url = f"https://data.sec.gov/submissions/{filename}"
        page_raw = _get(page_url)
        if page_raw is None:
            continue
        page_data = json.loads(page_raw)
        recent = page_data.get("recent", {})
        for key, vals in recent.items():
            existing = data["filings"]["recent"].get(key, [])
            data["filings"]["recent"][key] = existing + vals

    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def get_companyfacts(cik: int | str) -> dict:
    """
    Fetch and cache the companyfacts JSON for a given CIK.

    Cache path: data/raw/edgar_cache/companyfacts/{padded_cik}.json
    """
    padded = pad_cik(cik)
    cache_path = _CACHE_COMPANYFACTS / f"{padded}.json"
    _CACHE_COMPANYFACTS.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = _COMPANYFACTS_URL.format(cik=padded)
    raw = _get(url)
    if raw is None:
        raise RuntimeError(f"Failed to fetch companyfacts for CIK {padded}")

    data = json.loads(raw)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def get_filing_index(cik: int | str, accession_no: str) -> Optional[dict]:
    """
    Fetch the index.json for a specific filing.

    accession_no may be hyphenated ("0001234567-22-000001") or flat.
    Returns parsed JSON dict, or None on failure.
    Not cached independently — callers cache the content they care about.
    """
    padded = pad_cik(cik)
    accession_flat = accession_no.replace("-", "")
    index_url = _ARCHIVES_BASE_URL.format(cik=int(cik), accession=accession_flat) + "index.json"
    raw = _get(index_url)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def get_filing_text(accession_no: str, cik: int | str, doc_name: str) -> Optional[str]:
    """
    Fetch and cache the HTML of a specific document within a filing.

    accession_no: may be hyphenated or flat.
    doc_name:     primary document filename (e.g. "form10k.htm").
    Cache path:   data/raw/edgar_cache/filings/{padded_cik}/{accession_flat}/{doc_name}

    Returns raw HTML string, or None on failure.
    """
    padded = pad_cik(cik)
    accession_flat = accession_no.replace("-", "")
    cache_dir = _CACHE_FILINGS / padded / accession_flat
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use a safe filename in case doc_name contains path separators
    safe_name = Path(doc_name).name
    cache_path = cache_dir / safe_name

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    url = _ARCHIVES_BASE_URL.format(cik=int(cik), accession=accession_flat) + doc_name
    raw = _get(url, timeout=120)
    if raw is None:
        return None

    cache_path.write_text(raw, encoding="utf-8", errors="replace")
    return raw
