"""
Step 5b: Collect Item 1 (Business Description) + Item 1A (Risk Factors)
from SEC EDGAR 10-K filings.

For each firm, finds the most recent 10-K/20-F/40-F filed before Nov 2022,
fetches the primary document, extracts Item 1 text (or Item 4 for 20-F)
and Item 1A text.

Usage:
    python3 scripts/collect_10k_text.py           # all firms
    python3 scripts/collect_10k_text.py --test     # 5 test firms only
"""

import re
import os
import sys
import time
import logging
import argparse
from typing import Optional, Tuple
from html.parser import HTMLParser
import requests
import pandas as pd

INPUT_PATH = "data/raw/firm_universe.csv"
OUTPUT_DIR = "text_data/10k_extracts"
FAILURE_LOG = "logs/10k_v2_failures.log"

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 thesis-research go52geg@tum.de",
    "Accept-Encoding": "gzip, deflate",
}

RATE_LIMIT = 0.2
CUTOFF_DATE = "2022-11-01"

TEST_TICKERS = ["ZIP", "CRWD", "DDOG", "NOW", "ASAN"]

# Priority order for annual report form types
ANNUAL_TYPES = ["10-K", "10-K/A", "10-K405", "20-F", "20-F/A", "40-F", "40-F/A"]

logging.basicConfig(
    filename=FAILURE_LOG,
    filemode="w",
    level=logging.WARNING,
    format="%(asctime)s %(message)s",
)
fail_logger = logging.getLogger("10k_failures")


class HTMLTextExtractor(HTMLParser):
    """Strip HTML tags, return plain text."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False
        if tag in ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def get_text(self):
        return "".join(self.parts)


def strip_html(html: str) -> str:
    """Convert HTML to plain text."""
    parser = HTMLTextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.get_text()


def fetch_with_retry(url: str, timeout: int = 30) -> Optional[requests.Response]:
    """Fetch URL with one retry on 503 errors."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    if resp.status_code == 503:
        time.sleep(10)
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
    return resp


def _search_filings_block(block: dict) -> Optional[Tuple[str, str, str, str]]:
    """Search a filings block for annual report before cutoff.

    Filings are in reverse chronological order, so the first match is the
    most recent.  Prefers non-amended forms (10-K, 20-F, 40-F) over amended
    variants (10-K/A, 20-F/A, 40-F/A) because amended filings often have
    non-standard HTML that breaks Item 1 extraction.

    Returns (accession, filing_date, primary_document, form_type) or None.
    """
    forms = block.get("form", [])
    dates = block.get("filingDate", [])
    accessions = block.get("accessionNumber", [])
    primary_docs = block.get("primaryDocument", [])

    annual_set = set(ANNUAL_TYPES)
    non_amended = {"10-K", "10-K405", "20-F", "40-F"}

    best = None
    for i, form in enumerate(forms):
        form = form.strip()
        if form not in annual_set or dates[i] >= CUTOFF_DATE:
            continue
        pdoc = primary_docs[i] if i < len(primary_docs) else ""
        candidate = (accessions[i], dates[i], pdoc, form)
        if best is None:
            best = candidate
            if form in non_amended:
                break  # non-amended and most recent — ideal
        elif best[3] not in non_amended and form in non_amended:
            best = candidate
            break  # upgraded from amended to non-amended

    return best


def find_10k(cik: int) -> Optional[Tuple[str, str, str, str]]:
    """Find most recent annual report before cutoff.

    Returns (accession, filing_date, primary_document, form_type) or None.
    """
    url = SUBMISSIONS_URL.format(cik="%010d" % cik)
    try:
        resp = fetch_with_retry(url)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    filings = data.get("filings", {})
    recent = filings.get("recent", {})

    # First check filings.recent
    result = _search_filings_block(recent)
    if result is not None:
        return result

    # If not found, check older filing history pages in filings.files
    files = filings.get("files", [])
    for file_entry in files:
        filename = file_entry.get("name", "")
        if not filename:
            continue
        older_url = "https://data.sec.gov/submissions/%s" % filename
        time.sleep(RATE_LIMIT)
        try:
            older_resp = fetch_with_retry(older_url)
            if older_resp.status_code != 200:
                continue
            older_data = older_resp.json()
        except Exception:
            continue

        result = _search_filings_block(older_data)
        if result is not None:
            return result

    return None


def find_primary_doc(cik: int, accession: str, primary_doc_hint: str = "") -> Optional[str]:
    """Find the primary document URL for a filing.

    If primary_doc_hint is provided (from submissions JSON), use it directly.
    Otherwise fetch index.json and find the primary .htm document.
    """
    acc_no_dashes = accession.replace("-", "")
    base_url = ARCHIVES_URL.format(cik=cik, accession=acc_no_dashes)

    # If we have a primary document name from the submissions JSON, use it
    if primary_doc_hint:
        return base_url + primary_doc_hint

    # Fallback: fetch index.json
    index_json_url = base_url + "index.json"
    try:
        resp = fetch_with_retry(index_json_url)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    items = data.get("directory", {}).get("item", [])

    docs = []
    for item in items:
        name = item.get("name", "")
        doc_type = item.get("type", "")
        description = item.get("description", "")
        size = item.get("size", "0")
        try:
            size_int = int(str(size).replace(",", ""))
        except ValueError:
            size_int = 0
        docs.append({
            "name": name,
            "type": doc_type,
            "description": description,
            "size": size_int,
        })

    def _is_exhibit(doc):
        """Check if a document is an exhibit by description or filename."""
        desc = doc["description"].upper()
        name = doc["name"].lower()
        if "EX-" in desc or "EXHIBIT" in desc:
            return True
        if re.search(r"-ex\d", name):
            return True
        if name.startswith("exhibit"):
            return True
        return False

    def _is_viewer_page(doc):
        """Check if file is an EDGAR R*.htm viewer page."""
        return bool(re.match(r"^R\d+\.htm$", doc["name"], re.IGNORECASE))

    # Priority 1: entry where type matches an annual report form
    for target_type in ANNUAL_TYPES:
        for doc in docs:
            if doc["type"].strip() == target_type and doc["name"].lower().endswith((".htm", ".html")):
                return base_url + doc["name"]

    # Priority 2: largest .htm file that is not an exhibit or viewer page
    htm_docs = [
        d for d in docs
        if d["name"].lower().endswith((".htm", ".html"))
        and not _is_viewer_page(d)
    ]
    if htm_docs:
        non_exhibit = [d for d in htm_docs if not _is_exhibit(d)]
        if non_exhibit:
            non_exhibit.sort(key=lambda d: d["size"], reverse=True)
            return base_url + non_exhibit[0]["name"]

        # Fallback: largest .htm overall (excluding viewer pages)
        htm_docs.sort(key=lambda d: d["size"], reverse=True)
        return base_url + htm_docs[0]["name"]

    return None


def find_40f_exhibit(cik: int, accession: str) -> Optional[str]:
    """For 40-F filings, find the AIF exhibit URL from the filing index.json."""
    acc_no_dashes = accession.replace("-", "")
    base_url = ARCHIVES_URL.format(cik=cik, accession=acc_no_dashes)
    index_json_url = base_url + "index.json"

    try:
        resp = fetch_with_retry(index_json_url)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    items = data.get("directory", {}).get("item", [])

    # Look for AIF exhibit by type, description, or filename
    for item in items:
        name = item.get("name", "")
        doc_type = item.get("type", "").upper()
        description = item.get("description", "").upper()

        if not name.lower().endswith((".htm", ".html")):
            continue

        # Match by type field
        is_exhibit = "EX-99" in doc_type or "EX-13" in doc_type
        # Match by description
        is_aif = ("ANNUAL INFORMATION" in description
                  or "AIF" in description
                  or "ANNUAL REPORT" in description)

        if is_exhibit or is_aif:
            return base_url + name

    # Fallback: find ex99 exhibits by filename pattern (handles type="text.gif")
    # AIF is typically exhibit 99.1 (the first and often largest ex99 file)
    ex99_docs = []
    for item in items:
        name = item.get("name", "")
        name_lower = name.lower()
        if not name_lower.endswith((".htm", ".html")):
            continue
        # Match filenames like "xex99d1", "-ex99-1", "ex991", etc.
        if re.search(r"ex99", name_lower):
            size = item.get("size", "0")
            try:
                size_int = int(str(size).replace(",", ""))
            except ValueError:
                size_int = 0
            ex99_docs.append((size_int, name))

    if ex99_docs:
        # Pick the first ex99 .htm by name (AIF is typically exhibit 99.1)
        ex99_docs.sort(key=lambda x: x[1])
        return base_url + ex99_docs[0][1]

    return None


def _extract_section(text: str, start_pattern, end_pattern) -> Optional[str]:
    """Extract a section from text using start/end regex patterns.

    Picks the match with the MOST text before the next end marker,
    which is the actual section body (not TOC entries or cross-references).
    """
    starts = list(start_pattern.finditer(text))
    if not starts:
        return None

    best_section = None
    best_length = 0

    for start_match in starts:
        start_pos = start_match.end()
        end_match = end_pattern.search(text, start_pos)
        end_pos = end_match.start() if end_match else start_pos + 15000

        section = text[start_pos:end_pos].strip()
        if len(section) > best_length:
            best_length = len(section)
            best_section = section

    if best_section is None or best_length < 100:
        return None

    return best_section


def extract_item1(html: str, form_type: str = "10-K") -> Optional[str]:
    """Extract business description section from annual report HTML.

    For 10-K: Item 1. Business
    For 20-F: Item 4. Information on the Company
    For 40-F: Item 1. Business (or general business description)
    """
    text = strip_html(html)

    # Normalize whitespace
    text = re.sub(r"\xa0", " ", text)
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Try Item 1 pattern first (10-K, 40-F), then Item 4 fallback (20-F, 40-F)
    # 10-K: "Item 1. Business" → "Item 1A" or "Item 2"
    # Matches digit "1" or Roman numeral "I"
    item1_start = re.compile(
        r"item\s*[1I][\s.:\-–—]\s*business",
        re.IGNORECASE
    )
    item1_end = re.compile(
        r"item\s*[1I]\s*a[\s.:\-–—]|item\s*(?:2|II)[\s.:\-–—]",
        re.IGNORECASE
    )

    # 20-F/40-F: "Item 4. Information on the Company" → "Item 4A" or "Item 5"
    item4_start = re.compile(
        r"item\s*4[\s.:\-–—]\s*information\s+on\s+the\s+company",
        re.IGNORECASE
    )
    item4_end = re.compile(
        r"item\s*4\s*a[\s.:\-–—]|item\s*5[\s.:\-–—]",
        re.IGNORECASE
    )

    if form_type in ("20-F", "20-F/A"):
        # Try Item 4 first for 20-F, fall back to Item 1
        section = _extract_section(text, item4_start, item4_end)
        if section is None:
            section = _extract_section(text, item1_start, item1_end)
    elif form_type in ("40-F", "40-F/A"):
        # Canadian AIF structure: try multiple patterns
        # AIF Item 4: "Narrative Description of the Business" → "Item 5" (Risk Factors)
        # Separator between "4" and "NARRATIVE" is optional (body may omit it)
        aif_start = re.compile(
            r"item\s*4[\s.:\-–—\n]?\s*narrative\s+description\s+of\s+the\s+business",
            re.IGNORECASE
        )
        aif_end = re.compile(
            r"item\s*5[\s.:\-–—\n]?\s*risk\s+factors",
            re.IGNORECASE
        )
        section = _extract_section(text, aif_start, aif_end)
        if section is None:
            # Try Item 1 (some 40-F filings use 10-K structure)
            section = _extract_section(text, item1_start, item1_end)
        if section is None:
            # Try 20-F Item 4 pattern
            section = _extract_section(text, item4_start, item4_end)
    else:
        # 10-K: Item 1 only
        section = _extract_section(text, item1_start, item1_end)

    if section is None:
        return None

    result = _clean_section(section)

    if len(result.split()) < 20:
        return None

    return result


def _clean_section(section: str) -> str:
    """Clean extracted section text, preserving section headers."""
    lines = section.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        # Keep short lines if they look like headers (title case or caps)
        if len(stripped) < 60:
            is_header = (
                stripped.istitle() or
                stripped.isupper() or
                re.match(r'^[A-Z][A-Za-z\s&/,\-]{3,55}$', stripped)
            )
            if is_header:
                cleaned.append(stripped)
                continue
        # Remove very short non-header lines
        if len(stripped) < 10:
            continue
        # Remove pure number lines
        if re.match(r"^[\d\s.]+$", stripped):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def extract_item1a(html: str, form_type: str = "10-K") -> Optional[str]:
    """Extract Item 1A (Risk Factors) from annual report HTML.

    For 10-K: Item 1A. Risk Factors → Item 1B or Item 2
    For 20-F/40-F: Item 3D. Risk Factors → Item 4 or similar
    """
    text = strip_html(html)

    # Normalize whitespace
    text = re.sub(r"\xa0", " ", text)
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if form_type in ("20-F", "20-F/A"):
        # 20-F: Item 3D. Risk Factors → Item 4
        start = re.compile(
            r"item\s*3\s*d[\s.:\-–—]\s*risk\s*factors",
            re.IGNORECASE
        )
        end = re.compile(
            r"item\s*4[\s.:\-–—]",
            re.IGNORECASE
        )
    elif form_type in ("40-F", "40-F/A"):
        # 40-F AIF: Item 5. Risk Factors → Item 6
        start = re.compile(
            r"item\s*5[\s.:\-–—\n]?\s*risk\s+factors",
            re.IGNORECASE
        )
        end = re.compile(
            r"item\s*6[\s.:\-–—]",
            re.IGNORECASE
        )
    else:
        # 10-K: Item 1A. Risk Factors → Item 1B or Item 2
        start = re.compile(
            r"item\s*[1I]\s*a[\s.:\-–—]\s*risk\s*factors",
            re.IGNORECASE
        )
        end = re.compile(
            r"item\s*(?:[1I]\s*b|2|II)[\s.:\-–—]",
            re.IGNORECASE
        )

    section = _extract_section(text, start, end)
    if section is None:
        return None

    result = _clean_section(section)

    if len(result.split()) < 20:
        return None

    return result


def process_firm(cik: int, ticker: str, company_name: str) -> Optional[dict]:
    """Process one firm: find annual report, fetch doc, extract business section."""

    # Step 1: Find annual report filing
    filing = find_10k(cik)
    if filing is None:
        fail_logger.warning("%s (%s): no annual report found before %s" % (ticker, company_name, CUTOFF_DATE))
        return None

    accession, filing_date, primary_doc, form_type = filing
    time.sleep(RATE_LIMIT)

    # Step 2: Find primary document
    if form_type in ("40-F", "40-F/A"):
        # 40-F: the main doc is just a wrapper; fetch AIF exhibit instead
        doc_url = find_40f_exhibit(cik, accession)
        if doc_url is None:
            fail_logger.warning("%s (%s): no AIF exhibit found in 40-F %s" % (ticker, company_name, accession))
            return None
    else:
        doc_url = find_primary_doc(cik, accession, primary_doc)
        if doc_url is None:
            fail_logger.warning("%s (%s): could not find primary .htm in %s" % (ticker, company_name, accession))
            return None

    time.sleep(RATE_LIMIT)

    # Step 3: Fetch document (with 503 retry)
    try:
        resp = fetch_with_retry(doc_url, timeout=120)
        if resp.status_code != 200:
            fail_logger.warning("%s (%s): HTTP %d fetching %s" % (ticker, company_name, resp.status_code, doc_url))
            return None
    except Exception as e:
        fail_logger.warning("%s (%s): fetch error: %s" % (ticker, company_name, str(e)))
        return None

    time.sleep(RATE_LIMIT)

    # Step 4: Extract business description (Item 1)
    item1_text = extract_item1(resp.text, form_type)
    if item1_text is None:
        fail_logger.warning("%s (%s): could not extract business section from %s (form=%s)" % (
            ticker, company_name, doc_url, form_type))
        return None

    # Step 5: Extract risk factors (Item 1A) — optional
    item1a_text = extract_item1a(resp.text, form_type)
    if item1a_text is None:
        fail_logger.warning("%s (%s): Item 1A not found (Item 1 OK)" % (ticker, company_name))

    # Combine sections
    if item1a_text:
        combined = item1_text + "\n\n### ITEM_1A_START ###\n\n" + item1a_text
        sections = "Item1+Item1A"
    else:
        combined = item1_text
        sections = "Item1"

    item1_wc = len(item1_text.split())
    item1a_wc = len(item1a_text.split()) if item1a_text else 0
    total_wc = len(combined.split())

    return {
        "ticker": ticker,
        "company_name": company_name,
        "filing_date": filing_date,
        "accession": accession,
        "doc_url": doc_url,
        "form_type": form_type,
        "sections": sections,
        "text": combined,
        "item1_words": item1_wc,
        "item1a_words": item1a_wc,
        "word_count": total_wc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run on 5 test firms only")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Specific tickers to extract (e.g. --tickers MSFT CRM)")
    args = parser.parse_args()

    t0 = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df = df[df["meets_filters"] == True].reset_index(drop=True)

    if args.tickers:
        requested = [t.upper() for t in args.tickers]
        df = df[df["ticker"].isin(requested)].reset_index(drop=True)
        missing = set(requested) - set(df["ticker"])
        if missing:
            print("WARNING: not in firm_universe.csv: %s" % ", ".join(sorted(missing)), flush=True)
        print("TICKERS MODE: %d firms\n" % len(df), flush=True)
    elif args.test:
        df = df[df["ticker"].isin(TEST_TICKERS)].reset_index(drop=True)
        print("TEST MODE: %d firms\n" % len(df), flush=True)

    total = len(df)
    print("Loaded %d firms from %s" % (total, INPUT_PATH), flush=True)
    print("Extracting Item 1 + Item 1A from most recent pre-shock annual report\n", flush=True)

    successes = 0
    failures = 0
    failed_tickers = []
    word_counts = []
    item1a_count = 0

    for i, (_, row) in enumerate(df.iterrows()):
        cik = int(row["cik"])
        ticker = row["ticker"]
        company_name = row["company_name"]

        result = process_firm(cik, ticker, company_name)

        if result is None:
            failures += 1
            failed_tickers.append(ticker)
            print("  %s  FAILED" % ticker, flush=True)
        else:
            # Save text file
            output_path = "%s/%s.txt" % (OUTPUT_DIR, ticker)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("---\n")
                f.write("TICKER: %s\n" % result["ticker"])
                f.write("COMPANY: %s\n" % result["company_name"])
                f.write("FILING_DATE: %s\n" % result["filing_date"])
                f.write("ACCESSION: %s\n" % result["accession"])
                f.write("FORM_TYPE: %s\n" % result["form_type"])
                f.write("SECTIONS: %s\n" % result["sections"])
                f.write("---\n")
                f.write(result["text"])

            successes += 1
            word_counts.append(result["word_count"])
            if result["item1a_words"] > 0:
                item1a_count += 1

            print("  %s  OK: Item1=%d  Item1A=%d  Total=%d  (%s, %s)" % (
                ticker, result["item1_words"], result["item1a_words"],
                result["word_count"], result["filing_date"],
                result["sections"]), flush=True)

        count = i + 1
        if count % 10 == 0 or count == total:
            elapsed = time.time() - t0
            rate = elapsed / count
            remaining = rate * (total - count)
            print("  [%d/%d] ok=%d fail=%d  elapsed: %.0fs  remaining: ~%.0fs" % (
                count, total, successes, failures, elapsed, remaining), flush=True)

    # --- Summary ---
    elapsed = time.time() - t0
    mean_wc = sum(word_counts) / len(word_counts) if word_counts else 0

    print("\n" + "=" * 50, flush=True)
    print("Firms with text extracted:     %d" % successes, flush=True)
    print("  with Item 1A:                %d" % item1a_count, flush=True)
    print("  Item 1 only:                 %d" % (successes - item1a_count), flush=True)
    print("Firms failed:                  %d" % failures, flush=True)
    print("Mean word count:               %.0f" % mean_wc, flush=True)
    if word_counts:
        print("Min word count:                %d" % min(word_counts), flush=True)
        print("Max word count:                %d" % max(word_counts), flush=True)
    print("Time elapsed: %.0fs (%.1f min)" % (elapsed, elapsed / 60), flush=True)

    if failed_tickers:
        print("\nFailed firms (%d):" % len(failed_tickers), flush=True)
        for t in sorted(failed_tickers):
            print("  %s" % t, flush=True)

    print("\nSaved to %s/" % OUTPUT_DIR, flush=True)
    if failures > 0:
        print("Failures logged to %s" % FAILURE_LOG, flush=True)


if __name__ == "__main__":
    main()
