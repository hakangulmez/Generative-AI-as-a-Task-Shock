"""
Extract Item 1 (Business), Item 1A (Risk Factors), and Item 7 (MD&A)
from SEC 10-K annual report HTML.

Handles:
  - Plain HTML 10-Ks
  - Inline XBRL (iXBRL) with ix: namespace tags
  - Mixed HTML/iXBRL
  - 20-F (Item 4 / Item 3D) and 40-F (AIF Item 4 / Item 5) variants

Returns dict:
  {
    "item_1":  str | None,
    "item_1a": str | None,
    "item_7":  str | None,
    "length_words": {"item_1": int, "item_1a": int, "item_7": int},
  }

Item boundaries (10-K):
  Item 1  ends at "Item 1A." or "Item 2."
  Item 1A ends at "Item 1B." or "Item 2."
  Item 7  ends at "Item 7A." or "Item 8."
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional


# ---------------------------------------------------------------------------
# HTML → plain text
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Strip HTML/iXBRL tags; insert newlines at block boundaries."""

    # iXBRL namespace prefixes to treat as transparent wrappers
    _IXBRL_PREFIXES = ("ix:", "xbrli:", "xbrl:", "link:", "label:")

    # Block-level tags that should produce a newline
    _BLOCK_TAGS = frozenset({"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"})

    # Tags whose content to suppress entirely
    _SKIP_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag_lower = tag.lower()
        # Strip iXBRL namespace prefixes for classification
        bare = tag_lower.split(":")[-1] if ":" in tag_lower else tag_lower

        if bare in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if bare in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        bare = tag_lower.split(":")[-1] if ":" in tag_lower else tag_lower

        if bare in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if bare in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _html_to_text(html: str) -> str:
    """Convert raw HTML (including iXBRL) to plain text."""
    # Remove iXBRL hidden elements (continuations, footnotes)
    html = re.sub(r'<ix:hidden[^>]*>.*?</ix:hidden>', ' ', html,
                  flags=re.IGNORECASE | re.DOTALL)
    # Remove XML/XBRL header declarations that precede the HTML body
    html = re.sub(r'<\?xml[^?]*\?>', '', html)

    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass

    text = parser.get_text()
    # Normalize non-breaking spaces and zero-width chars
    text = text.replace("\xa0", " ").replace("\u200b", "")
    # Collapse horizontal whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

_NOISE_LINES: frozenset[str] = frozenset({
    "table of contents",
    "index to financial statements",
    "index",
    "continued",
    "(continued)",
    "(in thousands)",
    "(in millions)",
    "(dollars in thousands)",
    "(dollars in millions)",
})


def _clean(section: str) -> str:
    """Remove boilerplate lines and page numbers; preserve section headers."""
    lines = section.split("\n")
    cleaned: list[str] = []
    prev_blank = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
            continue
        prev_blank = False

        if stripped.lower() in _NOISE_LINES:
            continue
        if re.match(r"^[\d\s.]+$", stripped):  # page numbers
            continue
        if len(stripped) < 60:
            is_header = (
                stripped.istitle() or
                stripped.isupper() or
                re.match(r'^[A-Z][A-Za-z\s&/,\-]{3,55}$', stripped)
            )
            if is_header:
                cleaned.append(stripped)
            continue

        cleaned.append(stripped)

    return "\n".join(cleaned).strip()


def _extract_section(text: str,
                     start_re: re.Pattern,
                     end_re: re.Pattern,
                     min_words: int = 50) -> Optional[str]:
    """
    Find the longest match of start_re → end_re in text.

    Multiple start matches may occur (TOC entry + actual section body).
    We pick the one with the most content before the next end marker,
    which is the actual section body rather than a TOC cross-reference.
    """
    starts = list(start_re.finditer(text))
    if not starts:
        return None

    best: Optional[str] = None
    best_len = 0

    for m in starts:
        start_pos = m.end()
        end_m = end_re.search(text, start_pos)
        end_pos = end_m.start() if end_m else min(start_pos + 30_000, len(text))

        section = text[start_pos:end_pos].strip()
        if len(section) > best_len:
            best_len = len(section)
            best = section

    if best is None or len(best.split()) < min_words:
        return None

    return _clean(best)


# ---------------------------------------------------------------------------
# iXBRL fallback helpers (for large-cap filings with custom Item 1 titles)
# ---------------------------------------------------------------------------

# Many modern 10-K filings use a "FORM 10-K CROSS-REFERENCE INDEX" at the
# end of the document to map form item numbers to custom section titles
# (e.g. "ITEM 1 About Honeywell" instead of "ITEM 1. Business").  The primary
# extractor sees no "Item 1 ... Business" heading in the body and returns None.
# These fallbacks find the custom title from the cross-reference index and then
# locate the body section by searching for that title as a standalone heading.

_XREF_INDEX_RE = re.compile(r'CROSS.REFERENCE\s+INDEX', re.IGNORECASE)

# Captures custom Item 1 title from the cross-reference index.
# Handles:
#   "ITEM 1About Honeywell\n\n49\n"  (HON — no separator, newline before page)
#   "Item 1.Business4-6, ..."         (GE  — dot separator, digit immediately after title)
_FALLBACK_ITEM1_TITLE_RE = re.compile(
    r'ITEM\s*1[\s.:\-–—]*\s*([A-Za-z][^\n\d]{3,60}?)\s*(?:\n|\d)',
    re.IGNORECASE,
)

# Strategy 2: extended separator  (INTU/EMR: "ITEM 1 - BUSINESS")
# Primary pattern uses single [\s.:\-–—]; this uses + so " - " is matched in one step.
_FALLBACK_ITEM1_EXT_START = re.compile(
    r"item\s*[1I][\s.:\-–—]+\s*business",
    re.IGNORECASE,
)

# Strategy 3: combined Items 1 and 2  (PSX/OXY/COP: "Items 1 and 2. Business and Properties")
# Anchored to start-of-line to avoid matching inline cross-references like
# "see Environmental Management within Item 1 and 2—Business and Properties".
_FALLBACK_ITEM12_START = re.compile(
    r"(?:^|\n)[ \t]*items?\s*1\s*(?:and\s*2)?[\s.:\-–—]+\s*business",
    re.IGNORECASE,
)
_FALLBACK_ITEM12_END = re.compile(
    r"(?:^|\n)[ \t]*item\s*1\s*[Aa][\s.:\-–—]"
    r"|(?:^|\n)[ \t]*item\s*3[\s.:\-–—]",
    re.IGNORECASE,
)


def _find_xref_pos(text: str) -> int:
    """Return start position of the CROSS-REFERENCE INDEX, or len(text).

    Takes the LAST occurrence — the early mention ("see the Cross-Reference Index
    for a cross-reference to the traditional format") must not shadow the actual
    index block at the end of the document.
    """
    matches = list(_XREF_INDEX_RE.finditer(text))
    return matches[-1].start() if matches else len(text)


def _first_non_toc_occurrence(text: str, title: str, before: int) -> Optional[int]:
    """Return position of first occurrence of title that is NOT a TOC entry.

    A TOC entry is identified by having a digit as the first non-whitespace
    character immediately following the title (i.e. a page number reference).
    XBRL concatenations (e.g. "BusinessMember") are also skipped — detected by
    the title being immediately followed by a word character with no separator.
    """
    pattern = re.compile(re.escape(title), re.IGNORECASE)
    for m in pattern.finditer(text, 0, before):
        # Skip XBRL metadata concatenations: the char right after the match
        # is an alphanumeric (no space), e.g. "BusinessMember"
        raw_next = text[m.end():m.end() + 1]
        if raw_next and raw_next[0].isalpha():
            continue
        after = text[m.end(): m.end() + 30].strip()
        if not re.match(r'^\d', after):
            return m.start()
    return None


def _extract_item1_fallback(text: str) -> Optional[str]:
    """Item 1 fallback for modern iXBRL 10-Ks.  Three strategies tried in order:

    1. Cross-reference index approach (HON/GE): read custom section title from
       the FORM 10-K CROSS-REFERENCE INDEX at end, then locate body section.
    2. Extended separator (INTU/EMR): "ITEM 1 - BUSINESS" with multi-char separator.
    3. Combined Items 1 and 2 (PSX/OXY/COP): "Items 1 and 2. Business and Properties".
    """
    xp = _find_xref_pos(text)

    # --- Strategy 1: cross-reference index title lookup ---
    m = _FALLBACK_ITEM1_TITLE_RE.search(text[xp:])
    if m:
        title = m.group(1).strip()
        # Skip single-word titles (e.g. "Business"): too generic to reliably find
        # as a standalone section heading; mid-sentence occurrences would match first.
        if len(title.split()) >= 2:
            body_start = _first_non_toc_occurrence(text, title, xp)
            if body_start is not None:
                end_re = re.compile(
                    r'\n(?:RISK FACTORS|Risk Factors|Management.{0,5}s?\s+Discussion)',
                    re.IGNORECASE,
                )
                end_m = end_re.search(text, body_start + 500)
                body_end = end_m.start() if (end_m and end_m.start() < xp) else xp
                section = text[body_start:body_end].strip()
                if len(section.split()) >= 50:
                    return _clean(section)

    # --- Strategy 2: extended separator (INTU/EMR: "ITEM 1 - BUSINESS") ---
    result = _extract_section(text, _FALLBACK_ITEM1_EXT_START, _10K_ITEM1_END)
    if result is not None:
        return result

    # --- Strategy 3: combined Items 1 and 2 (PSX/OXY/COP) ---
    # Use a custom loop instead of _extract_section (longest-wins) because
    # COP has an earlier prose reference "Items 1 and 2—Business and Properties,
    # contain forward-looking statements..." (note trailing comma) that would
    # win the longest-content heuristic over the actual section heading.
    for m3 in _FALLBACK_ITEM12_START.finditer(text):
        # Detect prose references: the heading line ends with a comma
        line_end = text.find('\n', m3.end())
        if line_end == -1:
            line_end = len(text)
        rest_of_line = text[m3.end():line_end].strip()
        if rest_of_line.endswith(','):
            continue
        end_m3 = _FALLBACK_ITEM12_END.search(text, m3.end())
        end_pos3 = end_m3.start() if end_m3 else min(m3.end() + 60_000, len(text))
        section = text[m3.end():end_pos3].strip()
        if len(section.split()) >= 50:
            return _clean(section)

    # --- Strategy 5: "Item 1. [WORDS] Business" with newline-tolerant separator ---
    # Handles "ITEM\n1. DESCRIPTION OF BUSINESS" (VBIX) and "Item 1. Our Business" (IDAI).
    result = _extract_section(text, _FALLBACK_ITEM1_CUSTOM_START, _10K_ITEM1_END)
    if result is not None:
        return result

    # --- Strategy 4: branded "ABOUT [COMPANY]." inline heading (e.g. GE) ---
    for m4 in _FALLBACK_ABOUT_START.finditer(text):
        # Skip TOC entries: heading immediately followed by a digit
        after4 = text[m4.end():m4.end() + 10].strip()
        if re.match(r'^\d', after4):
            continue
        end_re4 = re.compile(
            r'RISK\s+FACTORS\.'           # GE-style inline heading (period not newline)
            r'|\nRISK\s+FACTORS\s*\n'     # standard heading (newline-delimited)
            r'|(?:^|\n)[ \t]*item\s*[1I]\s*[Aa][\s.:\-–—]',  # standard Item 1A
            re.IGNORECASE,
        )
        end_m4 = end_re4.search(text, m4.end())
        end_pos4 = end_m4.start() if end_m4 else min(m4.end() + 60_000, len(text))
        section = text[m4.end():end_pos4].strip()
        if len(section.split()) >= 50:
            return _clean(section)

    return None


_FALLBACK_ITEM1A_EXT_START = re.compile(
    r"item\s*[1I]\s*a[\s.:\-–—]+\s*risk\s*factors",
    re.IGNORECASE,
)

# Strategy 4: branded "ABOUT [COMPANY NAME]." inline heading
# (e.g. GE: "ABOUT GENERAL ELECTRIC. General Electric Company is...")
# ALL-CAPS requirement minimises false positives on mid-sentence "about".
_FALLBACK_ABOUT_START = re.compile(
    r"ABOUT\s+[A-Z][A-Z\s&]{5,50}\.",
)

# Strategy 5: "Item 1. [WORDS] Business" — allows newline in separator and
# 0–4 custom words before "business" (VBIX: "DESCRIPTION OF BUSINESS";
# IDAI: "Our Business").
_FALLBACK_ITEM1_CUSTOM_START = re.compile(
    r"item\s*[1I][\s.\n:–—]+(?:\w+\s+){0,4}business",
    re.IGNORECASE,
)

def _extract_item1a_fallback(text: str) -> Optional[str]:
    """Item 1A (Risk Factors) fallback for iXBRL filings with custom headers.

    Three strategies:
    1. Extended separator (INTU/EMR: "ITEM 1A - RISK FACTORS").
    2. Standalone newline-delimited "RISK FACTORS" heading (HON-style).
    3. Inline "RISK FACTORS." heading (GE-style: period not newline).
    """
    # Strategy 1: extended separator
    result = _extract_section(text, _FALLBACK_ITEM1A_EXT_START, _10K_ITEM1A_END)
    if result is not None:
        return result

    # Strategy 2: standalone newline-delimited heading
    xp = _find_xref_pos(text)
    rf_re = re.compile(r'(?:^|\n)(RISK FACTORS)\s*\n', re.IGNORECASE)
    body_start: Optional[int] = None
    for m in rf_re.finditer(text, 0, xp):
        after = text[m.end(): m.end() + 30].strip()
        if not re.match(r'^\d', after):
            body_start = m.start()
            break

    if body_start is not None:
        end_re = re.compile(
            r'\n(?:UNRESOLVED STAFF COMMENTS|PROPERTIES|'
            r'Quantitative\s+and\s+Qualitative)',
            re.IGNORECASE,
        )
        end_m = end_re.search(text, body_start + 500)
        body_end = end_m.start() if (end_m and end_m.start() < xp) else xp
        section = text[body_start:body_end].strip()
        if len(section.split()) >= 50:
            return _clean(section)

    # Strategy 3: inline "RISK FACTORS." heading (GE-style)
    rf_inline_re = re.compile(r'RISK\s+FACTORS\.\s+', re.IGNORECASE)
    for m3 in rf_inline_re.finditer(text, 0, xp):
        # Skip TOC entries (immediately followed by a digit or newline+digit)
        after3 = text[m3.end():m3.end() + 10].strip()
        if re.match(r'^\d', after3):
            continue
        end_re3 = re.compile(
            r'(?:^|\n)[ \t]*PROPERTIES\b'
            r'|(?:^|\n)[ \t]*UNRESOLVED\s+STAFF\s+COMMENTS\b'
            r'|(?:^|\n)[ \t]*item\s*(?:2|II)[\s.:\-–—]',
            re.IGNORECASE,
        )
        end_m3 = end_re3.search(text, m3.end())
        body_end3 = end_m3.start() if end_m3 else min(m3.end() + 60_000, len(text))
        section = text[m3.end():body_end3].strip()
        if len(section.split()) >= 50:
            return _clean(section)

    return None


def _extract_item7_fallback(text: str) -> Optional[str]:
    """Item 7 (MD&A) fallback for iXBRL filings with custom headers."""
    xp = _find_xref_pos(text)
    mda_re = re.compile(
        r'(?:^|\n)Management.{0,5}s?\s+Discussion\s+and\s+Analysis[^\n]*\n',
        re.IGNORECASE,
    )
    body_start: Optional[int] = None
    for m in mda_re.finditer(text, 0, xp):
        after = text[m.end(): m.end() + 30].strip()
        if not re.match(r'^\d', after):
            body_start = m.start()
            break
    if body_start is None:
        return None

    end_re = re.compile(
        r'\n(?:Quantitative\s+and\s+Qualitative|'
        r'FINANCIAL STATEMENTS|Financial Statements)',
        re.IGNORECASE,
    )
    end_m = end_re.search(text, body_start + 500)
    body_end = end_m.start() if (end_m and end_m.start() < xp) else xp

    section = text[body_start:body_end].strip()
    return _clean(section) if len(section.split()) >= 50 else None


# ---------------------------------------------------------------------------
# Form-type aware patterns
# ---------------------------------------------------------------------------

# 10-K Item 1
_10K_ITEM1_START = re.compile(
    r"item\s*[1I][\s.:\-–—]\s*business",
    re.IGNORECASE,
)
_10K_ITEM1_END = re.compile(
    r"(?:^|\n)[ \t]*item\s*[1I]\s*[Aa][\s.:\-–—]\s*risk\s*factors"
    r"|(?:^|\n)[ \t]*item\s*(?:2|II)[\s.:\-–—]",
    re.IGNORECASE,
)

# 10-K Item 1A
_10K_ITEM1A_START = re.compile(
    r"item\s*[1I]\s*a[\s.:\-–—]\s*risk\s*factors",
    re.IGNORECASE,
)
_10K_ITEM1A_END = re.compile(
    r"item\s*(?:[1I]\s*b|2|II)[\s.:\-–—]",
    re.IGNORECASE,
)

# 10-K Item 7 (MD&A)
_10K_ITEM7_START = re.compile(
    r"item\s*7[\s.\-–—:]+\s*management.{0,10}discussion",
    re.IGNORECASE,
)
_10K_ITEM7_END = re.compile(
    r"(?:^|\n)[ \t]*item\s*7\s*[Aa][\s.\-–—:]"
    r"|(?:^|\n)[ \t]*item\s*(?:8|VIII)[\s.\-–—:]",
    re.IGNORECASE,
)

# 20-F equivalents
_20F_ITEM4_START = re.compile(
    r"item\s*4[\s.\-–—:]+\s*information\s+on\s+the\s+company",
    re.IGNORECASE,
)
_20F_ITEM4_END = re.compile(
    r"item\s*4\s*[Aa][\s.\-–—:]|item\s*5[\s.\-–—:]",
    re.IGNORECASE,
)
_20F_ITEM3D_START = re.compile(
    r"item\s*3\s*[Dd][\s.\-–—:]+\s*risk\s*factors",
    re.IGNORECASE,
)
_20F_ITEM3D_END = re.compile(
    r"item\s*4[\s.\-–—:]",
    re.IGNORECASE,
)
_20F_ITEM5_START = re.compile(
    r"item\s*5[\s.\-–—:]+\s*operating\s+and\s+financial\s+review",
    re.IGNORECASE,
)
_20F_ITEM5_END = re.compile(
    r"item\s*6[\s.\-–—:]",
    re.IGNORECASE,
)

# 40-F (Canadian AIF) equivalents
_40F_ITEM4_START = re.compile(
    r"item\s*4[\s.\-–—:\n]*\s*narrative\s+description\s+of\s+the\s+business",
    re.IGNORECASE,
)
_40F_ITEM4_END = re.compile(
    r"item\s*5[\s.\-–—:\n]*\s*risk\s*factors",
    re.IGNORECASE,
)
_40F_ITEM5_START = re.compile(
    r"item\s*5[\s.\-–—:\n]*\s*risk\s*factors",
    re.IGNORECASE,
)
_40F_ITEM5_END = re.compile(
    r"item\s*6[\s.\-–—:]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_sections(html: str, form_type: str = "10-K") -> dict:
    """
    Extract Item 1, Item 1A, and Item 7 from a 10-K filing HTML.

    Parameters
    ----------
    html:      Raw HTML string (may include iXBRL markup).
    form_type: Filing form type string from SEC submissions JSON.
               Supported: "10-K", "10-K/A", "10-K405",
                          "20-F", "20-F/A", "40-F", "40-F/A".

    Returns
    -------
    dict with keys:
        item_1, item_1a, item_7  — extracted text (str) or None
        length_words             — {"item_1": int, "item_1a": int, "item_7": int}
    """
    text = _html_to_text(html)
    ft = form_type.upper().replace("/A", "").strip()

    item_1: Optional[str] = None
    item_1a: Optional[str] = None
    item_7: Optional[str] = None

    if ft in ("20-F",):
        # Business description: Item 4 "Information on the Company"
        item_1 = _extract_section(text, _20F_ITEM4_START, _20F_ITEM4_END)
        if item_1 is None:
            # Some 20-F filers use 10-K style Item 1
            item_1 = _extract_section(text, _10K_ITEM1_START, _10K_ITEM1_END)
        # Risk factors: Item 3D
        item_1a = _extract_section(text, _20F_ITEM3D_START, _20F_ITEM3D_END)
        # MD&A equivalent: Item 5 "Operating and Financial Review"
        item_7 = _extract_section(text, _20F_ITEM5_START, _20F_ITEM5_END)

    elif ft in ("40-F",):
        # Business: AIF Item 4 "Narrative Description of the Business"
        item_1 = _extract_section(text, _40F_ITEM4_START, _40F_ITEM4_END)
        if item_1 is None:
            item_1 = _extract_section(text, _10K_ITEM1_START, _10K_ITEM1_END)
            if item_1 is None:
                item_1 = _extract_section(text, _20F_ITEM4_START, _20F_ITEM4_END)
        # Risk factors: AIF Item 5
        item_1a = _extract_section(text, _40F_ITEM5_START, _40F_ITEM5_END)
        # 40-F filers rarely include a full MD&A in the AIF; Item 7 stays None

    else:
        # 10-K, 10-K405, unknown → standard 10-K patterns
        item_1  = _extract_section(text, _10K_ITEM1_START,  _10K_ITEM1_END)
        item_1a = _extract_section(text, _10K_ITEM1A_START, _10K_ITEM1A_END)
        item_7  = _extract_section(text, _10K_ITEM7_START,  _10K_ITEM7_END)
        # Fallback for modern iXBRL filings with custom section names
        if item_1 is None:
            item_1 = _extract_item1_fallback(text)
            if item_1 is not None:
                if item_1a is None:
                    item_1a = _extract_item1a_fallback(text)
                if item_7 is None:
                    item_7 = _extract_item7_fallback(text)

    def _wc(s: Optional[str]) -> int:
        return len(s.split()) if s else 0

    return {
        "item_1":  item_1,
        "item_1a": item_1a,
        "item_7":  item_7,
        "length_words": {
            "item_1":  _wc(item_1),
            "item_1a": _wc(item_1a),
            "item_7":  _wc(item_7),
        },
    }
