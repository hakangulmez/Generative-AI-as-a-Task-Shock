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
    r"item\s*7\s*[Aa][\s.\-–—:]"
    r"|item\s*(?:8|VIII)[\s.\-–—:]",
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
