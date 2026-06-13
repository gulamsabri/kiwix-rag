#!/usr/bin/env python3
"""
Shared chunk quality filter.

Imported by extract_zim.py (--filter flag) and filter_survivorlibrary.py.
Scores chunks against two categories of noise:
  - Classified ads / equipment-for-sale content (vintage magazines, store pages)
  - Conspiracy / misinformation keywords

score_chunk(text) -> (int, list[str])   full scoring with reasons
is_clean(text, threshold)  -> bool      convenience wrapper
"""

import re

_PRICE_RE = re.compile(r'\$\s*\d')

# Strong ad signals (+2 each)
_AD_STRONG_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bpostpaid\b|\bppd\b',
        r'send\s+\$|send\b.{0,30}\$\s*\d',
        r'\bsend\s+sase\b|send\s+stamped\s+(?:self.?addressed|envelope)',
        r'write\s+for\s+(?:free\s+)?(?:catalog|brochure|flyer|information|prices?|list)',
        r'\bitems?\s+for\s+sale\b',
    ]
]

# Moderate ad signals (+1 each)
_AD_MODERATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bfor\s+sale\b',
        r'\bwanted\b',
        r'(?:plus|add)\s+\$?\d+.*?postage',
        r'\ball\s+orders?\b',
        r'\bask(?:ing)?\s+\$\s*\d',
    ]
]

# Conspiracy / misinformation keywords (+1 each)
_CONSPIRACY_KEYWORDS = [
    "chemtrail", "chem trail",
    "deep state",
    "new world order",
    r"\bnwo\b",
    "globalist",
    "plandemic",
    r"\bvaxxed\b", "anti-vaxxer",
    "sheeple",
    "false flag",
    "fema camp",
    "reptilian",
    r"\billuminati\b",
    "crisis actor",
    r"\bpsyop\b",
    "nanobots",
    "adrenochrome",
    "flat earth",
    "sandy hook hoax",
    r"\bqanon\b", r"\bq anon\b",
    "great reset conspiracy",
    "depopulation agenda",
    "microchip vaccine",
    "5g microchip",
    "bill gates depopulation",
    "george soros agenda",
    r"\bsatanic cabal\b",
    "lizard people",
    "moon landing hoax",
    "nasa hoax",
    "population control agenda",
    r"\bwoke agenda\b",
    r"\bclimate hoax\b",
]
_CONSPIRACY_RE = [re.compile(p, re.IGNORECASE) for p in _CONSPIRACY_KEYWORDS]

DEFAULT_THRESHOLD = 2


def score_chunk(text: str) -> tuple[int, list[str]]:
    """Return (score, reasons). Higher score = more likely to be noise."""
    score = 0
    reasons = []

    price_hits = len(_PRICE_RE.findall(text))
    if price_hits >= 2:
        score += 2
        reasons.append(f"multiple prices ({price_hits} hits)")

    for pat in _AD_STRONG_PATTERNS:
        if pat.search(text):
            score += 2
            reasons.append(f"ad (strong): {pat.pattern}")

    for pat in _AD_MODERATE_PATTERNS:
        if pat.search(text):
            score += 1
            reasons.append(f"ad: {pat.pattern}")

    for pat in _CONSPIRACY_RE:
        if pat.search(text):
            score += 1
            reasons.append(f"conspiracy: {pat.pattern}")

    return score, reasons


def is_clean(text: str, threshold: int = DEFAULT_THRESHOLD) -> bool:
    score, _ = score_chunk(text)
    return score < threshold
