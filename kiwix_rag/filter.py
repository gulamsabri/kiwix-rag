from __future__ import annotations
import re

_PRICE_RE = re.compile(r'\$\s*\d')

_AD_STRONG = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bpostpaid\b|\bppd\b',
        r'send\s+\$|send\b.{0,30}\$\s*\d',
        r'\bsend\s+sase\b|send\s+stamped\s+(?:self.?addressed|envelope)',
        r'write\s+for\s+(?:free\s+)?(?:catalog|brochure|flyer|information|prices?|list)',
        r'\bitems?\s+for\s+sale\b',
    ]
]

_AD_MODERATE = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bfor\s+sale\b',
        r'\bwanted\b',
        r'(?:plus|add)\s+\$?\d+.*?postage',
        r'\ball\s+orders?\b',
        r'\bask(?:ing)?\s+\$\s*\d',
    ]
]

_CONSPIRACY_PATTERNS = [
    "chemtrail", "chem trail", "deep state", "new world order", r"\bnwo\b",
    "globalist", "plandemic", r"\bvaxxed\b", "anti-vaxxer", "sheeple",
    "false flag", "fema camp", "reptilian", r"\billuminati\b", "crisis actor",
    r"\bpsyop\b", "nanobots", "adrenochrome", "flat earth", "sandy hook hoax",
    r"\bqanon\b", r"\bq anon\b", "great reset conspiracy", "depopulation agenda",
    "microchip vaccine", "5g microchip", "bill gates depopulation",
    "george soros agenda", r"\bsatanic cabal\b", "lizard people",
    "moon landing hoax", "nasa hoax", "population control agenda",
    r"\bwoke agenda\b", r"\bclimate hoax\b",
]
_CONSPIRACY_RE = [re.compile(p, re.IGNORECASE) for p in _CONSPIRACY_PATTERNS]

DEFAULT_THRESHOLD = 2


class ChunkFilter:
    """Score text chunks for noise (ads, conspiracy content)."""

    def score(self, text: str) -> tuple[int, list[str]]:
        """Return (score, reasons). Higher score = more likely noise."""
        s, reasons = 0, []
        hits = len(_PRICE_RE.findall(text))
        if hits >= 2:
            s += 2
            reasons.append(f"multiple prices ({hits} hits)")
        for pat in _AD_STRONG:
            if pat.search(text):
                s += 2
                reasons.append(f"ad (strong): {pat.pattern}")
        for pat in _AD_MODERATE:
            if pat.search(text):
                s += 1
                reasons.append(f"ad: {pat.pattern}")
        for pat in _CONSPIRACY_RE:
            if pat.search(text):
                s += 1
                reasons.append(f"conspiracy: {pat.pattern}")
        return s, reasons

    def is_clean(self, text: str, threshold: int = DEFAULT_THRESHOLD) -> bool:
        score, _ = self.score(text)
        return score < threshold
