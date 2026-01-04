from __future__ import annotations

import re
from typing import Iterable, List


_WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))
