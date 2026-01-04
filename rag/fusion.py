from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from langchain_core.documents import Document

from .stores import doc_key


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Document]],
    *,
    k: int = 60,
    top_n: int = 20,
) -> List[Tuple[Document, float]]:
    scores: Dict[Tuple[str, int, str], float] = defaultdict(float)
    first_doc: Dict[Tuple[str, int, str], Document] = {}

    for docs in ranked_lists:
        for rank, d in enumerate(docs, start=1):
            key = doc_key(d)
            first_doc.setdefault(key, d)
            scores[key] += 1.0 / (k + rank)

    fused = [(first_doc[key], score) for key, score in scores.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_n]


def strip_scores(fused: Iterable[Tuple[Document, float]]) -> List[Document]:
    return [d for d, _ in fused]
