from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document


@dataclass
class RerankResult:
    doc: Document
    score: float


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], *, top_n: int) -> List[RerankResult]:
        pairs = [(query, d.page_content) for d in docs]
        scores = self._model.predict(pairs)
        results = [RerankResult(doc=d, score=float(s)) for d, s in zip(docs, scores)]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]


def strip_rerank(results: List[RerankResult]) -> List[Document]:
    return [r.doc for r in results]
