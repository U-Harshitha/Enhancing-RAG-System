from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document

from .utils import jaccard, split_sentences, tokenize


@dataclass
class CompressedContext:
    doc: Document
    compressed_text: str


class HeuristicCompressor:
    def __init__(self, *, min_sentence_overlap: float = 0.08, max_sentences: int = 8):
        self.min_sentence_overlap = min_sentence_overlap
        self.max_sentences = max_sentences

    def compress(self, query: str, docs: List[Document]) -> List[CompressedContext]:
        q_tokens = tokenize(query)
        out: List[CompressedContext] = []
        for d in docs:
            sents = split_sentences(d.page_content or "")
            ranked = []
            for s in sents:
                score = jaccard(q_tokens, tokenize(s))
                if score >= self.min_sentence_overlap:
                    ranked.append((score, s))
            ranked.sort(key=lambda x: x[0], reverse=True)
            kept = [s for _, s in ranked[: self.max_sentences]]
            out.append(CompressedContext(doc=d, compressed_text=" ".join(kept) if kept else (d.page_content or "")))
        return out


class OllamaLLMCompressor:
    def __init__(self, *, model: str):
        from langchain_community.chat_models import ChatOllama

        self._llm = ChatOllama(model=model, temperature=0.0)

    def compress(self, query: str, docs: List[Document]) -> List[CompressedContext]:
        out: List[CompressedContext] = []
        for d in docs:
            prompt = (
                "You are compressing retrieved context for a RAG system. "
                "Given the user question and a context chunk, return only the sentences that are directly useful to answer the question. "
                "If nothing is useful, return an empty string.\n\n"
                f"Question: {query}\n\nContext chunk:\n{d.page_content}\n\nUseful sentences only:"
            )
            resp = self._llm.invoke(prompt)
            text = getattr(resp, "content", str(resp))
            out.append(CompressedContext(doc=d, compressed_text=(text or "").strip()))
        return out
