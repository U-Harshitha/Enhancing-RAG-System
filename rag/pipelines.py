from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .compress import HeuristicCompressor, OllamaLLMCompressor
from .config import RAGConfig
from .fusion import rrf_fuse, strip_scores
from .ingest import chunk_documents, load_pdfs, write_corpus_manifest
from .rerank import CrossEncoderReranker, strip_rerank
from .stores import build_bm25, build_faiss, load_bm25, load_faiss, save_bm25, save_faiss


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


class BaselineRAG:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._emb = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        self._vs = None
        self._chunks: List[Document] = []

    @property
    def embeddings(self):
        return self._emb

    def build(self) -> Dict[str, Any]:
        _ensure_dir(self.config.persist_dir)
        docs = load_pdfs(self.config.corpus_dir)
        chunks = chunk_documents(docs, chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
        self._chunks = chunks
        manifest = write_corpus_manifest(chunks, self.config.persist_dir, self.config)
        vs = build_faiss(chunks, self._emb)
        save_faiss(vs, self.config.persist_dir)
        self._vs = vs
        return {"manifest": manifest, "num_chunks": len(chunks), "config": asdict(self.config)}

    def load(self) -> None:
        self._vs = load_faiss(self.config.persist_dir, self._emb)

    def retrieve(self, query: str, *, k: Optional[int] = None) -> List[Document]:
        if self._vs is None:
            self.load()
        retriever = self._vs.as_retriever(search_kwargs={"k": k or self.config.vector_top_k})
        return retriever.get_relevant_documents(query)


class AdvancedRAG:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._emb = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        self._vs = None
        self._bm25 = None
        self._chunks: List[Document] = []

        self._reranker = CrossEncoderReranker(self.config.reranker_model) if self.config.rerank else None
        if self.config.use_llm_compression:
            self._compressor = OllamaLLMCompressor(model=self.config.ollama_model)
        else:
            self._compressor = HeuristicCompressor()

    @property
    def embeddings(self):
        return self._emb

    def build(self) -> Dict[str, Any]:
        _ensure_dir(self.config.persist_dir)
        docs = load_pdfs(self.config.corpus_dir)
        chunks = chunk_documents(docs, chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
        self._chunks = chunks
        manifest = write_corpus_manifest(chunks, self.config.persist_dir, self.config)

        vs = build_faiss(chunks, self._emb)
        save_faiss(vs, self.config.persist_dir)
        self._vs = vs

        bm25 = build_bm25(chunks, k=self.config.bm25_top_k)
        save_bm25(bm25, self.config.persist_dir)
        self._bm25 = bm25

        return {"manifest": manifest, "num_chunks": len(chunks), "config": asdict(self.config)}

    def load(self) -> None:
        self._vs = load_faiss(self.config.persist_dir, self._emb)
        self._bm25 = load_bm25(self.config.persist_dir)

    def retrieve(self, query: str) -> List[Document]:
        if self._vs is None or self._bm25 is None:
            self.load()

        vector_docs = self._vs.as_retriever(search_kwargs={"k": self.config.vector_top_k}).get_relevant_documents(query)
        bm25_docs = self._bm25.get_relevant_documents(query)

        fused = rrf_fuse([vector_docs, bm25_docs], k=self.config.rrf_k, top_n=self.config.fused_top_k)
        docs = strip_scores(fused)

        if self._reranker is not None:
            rr = self._reranker.rerank(query, docs, top_n=self.config.rerank_top_n)
            docs = strip_rerank(rr)
        else:
            docs = docs[: self.config.rerank_top_n]

        compressed = self._compressor.compress(query, docs)
        out: List[Document] = []
        for c in compressed:
            d = Document(page_content=c.compressed_text, metadata=dict(c.doc.metadata or {}))
            out.append(d)
        return out
