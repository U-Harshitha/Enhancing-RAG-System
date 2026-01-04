from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    corpus_dir: str = "database"
    persist_dir: str = "artifacts"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = 1200
    chunk_overlap: int = 200

    vector_top_k: int = 20
    bm25_top_k: int = 20
    fused_top_k: int = 20

    rrf_k: int = 60

    rerank: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 5

    use_llm_compression: bool = False
    ollama_model: str = "llama3.1:8b"

    answer_max_contexts: int = 5
