from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever


def build_faiss(chunks: List[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


def save_faiss(store: FAISS, persist_dir: str) -> str:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(persist_dir) / "faiss_index")
    store.save_local(path)
    return path


def load_faiss(persist_dir: str, embeddings: Embeddings) -> FAISS:
    path = str(Path(persist_dir) / "faiss_index")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def build_bm25(chunks: List[Document], *, k: int) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def save_bm25(retriever: BM25Retriever, persist_dir: str) -> str:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    path = Path(persist_dir) / "bm25.pkl"
    path.write_bytes(pickle.dumps(retriever))
    return str(path)


def load_bm25(persist_dir: str) -> BM25Retriever:
    path = Path(persist_dir) / "bm25.pkl"
    return pickle.loads(path.read_bytes())


def doc_key(d: Document) -> Tuple[str, int, str]:
    md = d.metadata or {}
    return (str(md.get("source", "")), int(md.get("page", -1)), str(md.get("chunk_id", "")))
