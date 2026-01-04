from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAGConfig


def load_pdfs(corpus_dir: str) -> List[Document]:
    base = Path(corpus_dir)
    pdfs = sorted(base.glob("*.pdf"))
    docs: List[Document] = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        for d in loader.load():
            d.metadata = dict(d.metadata or {})
            d.metadata["source"] = str(pdf)
            docs.append(d)
    return docs


def chunk_documents(docs: List[Document], *, chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata = dict(c.metadata or {})
        c.metadata["chunk_id"] = c.metadata.get("chunk_id") or f"chunk-{i}"
    return chunks


def write_corpus_manifest(chunks: List[Document], persist_dir: str, config: RAGConfig) -> str:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(persist_dir) / "corpus_manifest.json"
    payload = {
        "config": asdict(config),
        "num_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": c.metadata.get("chunk_id"),
                "source": c.metadata.get("source"),
                "page": c.metadata.get("page"),
                "text_preview": (c.page_content or "")[:300],
            }
            for c in chunks
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(manifest_path)
