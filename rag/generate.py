from __future__ import annotations

from typing import List

from langchain_core.documents import Document


def format_citations(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        src = md.get("source", "")
        page = md.get("page", "")
        lines.append(f"[{i}] {src} (page {page})")
    return "\n".join(lines)


def build_prompt(question: str, docs: List[Document]) -> str:
    ctx = "\n\n".join([f"[Chunk {i}]\n{d.page_content}" for i, d in enumerate(docs, start=1)])
    cites = format_citations(docs)
    return (
        "You are a research assistant answering questions using only the provided context from research papers. "
        "If the context is insufficient, say you don't have enough evidence. "
        "Provide a concise technical answer and cite sources as [1], [2], ... where relevant.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Sources:\n{cites}\n\n"
        "Answer:"
    )


def answer_with_ollama(question: str, docs: List[Document], *, model: str) -> str:
    from langchain_community.chat_models import ChatOllama

    llm = ChatOllama(model=model, temperature=0.0)
    prompt = build_prompt(question, docs)
    resp = llm.invoke(prompt)
    return (getattr(resp, "content", str(resp)) or "").strip()
