from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from rag.config import RAGConfig
from rag.generate import answer_with_ollama
from rag.pipelines import AdvancedRAG, BaselineRAG


def _load_eval_results(persist_dir: str) -> pd.DataFrame | None:
    path = Path(persist_dir) / "eval_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_doc(d) -> str:
    md = d.metadata or {}
    src = md.get("source", "")
    page = md.get("page", "")
    chunk_id = md.get("chunk_id", "")
    return f"{src} | page {page} | {chunk_id}".strip()


@st.cache_resource
def _get_pipelines(corpus_dir: str, persist_dir: str, ollama_model: str, rerank: bool, llm_compress: bool):
    base_cfg = RAGConfig(corpus_dir=corpus_dir, persist_dir=persist_dir, ollama_model=ollama_model)
    adv_cfg = RAGConfig(
        corpus_dir=corpus_dir,
        persist_dir=persist_dir,
        ollama_model=ollama_model,
        rerank=rerank,
        use_llm_compression=llm_compress,
    )
    baseline = BaselineRAG(base_cfg)
    advanced = AdvancedRAG(adv_cfg)
    return baseline, advanced


def _safe_load(pipeline) -> str | None:
    try:
        pipeline.load()
        return None
    except Exception as e:
        return str(e)


def main() -> None:
    st.set_page_config(page_title="Enhancing RAG System", layout="wide")
    st.title("Enhancing-RAG-System")

    with st.sidebar:
        st.header("Settings")
        corpus_dir = st.text_input("Corpus folder", value="database")
        persist_dir = st.text_input("Artifacts folder", value="artifacts")
        ollama_model = st.text_input("Ollama model", value="llama3.1:8b")
        rerank = st.toggle("Enable cross-encoder reranking (bge-reranker-base)", value=False)
        llm_compress = st.toggle("Enable LLM compression (Ollama)", value=False)
        st.caption("Tip: run `python ingest.py --pipeline advanced` first to build indexes.")

    baseline, advanced = _get_pipelines(corpus_dir, persist_dir, ollama_model, rerank, llm_compress)

    err_b = _safe_load(baseline)
    err_a = _safe_load(advanced)

    if err_b or err_a:
        st.warning(
            "Indexes not found or failed to load. Build them first by running:\n\n"
            "`python ingest.py --pipeline advanced --corpus_dir database --persist_dir artifacts`\n\n"
            "Then come back and refresh."
        )
        with st.expander("Details"):
            if err_b:
                st.code(f"Baseline load error: {err_b}")
            if err_a:
                st.code(f"Advanced load error: {err_a}")
        st.stop()

    default_q = "Compare RMSNorm in LLaMA to LayerNorm in the original Transformer. What is normalized and why is RMSNorm more efficient?"
    question = st.text_area("Question", value=default_q, height=110)

    if st.button("Run both pipelines", type="primary"):
        col1, col2 = st.columns(2)

        with st.spinner("Baseline: retrieving..."):
            base_docs = baseline.retrieve(question)
        with st.spinner("Baseline: generating answer..."):
            base_answer = answer_with_ollama(question, base_docs[:5], model=ollama_model)

        with st.spinner("Advanced: retrieving (hybrid + RRF + optional rerank + compression)..."):
            adv_docs = advanced.retrieve(question)
        with st.spinner("Advanced: generating answer..."):
            adv_answer = answer_with_ollama(question, adv_docs[:5], model=ollama_model)

        with col1:
            st.subheader("Baseline RAG")
            st.markdown(base_answer)
            st.markdown("### Retrieved chunks")
            for i, d in enumerate(base_docs[:5], start=1):
                with st.expander(f"[{i}] {_format_doc(d)}", expanded=i == 1):
                    st.write(d.page_content)

        with col2:
            st.subheader("Improved RAG")
            st.markdown(adv_answer)
            st.markdown("### Retrieved (compressed) chunks")
            for i, d in enumerate(adv_docs[:5], start=1):
                with st.expander(f"[{i}] {_format_doc(d)}", expanded=i == 1):
                    st.write(d.page_content)

    st.divider()
    st.subheader("RAGAS Evaluation Metrics")
    df = _load_eval_results(persist_dir)
    if df is None:
        st.info(
            "No evaluation results found. Run:\n\n"
            "`python evaluate.py --persist_dir artifacts --ollama_model llama3.1:8b --rebuild`\n\n"
            "Then refresh this page."
        )
    else:
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
