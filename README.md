# Enhancing-RAG-System
Context-Aware RAG Retriever

## Setup (Local / Free)

### 1) Create a Python environment

```
python -m venv .venv
```

Activate:

- Windows (PowerShell)

```
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```
pip install -r requirements.txt
```

### 3) Install Ollama (for local generation + evaluation)

Install Ollama and pull a model (example):

```
ollama pull llama3.1:8b
```

## Data

Put your PDFs in:

```
database/
```

## Build indexes (ingestion)

Baseline (FAISS only):

```
python ingest.py --pipeline baseline --corpus_dir database --persist_dir artifacts
```

Improved (FAISS + BM25 for hybrid retrieval):

```
python ingest.py --pipeline advanced --corpus_dir database --persist_dir artifacts
```

## Run evaluation (RAGAS)

This runs both pipelines on 5 technical queries and outputs:

- Context Precision
- Context Recall
- Faithfulness
- Answer Relevance

```
python evaluate.py --persist_dir artifacts --ollama_model llama3.1:8b --rebuild
```

Optional flags:

- `--rerank` enables cross-encoder reranking (BAAI/bge-reranker-base)
- `--llm_compress` enables LLM-based context compression (Ollama)

## Streamlit UI

```
streamlit run streamlit_app.py
```

The UI shows:

- Left: Baseline RAG retrieved chunks + grounded answer
- Right: Improved RAG (hybrid + optional rerank + compression)
- Bottom: the latest `artifacts/eval_results.csv` metrics
