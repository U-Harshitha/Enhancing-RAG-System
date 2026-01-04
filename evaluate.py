from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from rag.config import RAGConfig
from rag.generate import answer_with_ollama
from rag.pipelines import AdvancedRAG, BaselineRAG


def _to_metrics_dict(result: Any) -> Dict[str, float]:
    if isinstance(result, dict):
        return {k: float(v) for k, v in result.items()}
    if hasattr(result, "__iter__") and not hasattr(result, "to_pandas"):
        try:
            return {k: float(v) for k, v in dict(result).items()}
        except Exception:
            pass
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        # Expect one row per sample; average columns to get a single score per metric.
        out: Dict[str, float] = {}
        for c in df.columns:
            if c in {"question", "answer", "contexts", "ground_truth", "ground_truths"}:
                continue
            try:
                out[c] = float(df[c].mean())
            except Exception:
                continue
        return out
    # Fallback: try attribute access
    try:
        return {k: float(getattr(result, k)) for k in dir(result) if not k.startswith("_")}
    except Exception:
        return {}


DEFAULT_QUERIES: List[Dict[str, str]] = [
    {
        "question": "Compare RMSNorm in LLaMA to LayerNorm in the original Transformer. What is normalized, where is the mean used, and why is RMSNorm considered more efficient?",
        "ground_truth": "LayerNorm normalizes using mean and variance across hidden features, while RMSNorm normalizes using only the root mean square (no mean subtraction). RMSNorm is cheaper because it avoids mean computation and uses only RMS scaling.",
    },
    {
        "question": "How does LoRA reduce trainable parameters compared to full fine-tuning, and what is the role of low-rank decomposition in its design?",
        "ground_truth": "LoRA freezes the base weights and injects trainable low-rank matrices whose product forms a low-rank update to weight matrices, reducing the number of trainable parameters while maintaining adaptation capacity.",
    },
    {
        "question": "Explain the key idea behind Attention is All You Need and how self-attention enables parallelization compared to RNNs.",
        "ground_truth": "The Transformer replaces recurrence with self-attention, letting each token attend to all others; this removes sequential dependencies in computation and enables parallel processing across sequence positions.",
    },
    {
        "question": "What are the core differences between instruction tuning and RLHF for aligning LLM behavior, and why might they be combined?",
        "ground_truth": "Instruction tuning uses supervised fine-tuning on instruction-response pairs; RLHF optimizes against a learned reward model with preference data. They can be combined to get a good supervised starting point and then refine behavior with preference optimization.",
    },
    {
        "question": "In GPT-3 style scaling, what is the relationship between model size, data, and performance, and how do scaling laws guide training decisions?",
        "ground_truth": "Scaling laws show predictable improvements with increased parameters, data, and compute; they guide how to allocate compute between bigger models and more data to achieve better loss/performance under compute constraints.",
    },
]


def run_pipeline(name: str, rag: Any, cfg: RAGConfig, questions: List[Dict[str, str]]) -> pd.DataFrame:
    rows = []
    for q in questions:
        docs = rag.retrieve(q["question"])
        contexts = [d.page_content for d in docs]
        answer = answer_with_ollama(q["question"], docs, model=cfg.ollama_model)
        rows.append(
            {
                "pipeline": name,
                "question": q["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": q["ground_truth"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_dir", default="database")
    p.add_argument("--persist_dir", default="artifacts")
    p.add_argument("--ollama_model", default="llama3.1:8b")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--llm_compress", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()

    cfg_base = RAGConfig(corpus_dir=args.corpus_dir, persist_dir=args.persist_dir, ollama_model=args.ollama_model)
    cfg_adv = RAGConfig(
        corpus_dir=args.corpus_dir,
        persist_dir=args.persist_dir,
        ollama_model=args.ollama_model,
        rerank=args.rerank,
        use_llm_compression=args.llm_compress,
    )

    baseline = BaselineRAG(cfg_base)
    advanced = AdvancedRAG(cfg_adv)

    if args.rebuild:
        baseline.build()
        advanced.build()
    else:
        baseline.load()
        advanced.load()

    df_base = run_pipeline("baseline", baseline, cfg_base, DEFAULT_QUERIES)
    df_adv = run_pipeline("advanced", advanced, cfg_adv, DEFAULT_QUERIES)

    dataset_base = Dataset.from_pandas(df_base.drop(columns=["pipeline"]))
    dataset_adv = Dataset.from_pandas(df_adv.drop(columns=["pipeline"]))

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]

    from langchain_community.chat_models import ChatOllama

    eval_llm = ChatOllama(model=args.ollama_model, temperature=0.0)

    res_base = evaluate(dataset_base, metrics=metrics, llm=eval_llm, embeddings=baseline.embeddings)
    res_adv = evaluate(dataset_adv, metrics=metrics, llm=eval_llm, embeddings=advanced.embeddings)

    base_scores = _to_metrics_dict(res_base)
    adv_scores = _to_metrics_dict(res_adv)

    summary = pd.DataFrame(
        [
            {"pipeline": "baseline", **base_scores},
            {"pipeline": "advanced", **adv_scores},
        ]
    )
    print(summary)

    out_dir = args.persist_dir
    import os

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_results.csv")
    summary.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
