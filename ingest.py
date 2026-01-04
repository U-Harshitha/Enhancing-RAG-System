from __future__ import annotations

import argparse

from rag.config import RAGConfig
from rag.pipelines import AdvancedRAG, BaselineRAG


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=["baseline", "advanced"], default="baseline")
    p.add_argument("--corpus_dir", default="database")
    p.add_argument("--persist_dir", default="artifacts")
    args = p.parse_args()

    cfg = RAGConfig(corpus_dir=args.corpus_dir, persist_dir=args.persist_dir)

    if args.pipeline == "baseline":
        rag = BaselineRAG(cfg)
    else:
        rag = AdvancedRAG(cfg)

    info = rag.build()
    print(info)


if __name__ == "__main__":
    main()
