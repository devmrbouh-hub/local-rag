"""Поиск по индексу без генерации. Ответ LLM даёт только POST /ask."""

from __future__ import annotations

import argparse

from app.config import DEFAULT_TOP_K
from app.retrieve import search


def main() -> int:
    parser = argparse.ArgumentParser(description="Поиск по индексу Chroma без LLM")
    parser.add_argument("question", help="Вопрос")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Сколько фрагментов показать")
    args = parser.parse_args()

    hits = search(args.question, top_k=args.top_k)
    for rank, hit in enumerate(hits, start=1):
        snippet = hit["text"].replace("\n", " ")[:200]
        print(f"\n[{rank}] score={hit['score']} distance={hit['distance']:.4f}")
        print(f"    file={hit['source_file']} article={hit['article']} chapter={hit['chapter']}")
        print(f"    {snippet}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
