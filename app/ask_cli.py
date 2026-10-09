"""Вопрос в терминал: поиск + ответ модели (POST /ask без HTTP)."""

from __future__ import annotations

import argparse
import sys

from app.config import ASK_DEFAULT_TOP_K
from app.llm import answer_question
from app.ollama_client import OllamaError
from app.retrieve import IndexNotReadyError


def main() -> int:
    parser = argparse.ArgumentParser(description="Вопрос по ТК: ответ модели и источники в терминале")
    parser.add_argument("question", help="Вопрос")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Сколько фрагментов искать в Chroma (по умолчанию {ASK_DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    try:
        answer, sources = answer_question(args.question, top_k=args.top_k)
    except IndexNotReadyError as exc:
        print(exc, file=sys.stderr)
        return 1
    except OllamaError as exc:
        print(exc, file=sys.stderr)
        return 1

    # UTF-8 в Windows Terminal / PowerShell 7 обычно ок; в PS 5.1 при кракозябрах: chcp 65001
    print("\n--- Ответ ---\n")
    print(answer)
    print("\n--- Источники ---\n")
    for rank, src in enumerate(sources, start=1):
        art = src.get("article") or "—"
        print(f"[{rank}] ст. {art}  score={src.get('score')}  file={src.get('file')}")
        chunk = (src.get("chunk") or "").replace("\n", " ")
        print(f"    {chunk}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
