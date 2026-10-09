"""Ручной прогон первых пяти контрольных вопросов через поиск.

Это не pytest: нужен живой индекс и Ollama. Для автотестов смотрите tests/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieve import search  # noqa: E402

QUESTIONS_PATH = ROOT / "docs" / "tk_rf" / "control_questions.md"


def load_questions(limit: int = 5) -> list[str]:
    """Нумерованные строки из control_questions.md, не больше limit."""
    text = QUESTIONS_PATH.read_text(encoding="utf-8")
    qs: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\d+\.\s+(.+)$", line.strip())
        if m:
            qs.append(m.group(1))
    return qs[:limit]


def main() -> int:
    for i, q in enumerate(load_questions(), start=1):
        print(f"\n=== Вопрос {i}: {q}")
        hits = search(q, top_k=5)
        for rank, hit in enumerate(hits, start=1):
            print(
                f"  {rank}. {hit['source_file']} ст.={hit['article']} "
                f"score={hit['score']} | {hit['text'][:120].replace(chr(10), ' ')}..."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
