"""Оценка retrieval и, отдельно, ответов /ask-логики.

Нужны построенный индекс и Ollama. Это не pytest.
--retrieval-only считает, попала ли ожидаемая статья в top-k.
--ask дополнительно зовёт модель; на вопросах вне кодекса ждём фразу «нет информации».
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm import answer_question  # noqa: E402
from app.prompts import NO_CONTEXT_ANSWER  # noqa: E402
from app.retrieve import search  # noqa: E402

QUESTIONS_PATH = ROOT / "docs" / "tk_rf" / "control_questions.md"
EXPECTATIONS_PATH = ROOT / "docs" / "tk_rf" / "control_expectations.json"

# Вопросы не из ТК: удачный ответ — отказ, а не выдуманная статья.
OFF_TOPIC = [
    "Какой штраф по УК РФ за увольнение работника?",
    "Какая ставка НДС на отпускные в 2026 году?",
    "Как оформить визу для командировки в Японию?",
]


def load_questions() -> list[str]:
    text = QUESTIONS_PATH.read_text(encoding="utf-8")
    questions: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\d+\.\s+(.+)$", line.strip())
        if match:
            questions.append(match.group(1))
    return questions


def load_expectations() -> dict[int, dict]:
    raw = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    return {int(item["n"]): item for item in raw}


def article_in_hits(hits: list[dict], article: str) -> bool:
    return any(str(hit.get("article") or "") == article for hit in hits)


def retrieval_report(questions: list[str], expectations: dict[int, dict], top_k: int) -> tuple[int, int]:
    hits_ok = 0
    print(f"\n## Retrieval top-{top_k}")
    print("| # | статья | в top-k | файл первого |")
    print("|---|--------|---------|--------------|")
    for n, question in enumerate(questions, start=1):
        expected = expectations.get(n)
        if not expected:
            continue
        found = search(question, top_k=top_k)
        ok = article_in_hits(found, expected["article"])
        hits_ok += int(ok)
        first = found[0]["source_file"] if found else "—"
        mark = "да" if ok else "нет"
        print(f"| {n} | {expected['article']} | {mark} | {first} |")
    total = sum(1 for n in range(1, len(questions) + 1) if n in expectations)
    print(f"\nПопадание статьи: {hits_ok}/{total}")
    return hits_ok, total


def ask_report(questions: list[str]) -> None:
    print("\n## Ответы модели")
    for n, question in enumerate(questions, start=1):
        answer, sources = answer_question(question)
        article = sources[0]["article"] if sources else "—"
        preview = answer.replace("\n", " ")[:180]
        print(f"\n{n}. {question}", flush=True)
        print(f"   ответ: {preview}", flush=True)
        print(f"   первая статья в источниках: {article}", flush=True)

    refused = 0
    print("\n## Вопросы вне кодекса")
    for question in OFF_TOPIC:
        answer, _sources = answer_question(question)
        ok = NO_CONTEXT_ANSWER in answer
        refused += int(ok)
        print(f"- {'отказ' if ok else 'не отказ'}: {question}")
        print(f"  {answer.replace(chr(10), ' ')[:180]}")
    print(f"\nОтказов: {refused}/{len(OFF_TOPIC)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка RAG по контрольным вопросам")
    parser.add_argument("--retrieval-only", action="store_true", help="Только поиск, без LLM")
    parser.add_argument("--ask", action="store_true", help="Поиск и генерация ответа")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--compare-k",
        action="store_true",
        help="Для retrieval посчитать и top-5, и top-10",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "vector"),
        default=None,
        help="Режим поиска: hybrid (RRF) или только vector (без смены .env)",
    )
    args = parser.parse_args()
    if args.mode == "vector":
        os.environ["HYBRID_SEARCH"] = "false"
    elif args.mode == "hybrid":
        os.environ["HYBRID_SEARCH"] = "true"
    if not args.retrieval_only and not args.ask:
        args.retrieval_only = True

    questions = load_questions()
    expectations = load_expectations()
    if args.retrieval_only or args.compare_k:
        if args.compare_k:
            retrieval_report(questions, expectations, 5)
            retrieval_report(questions, expectations, 10)
        else:
            retrieval_report(questions, expectations, args.top_k)
    if args.ask:
        ask_report(questions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
