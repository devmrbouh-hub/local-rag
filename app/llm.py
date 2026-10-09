"""Связка поиск → промпт → ответ модели.

HTTP-коды здесь не выбираются. Пустой индекс — IndexNotReadyError, сбой Ollama — OllamaError.
"""

from __future__ import annotations

from app.config import (
    ASK_DEFAULT_TOP_K,
    ASK_PROMPT_CHUNKS,
    ASK_SOURCES_MAX,
    SOURCE_CHUNK_CHARS,
)
from app.ollama_client import chat
from app.prompts import build_rag_messages, excerpt_for_source
from app.retrieve import IndexNotReadyError, get_index_stats, search


def hits_to_sources(hits: list[dict], limit: int | None = None) -> list[dict]:
    """Контракт POST /ask: file, chunk, score. chunk — короткая выдержка, не весь фрагмент."""
    cap = limit if limit is not None else ASK_SOURCES_MAX
    sources: list[dict] = []
    for hit in hits[:cap]:
        sources.append(
            {
                "file": hit.get("source_file") or "",
                "chunk": excerpt_for_source(hit.get("text") or "", SOURCE_CHUNK_CHARS),
                "score": hit.get("score", 0.0),
                "article": hit.get("article") or "",
            }
        )
    return sources


def answer_question(question: str, top_k: int | None = None) -> tuple[str, list[dict]]:
    """Ответ и источники.

    top_k — сколько фрагментов искать в Chroma (широкий поиск).
    В промпт и в sources по умолчанию попадает меньше лучших — иначе 3B смешивает статьи.
    """
    if not get_index_stats()["ready"]:
        raise IndexNotReadyError(
            "Индекс не построен. Сначала выполните POST /ingest или python -m app.ingest_cli"
        )
    k = top_k if top_k is not None else ASK_DEFAULT_TOP_K
    hits = search(question, top_k=k)
    prompt_hits = hits[:ASK_PROMPT_CHUNKS]
    answer = chat(build_rag_messages(question, prompt_hits))
    answer = answer.strip()
    return answer, hits_to_sources(hits)
