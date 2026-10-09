"""Поиск по уже построенному индексу Chroma.

Модуль считает эмбеддинг вопроса и возвращает ближайшие чанки.
Генерацию ответа здесь не делаем — это слой llm.py.
"""

from __future__ import annotations

import chromadb

from app.config import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    HYBRID_VECTOR_K,
    get_chroma_dir,
    hybrid_search_enabled,
)
from app.ollama_client import embed


class IndexNotReadyError(RuntimeError):
    """Индекса нет или он пуст. HTTP-слой переводит это в 404, не в 503."""


def _open_collection() -> chromadb.Collection | None:
    chroma_dir = get_chroma_dir()
    if not chroma_dir.is_dir():
        return None
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        return client.get_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        return None


def get_index_stats() -> dict:
    """Число чанков и флаг ready. Не бросает исключение: так устроен /health."""
    collection = _open_collection()
    if collection is None:
        return {"chunks": 0, "ready": False}
    try:
        count = int(collection.count())
    except Exception:
        return {"chunks": 0, "ready": False}
    return {"chunks": count, "ready": count > 0}


def _hit_from_chroma_row(
    doc_id: str,
    doc: str,
    meta: dict,
    distance: float,
) -> dict:
    return {
        "id": doc_id,
        "text": doc,
        "source_file": meta.get("source_file", ""),
        "chunk_index": meta.get("chunk_index", 0),
        "chapter": meta.get("chapter", ""),
        "article": meta.get("article", ""),
        "distance": distance,
        "score": round(1.0 - distance, 4),
        "score_type": "cosine",
    }


def search_vector(question: str, top_k: int) -> list[dict]:
    """Только Chroma + cosine. score = 1 - distance."""
    stats = get_index_stats()
    if not stats["ready"]:
        raise IndexNotReadyError(
            "Индекс не построен. Сначала выполните ingest: python -m app.ingest_cli"
        )
    collection = _open_collection()
    if collection is None:
        raise IndexNotReadyError(
            "Индекс не построен. Сначала выполните ingest: python -m app.ingest_cli"
        )

    k = min(top_k, int(collection.count()))
    if k <= 0:
        raise IndexNotReadyError("Индекс пуст. Запустите ingest заново.")

    q_emb = embed(question)
    result = collection.query(query_embeddings=[q_emb], n_results=k)

    out: list[dict] = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    for i, doc_id in enumerate(ids):
        distance = float(dists[i]) if dists else 0.0
        meta = metas[i] or {}
        out.append(_hit_from_chroma_row(doc_id, docs[i], meta, distance))
    return out


def search(question: str, top_k: int | None = None) -> list[dict]:
    """Ближайшие чанки. При HYBRID_SEARCH=true — RRF vector + BM25."""
    k = top_k or DEFAULT_TOP_K
    stats = get_index_stats()
    if not stats["ready"]:
        raise IndexNotReadyError(
            "Индекс не построен. Сначала выполните ingest: python -m app.ingest_cli"
        )

    if hybrid_search_enabled():
        from app.hybrid import merge_rrf

        candidate_k = min(HYBRID_VECTOR_K, stats["chunks"])
        return merge_rrf(question, top_k=k, candidate_k=candidate_k, chunks_count=stats["chunks"])

    return search_vector(question, top_k=k)
