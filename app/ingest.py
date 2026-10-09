"""Сборка индекса Chroma из markdown-корпуса.

Каждый запуск удаляет коллекцию и пишет её заново: частичного обновления нет.
Эмбеддинги считаются в коде и кладутся в Chroma вручную (не встроенной функцией Chroma).
"""

from __future__ import annotations

import time
from pathlib import Path

import chromadb

from app.bm25_index import build_index, invalidate_bm25_cache, save_index
from app.chunking import load_chunks_from_dir
from app.config import COLLECTION_NAME, get_chroma_dir, get_docs_dir
from app.ollama_client import embed
from app.search_text import chunk_search_text


def _get_client() -> chromadb.PersistentClient:
    chroma_dir = get_chroma_dir()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def _reset_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Новая коллекция с cosine: расстояние 0 — тот же смысл, чем больше — тем дальше."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        pass
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def run_ingest(path: Path | None = None) -> dict:
    """Индексирует каталог. Возвращает files, chunks, seconds и путь к индексу.

    Вызов синхронный: на полном ТК это несколько минут из-за эмбеддингов.
    """
    docs_path = path or get_docs_dir()
    if not docs_path.is_dir():
        raise FileNotFoundError(f"Папка документов не найдена: {docs_path}")

    t0 = time.perf_counter()
    chunks = load_chunks_from_dir(docs_path)
    if not chunks:
        raise RuntimeError(f"Нет чанков в {docs_path}")

    client = _get_client()
    collection = _reset_collection(client)

    files_seen = {c.source_file for c in chunks}
    progress_every = 50

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    search_texts: list[str] = []
    embeddings: list[list[float]] = []

    for i, ch in enumerate(chunks):
        if i > 0 and i % progress_every == 0:
            print(f"  эмбеддинги: {i}/{len(chunks)} чанков...")
        chunk_id = f"{ch.source_file}:{ch.chunk_index}"
        ids.append(chunk_id)
        documents.append(ch.text)
        stext = chunk_search_text(ch)
        search_texts.append(stext)
        metadatas.append(
            {
                "source_file": ch.source_file,
                "chunk_index": ch.chunk_index,
                "chapter": ch.chapter or "",
                "article": ch.article or "",
            }
        )
        # Эмбеддинг по обогащённому тексту; в Chroma — цитата ch.text.
        embeddings.append(embed(stext))

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    bm25 = build_index(ids, documents, metadatas, search_texts, chunks_count=len(chunks))
    save_index(bm25)
    invalidate_bm25_cache()

    elapsed = time.perf_counter() - t0
    return {
        "files": len(files_seen),
        "chunks": len(chunks),
        "seconds": round(elapsed, 2),
        "chroma_dir": str(get_chroma_dir()),
    }
