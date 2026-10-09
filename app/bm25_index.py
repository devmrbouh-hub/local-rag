"""Лексический индекс BM25: pickle на диске и кэш в процессе."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

import chromadb

from app.config import COLLECTION_NAME, get_bm25_dir, get_chroma_dir


def _open_collection() -> chromadb.Collection | None:
    chroma_dir = get_chroma_dir()
    if not chroma_dir.is_dir():
        return None
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        return client.get_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        return None

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Простая токенизация для русского: lower + слова и цифры."""
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1 or t.isdigit()]


@dataclass
class Bm25Index:
    chunks_count: int
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    _bm25: BM25Okapi
    _tokenized: list[list[str]]

    def query(self, question: str, top_k: int) -> list[tuple[str, float]]:
        tokens = tokenize(question)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.ids[idx], float(score)) for idx, score in ranked[:top_k]]

    def hit_dict(self, doc_id: str) -> dict | None:
        try:
            idx = self.ids.index(doc_id)
        except ValueError:
            return None
        meta = self.metadatas[idx] or {}
        return {
            "id": doc_id,
            "text": self.documents[idx],
            "source_file": meta.get("source_file", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "chapter": meta.get("chapter", ""),
            "article": meta.get("article", ""),
            "distance": None,
        }


_cached: Bm25Index | None = None


def invalidate_bm25_cache() -> None:
    """После ingest или смены CHROMA_DIR — сбросить кэш в памяти."""
    global _cached
    _cached = None


def _pickle_path() -> Path:
    return get_bm25_dir() / "index.pkl"


def build_index(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    search_texts: list[str],
    chunks_count: int,
) -> Bm25Index:
    tokenized = [tokenize(t) for t in search_texts]
    bm25 = BM25Okapi(tokenized)
    return Bm25Index(
        chunks_count=chunks_count,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        _bm25=bm25,
        _tokenized=tokenized,
    )


def save_index(index: Bm25Index, path: Path | None = None) -> Path:
    path = path or _pickle_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks_count": index.chunks_count,
        "ids": index.ids,
        "documents": index.documents,
        "metadatas": index.metadatas,
        "tokenized": index._tokenized,
    }
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return path


def load_index_from_file(path: Path | None = None) -> Bm25Index | None:
    path = path or _pickle_path()
    if not path.is_file():
        return None
    payload: dict[str, Any] = pickle.loads(path.read_bytes())
    tokenized = payload["tokenized"]
    return Bm25Index(
        chunks_count=int(payload["chunks_count"]),
        ids=list(payload["ids"]),
        documents=list(payload["documents"]),
        metadatas=list(payload["metadatas"]),
        _bm25=BM25Okapi(tokenized),
        _tokenized=tokenized,
    )


def _rebuild_from_chroma() -> Bm25Index | None:
    collection = _open_collection()
    if collection is None:
        return None
    count = collection.count()
    if count == 0:
        return None
    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    if not ids:
        return None
    # Fallback: BM25 по тому же тексту, что в Chroma (без enrich-префикса).
    index = build_index(ids, docs, metas, docs, chunks_count=count)
    save_index(index)
    return index


def get_bm25_index(expected_chunks: int | None = None) -> Bm25Index | None:
    """Кэш в процессе; при расхождении chunks_count — пересборка из Chroma."""
    global _cached
    if _cached is not None:
        if expected_chunks is None or _cached.chunks_count == expected_chunks:
            return _cached
        invalidate_bm25_cache()

    loaded = load_index_from_file()
    if loaded is not None and (expected_chunks is None or loaded.chunks_count == expected_chunks):
        _cached = loaded
        return _cached

    rebuilt = _rebuild_from_chroma()
    if rebuilt is not None:
        _cached = rebuilt
    return _cached
