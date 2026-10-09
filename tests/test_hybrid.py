"""Гибрид BM25 + RRF без Ollama."""

from __future__ import annotations

from app.bm25_index import build_index, tokenize
from app.hybrid import merge_rrf


def test_tokenize_keeps_digits() -> None:
    tokens = tokenize("28 календарных дней отпуска")
    assert "28" in tokens


def test_bm25_ranks_rare_term() -> None:
    filler = [f"filler doc number {i} about workplace" for i in range(8)]
    ids = [f"f:{i}" for i in range(8)] + ["target:0"]
    documents = filler + ["annual leave ZZTOP28ZZ calendar days"]
    metas = [{"article": str(i)} for i in range(8)] + [{"article": "115"}]
    search_texts = documents
    index = build_index(ids, documents, metas, search_texts, chunks_count=len(ids))
    hits = index.query("ZZTOP28ZZ", top_k=3)
    assert hits[0][0] == "target:0"
    assert hits[0][1] > hits[1][1]


def test_merge_rrf_prefers_bm25_only_id(monkeypatch) -> None:
    vec_hits = [
        {
            "id": "noise:0",
            "text": "other topic",
            "source_file": "x.md",
            "chunk_index": 0,
            "chapter": "",
            "article": "99",
            "distance": 0.2,
            "score": 0.8,
            "score_type": "cosine",
        }
    ]

    def fake_vector(_question: str, top_k: int) -> list[dict]:
        return vec_hits[:top_k]

    monkeypatch.setattr("app.hybrid.search_vector", fake_vector)

    ids = ["noise:0", "target:0"]
    documents = ["other topic", "business trip ZZTRIP167 without consent"]
    metas = [{"article": "99"}, {"article": "167"}]
    search_texts = documents
    bm25 = build_index(ids, documents, metas, search_texts, chunks_count=2)

    monkeypatch.setattr("app.hybrid.get_bm25_index", lambda expected_chunks=None: bm25)

    merged = merge_rrf("ZZTRIP167 business trip", top_k=2, candidate_k=5, chunks_count=2)
    articles = [h["article"] for h in merged]
    assert "167" in articles
    assert merged[0]["score_type"] == "rrf"
    assert merged[0]["distance"] is None
