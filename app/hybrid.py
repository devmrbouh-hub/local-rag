"""Слияние векторного поиска и BM25 через RRF."""

from __future__ import annotations

from app.bm25_index import get_bm25_index
from app.config import HYBRID_RRF_K
from app.retrieve import search_vector


def merge_rrf(question: str, top_k: int, candidate_k: int, chunks_count: int) -> list[dict]:
    """top_k финальных чанков; candidate_k — глубина каждого списка кандидатов."""
    vec_hits = search_vector(question, top_k=candidate_k)
    bm25_index = get_bm25_index(expected_chunks=chunks_count)
    bm25_pairs: list[tuple[str, float]] = []
    if bm25_index is not None:
        bm25_pairs = bm25_index.query(question, top_k=candidate_k)

    rrf: dict[str, float] = {}
    for rank, hit in enumerate(vec_hits, start=1):
        rrf[hit["id"]] = rrf.get(hit["id"], 0.0) + 1.0 / (HYBRID_RRF_K + rank)
    for rank, (doc_id, _score) in enumerate(bm25_pairs, start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (HYBRID_RRF_K + rank)

    if not rrf:
        return []

    sorted_ids = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:top_k]
    vec_map = {h["id"]: h for h in vec_hits}
    out: list[dict] = []
    for doc_id in sorted_ids:
        if doc_id in vec_map:
            row = dict(vec_map[doc_id])
        elif bm25_index is not None:
            row = bm25_index.hit_dict(doc_id)
            if row is None:
                continue
        else:
            continue
        row["score"] = round(rrf[doc_id], 6)
        row["distance"] = None
        row["score_type"] = "rrf"
        out.append(row)
    return out
