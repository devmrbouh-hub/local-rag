"""Поиск по фиктивным эмбеддингам: статья про отпуск должна быть первой."""

from __future__ import annotations

from app.ingest import run_ingest
from app.retrieve import search
from tests.fakes import fake_embed


def test_search_puts_vacation_article_first(monkeypatch, sample_docs_dir, tmp_path) -> None:
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr("app.ingest.embed", fake_embed)
    monkeypatch.setattr("app.retrieve.embed", fake_embed)
    run_ingest(sample_docs_dir)

    hits = search("какова продолжительность ежегодного отпуска", top_k=2)
    assert hits[0]["article"] == "115"
    assert hits[0]["source_file"] == "sample_doc.md"
    assert hits[0]["score"] >= hits[1]["score"]
