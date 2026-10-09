"""Ingest на временном корпусе. Ollama не вызывается: embed подменён."""

from __future__ import annotations

from app.ingest import run_ingest
from tests.fakes import fake_embed


def test_run_ingest_writes_chunks(monkeypatch, sample_docs_dir, tmp_path) -> None:
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr("app.ingest.embed", fake_embed)
    stats = run_ingest(sample_docs_dir)
    assert stats["files"] == 2
    assert stats["chunks"] >= 2
    assert (tmp_path / "chroma").is_dir()
