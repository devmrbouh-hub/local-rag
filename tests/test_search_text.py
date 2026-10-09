from __future__ import annotations

from app.chunking import Chunk
from app.search_text import chunk_search_text


def test_chunk_search_text_adds_article_prefix() -> None:
    ch = Chunk(
        text="## Статья 115. Продолжительность ежегодного оплачиваемого отпуска\nТекст нормы.",
        source_file="glava.md",
        chunk_index=0,
        chapter="19",
        article="115",
    )
    out = chunk_search_text(ch)
    assert out.startswith("Статья 115. Продолжительность")
    assert "Текст нормы." in out
