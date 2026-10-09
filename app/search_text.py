"""Текст чанка для поиска: эмбеддинг и BM25 (не то, что показываем в sources)."""

from __future__ import annotations

import re

from app.chunking import Chunk

_ARTICLE_HEAD_RE = re.compile(
    r"^##\s+Статья\s+(\d+(?:\.\d+)?)\.\s*(.+?)\s*$",
    re.MULTILINE,
)


def chunk_search_text(chunk: Chunk) -> str:
    """Статья + название нормы + тело. В Chroma documents остаётся chunk.text без изменений."""
    text = chunk.text.strip()
    prefix = ""
    lines = text.splitlines()
    if lines:
        head = lines[0].strip()
        m = _ARTICLE_HEAD_RE.match(head)
        if m:
            prefix = f"Статья {m.group(1)}. {m.group(2).strip()}"
        elif chunk.article:
            prefix = f"Статья {chunk.article}."
    elif chunk.article:
        prefix = f"Статья {chunk.article}."
    if prefix:
        return f"{prefix}\n{text}"
    return text
