"""Нарезка markdown-глав ТК на чанки для эмбеддингов.

Граница чанка — заголовок «## Статья». Длинная статья режется по токенам
с перекрытием. Файлы из SKIP_FILES (оглавление и контрольные вопросы) не индексируются.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from app.config import CHUNK_OVERLAP, CHUNK_TOKENS, SKIP_FILES

# cl100k — грубая оценка длины, не токенизатор llama. Для лимита чанка этого достаточно.
_ENC = tiktoken.get_encoding("cl100k_base")

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
ARTICLE_BLOCK_RE = re.compile(
    r"^##\s+Статья\s+(\d+(?:\.\d+)?)\.\s*(.+?)$",
    re.MULTILINE,
)


@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_index: int
    chapter: str | None
    article: str | None


def token_len(text: str) -> int:
    """Число токенов cl100k."""
    return len(_ENC.encode(text))


def split_text_with_overlap(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Режет текст окнами. overlap нужен, чтобы норма на стыке окон не потерялась."""
    tokens = _ENC.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        parts.append(_ENC.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        # Без max(0, ...) при overlap > max_tokens цикл не сдвинется.
        start = max(0, end - overlap)
    return parts


def parse_frontmatter_chapter(content: str) -> str | None:
    """Номер главы из YAML-шапки файла, если она есть."""
    m = re.search(r"^chapter:\s*([^\n]+)", content, re.MULTILINE)
    return m.group(1).strip() if m else None


def chunk_markdown_file(path: Path) -> list[Chunk]:
    """Чанки одного файла. Если статей нет, режется весь текст после шапки."""
    raw = path.read_text(encoding="utf-8")
    chapter = parse_frontmatter_chapter(raw)
    body = FRONTMATTER_RE.sub("", raw, count=1).strip()

    chunks: list[Chunk] = []
    matches = list(ARTICLE_BLOCK_RE.finditer(body))
    if not matches:
        for i, part in enumerate(split_text_with_overlap(body, CHUNK_TOKENS, CHUNK_OVERLAP)):
            chunks.append(
                Chunk(
                    text=part,
                    source_file=path.name,
                    chunk_index=i,
                    chapter=chapter,
                    article=None,
                )
            )
        return chunks

    for idx, m in enumerate(matches):
        article_num = m.group(1)
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        block = body[start:end].strip()
        subparts = split_text_with_overlap(block, CHUNK_TOKENS, CHUNK_OVERLAP)
        for part in subparts:
            chunks.append(
                Chunk(
                    text=part,
                    source_file=path.name,
                    chunk_index=0,
                    chapter=chapter,
                    article=article_num,
                )
            )
    # Индекс сквозной по файлу: внутри статьи он сначала 0 у каждого куска.
    for i, ch in enumerate(chunks):
        ch.chunk_index = i
    return chunks


def load_chunks_from_dir(docs_dir: Path) -> list[Chunk]:
    """Все чанки каталога, файлы по имени. README и control_questions пропускаются."""
    all_chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name in SKIP_FILES:
            continue
        file_chunks = chunk_markdown_file(path)
        for i, ch in enumerate(file_chunks):
            ch.chunk_index = i
        all_chunks.extend(file_chunks)
    return all_chunks
