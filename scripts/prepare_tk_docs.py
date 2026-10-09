"""Сборка корпуса: PDF или текст ТК → один markdown на главу.

Это подготовка данных, не runtime RAG. pymupdf нужен только если есть PDF.
Колонтитулы КонсультантПлюс выкидываются, рваные строки PDF склеиваются обратно.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "tk_rf"
SOURCE_DIR = ROOT / "data" / "source"
PDF_PATH = SOURCE_DIR / "tk_rf.pdf"
FALLBACK_TEXT = SOURCE_DIR / "tk_rf_extracted.txt"

HEADER = """---
law: "ТК РФ 197-ФЗ"
source_url: "https://www.debet.ru/site/assets/files/1020/tk.pdf"
source_edition: "ред. от 25.05.2026 (по PDF КонсультантПлюс)"
prepared: "{date}"
chapter: {chapter_num}
chapter_title: "{chapter_title}"
---

# Трудовой кодекс РФ — Глава {chapter_num}. {chapter_title}

"""

CHAPTER_MARKER_RE = re.compile(
    r"Глава\s+(\d+(?:\.\d+)?)\.\s*",
    re.IGNORECASE,
)

ARTICLE_INLINE_RE = re.compile(
    r"\s+(Статья\s+\d+(?:\.\d+)?\.)",
    re.IGNORECASE,
)

NOISE_LINE_RE = re.compile(
    r"(?i)(консультантплюс|www\.consultant\.ru|надежная правовая поддержка|"
    r"дата сохранения:|страница\s+\d+\s+из\s+\d+|"
    r'"трудовой кодекс российской федерации"\s+от\s+30\.12\.2001|'
    r"^197-фз$)",
)

SENTENCE_END_RE = re.compile(r"[.;:!?»\)]$")
CONTINUE_START_RE = re.compile(r"^[а-яё(«0-9]")
STRUCTURAL_RE = re.compile(r"^(##\s+)?(Глава\s+\d|Статья\s+\d)", re.IGNORECASE)


def slugify(title: str, max_len: int = 60) -> str:
    title = title.strip().lower()
    out: list[str] = []
    for ch in title:
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("-")
    slug = "".join(out)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return (slug[:max_len] or "glava").replace("--", "-")


def extract_text_from_pdf(path: Path) -> str:
    import pymupdf

    doc = pymupdf.open(path)
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return "\n".join(parts)


def normalize_text(text: str) -> str:
    """Вставляет разрывы перед «Статья», если PDF склеил заголовок с предыдущей строкой."""
    text = ARTICLE_INLINE_RE.sub(r"\n\n\1", text)
    text = re.sub(
        r"(Глава\s+\d+(?:\.\d+)?\.\s+[^\n]+?)(?=\n\nСтатья\s)",
        lambda m: m.group(1).strip() + "\n",
        text,
        flags=re.IGNORECASE,
    )
    return text


def load_raw_text() -> str:
    if PDF_PATH.is_file():
        return normalize_text(extract_text_from_pdf(PDF_PATH))
    if FALLBACK_TEXT.is_file():
        return normalize_text(FALLBACK_TEXT.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"Положите PDF в {PDF_PATH} или текст в {FALLBACK_TEXT}"
    )


def reflow_legal_text(lines: list[str]) -> list[str]:
    """Склеивает переносы PDF в абзацы. Заголовки статей и пустые строки не трогает.

    Дефис в конце строки считаем переносом слова, не тире. Одиночные слова
    (обрывки вёрстки) присоединяются к соседу, иначе в чанках остаётся мусор.
    """
    out: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf:
            out.append(buf)
            buf = ""

    for line in lines:
        s = line.strip()
        if not s:
            flush()
            if out and out[-1] != "":
                out.append("")
            continue

        if STRUCTURAL_RE.match(s) or re.match(r"^\(в ред\.", s, re.IGNORECASE):
            flush()
            out.append(s)
            continue

        if not buf:
            buf = s
            continue

        # Перенос слова в PDF: «рабо-\nтодатель», не пунктуационное тире.
        if buf.endswith("-") and not buf.endswith("--"):
            buf = buf[:-1] + s
            continue

        buf_words = len(buf.split())
        s_words = len(s.split())
        if not SENTENCE_END_RE.search(buf.rstrip()) and CONTINUE_START_RE.match(s):
            buf = f"{buf} {s}"
            continue
        if buf_words == 1 or s_words == 1:
            buf = f"{buf} {s}"
            continue
        if not SENTENCE_END_RE.search(buf.rstrip()):
            buf = f"{buf} {s}"
            continue

        flush()
        buf = s

    flush()
    return out


def count_single_word_lines(lines: list[str]) -> int:
    return sum(1 for ln in lines if ln.strip() and len(ln.split()) == 1 and not STRUCTURAL_RE.match(ln.strip()))


def chapter_title_at(text: str, marker_end: int) -> str:
    rest = text[marker_end:]
    line = rest.split("\n", 1)[0].strip()
    line = re.sub(r"\s+", " ", line)
    # Плоский PDF: «ОТПУСКА Статья 114. ...» — название главы кончается перед «Статья».
    parts = re.split(r"\s+Статья\s+\d", line, maxsplit=1, flags=re.IGNORECASE)
    title = parts[0].strip()
    if not title:
        title = "Без названия"
    return title


def format_body(chunk: str, num: str, title: str) -> str:
    lines: list[str] = []
    lines.append(f"Глава {num}. {title}")
    lines.append("")

    skip_header = True
    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        # Колонтитулы КонсультантПлюс повторяются на каждой странице и портят эмбеддинги.
        if NOISE_LINE_RE.search(stripped):
            continue
        if stripped.startswith("### Документ предоставлен"):
            continue
        if skip_header:
            if re.match(rf"Глава\s+{re.escape(num)}\.", stripped, re.IGNORECASE):
                continue
            if stripped.upper() == title.upper():
                continue
            skip_header = False
        if re.match(r"Статья\s+\d", stripped, re.IGNORECASE):
            lines.append("")
            lines.append(f"## {stripped}")
            continue
        lines.append(stripped)

    lines = reflow_legal_text(lines)
    body = "\n".join(lines)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def split_chapters(text: str) -> list[tuple[str, str, str]]:
    matches = list(CHAPTER_MARKER_RE.finditer(text))
    if not matches:
        raise ValueError("Главы не найдены — проверьте кодировку и исходный файл")

    chapters: list[tuple[str, str, str]] = []

    for i, m in enumerate(matches):
        num = m.group(1)
        title = chapter_title_at(text, m.end())
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end]
        chapters.append((num, title, format_body(chunk, num, title)))

    return chapters


def remove_old_chapter_files() -> None:
    for path in DOCS_DIR.glob("glava_*.md"):
        path.unlink()


def write_chapters(chapters: list[tuple[str, str, str]], date_str: str) -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    remove_old_chapter_files()
    written = 0
    for num, title, body in chapters:
        num_slug = num.replace(".", "_")
        fname = f"glava_{num_slug}_{slugify(title)}.md"
        path = DOCS_DIR / fname
        safe_title = title.replace('"', "'")
        header = HEADER.format(
            date=date_str,
            chapter_num=num,
            chapter_title=safe_title,
        )
        path.write_text(header + body + "\n", encoding="utf-8")
        written += 1
    return written


def main() -> int:
    from datetime import date

    raw = load_raw_text()
    chapters = split_chapters(raw)
    count = write_chapters(chapters, date.today().isoformat())
    sample = (DOCS_DIR / "glava_19_отпуска.md").read_text(encoding="utf-8")
    body_lines = [ln for ln in sample.splitlines() if ln.strip() and not ln.startswith("---") and not ln.startswith("#") and not ln.startswith("law:") and not ln.startswith("chapter")]
    single = count_single_word_lines(body_lines)
    print(f"Записано файлов глав: {count}, каталог {DOCS_DIR}")
    print(f"glava_19, строк из одного слова (проба склейки): {single}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
