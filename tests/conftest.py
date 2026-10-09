"""Временный каталог индекса задаётся до импорта app, иначе тесты попадут в data/chroma."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="local-rag-pytest-"))
os.environ["CHROMA_DIR"] = str(_TMP / "chroma")
os.environ["BM25_DIR"] = str(_TMP / "bm25")
os.environ["HYBRID_SEARCH"] = "false"

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """Два коротких файла: статья про отпуск и статья без этого слова."""
    dest = tmp_path / "docs"
    dest.mkdir()
    for name in ("sample_doc.md", "other.md"):
        (dest / name).write_text((FIXTURES / name).read_text(encoding="utf-8"), encoding="utf-8")
    return dest
