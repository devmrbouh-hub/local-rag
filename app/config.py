"""Настройки из переменных окружения.

Здесь нет обращений к Ollama и Chroma: только пути, имена моделей и лимиты.
Пути к документам и индексу читаются при вызове, а не при импорте, чтобы pytest
мог подставить временный каталог без перезапуска интерпретатора.
"""

from __future__ import annotations

import os
from pathlib import Path

# Иначе Chroma на каждый запрос пишет ошибку телеметрии в консоль.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b-instruct-q4_K_M")

COLLECTION_NAME = os.environ.get("COLLECTION", "tk_rf")

CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

# CLI retrieval (query_cli): короткий список, чтобы глазами смотреть top-5.
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "5"))
# /ask и llm.answer_question: шире, потому что короткий вопрос часто не совпадает
# со словами статьи (пример: «минимальная продолжительность отпуска» и ст. 115).
ASK_DEFAULT_TOP_K = int(os.environ.get("ASK_DEFAULT_TOP_K", "10"))
# Сколько лучших фрагментов отдаём модели и в sources (поиск может быть шире top_k).
ASK_PROMPT_CHUNKS = int(os.environ.get("ASK_PROMPT_CHUNKS", "5"))
ASK_SOURCES_MAX = int(os.environ.get("ASK_SOURCES_MAX", "3"))

# Эмбеддинги пачками идут минутами; один чат 3B на CPU тоже не мгновенный.
OLLAMA_EMBED_TIMEOUT = float(os.environ.get("OLLAMA_EMBED_TIMEOUT", "120"))
OLLAMA_CHAT_TIMEOUT = float(os.environ.get("OLLAMA_CHAT_TIMEOUT", "180"))
# /health не должен зависать, если Ollama выключена.
OLLAMA_META_TIMEOUT = float(os.environ.get("OLLAMA_META_TIMEOUT", "3"))

# В промпт режем сильнее, чем в JSON: 10 чанков целиком раздувают контекст 3B.
PROMPT_CHUNK_CHARS = int(os.environ.get("PROMPT_CHUNK_CHARS", "700"))
# В JSON — одно предложение или начало абзаца (см. excerpt_for_source).
SOURCE_CHUNK_CHARS = int(os.environ.get("SOURCE_CHUNK_CHARS", "280"))

HYBRID_VECTOR_K = int(os.environ.get("HYBRID_VECTOR_K", "30"))
HYBRID_RRF_K = int(os.environ.get("HYBRID_RRF_K", "60"))


def hybrid_search_enabled() -> bool:
    """Читается при каждом search(): eval_rag --mode и pytest conftest могут менять env."""
    return os.environ.get("HYBRID_SEARCH", "true").lower() in ("1", "true", "yes")

SKIP_FILES = {"README.md", "control_questions.md"}


def _path_from_env(env_name: str, default: str) -> Path:
    raw = os.environ.get(env_name, default)
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def get_docs_dir() -> Path:
    """Каталог markdown-корпуса. По умолчанию docs/tk_rf."""
    return _path_from_env("DOCS_DIR", "docs/tk_rf")


def get_chroma_dir() -> Path:
    """Каталог persistent-индекса Chroma. По умолчанию data/chroma."""
    return _path_from_env("CHROMA_DIR", "data/chroma")


def get_bm25_dir() -> Path:
    """Pickle BM25. По умолчанию data/bm25."""
    return _path_from_env("BM25_DIR", "data/bm25")
