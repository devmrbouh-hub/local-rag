"""HTTP-оболочка: /ingest, /ask, /health.

Промпты и обращение к Chroma живут в llm/ingest/retrieve.
Здесь только разбор пути, блокировка повторного ingest и перевод ошибок в коды HTTP.
"""

from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import EMBED_MODEL, LLM_MODEL, PROJECT_ROOT
from app.ingest import run_ingest
from app.llm import answer_question
from app.ollama_client import OllamaError, get_version, list_models
from app.retrieve import IndexNotReadyError, get_index_stats
from app.schemas import AskRequest, AskResponse, HealthResponse, IngestRequest, IngestResponse

app = FastAPI(title="local-rag", version="0.2.0")

# Второй POST /ingest на том же процессе не должен сносить коллекцию посередине первого.
_ingest_lock = threading.Lock()

_QUANT_RE = re.compile(r"(q\d+(?:_[0-9A-Za-z]+)*)")


def get_ingest_runner() -> Callable[..., dict]:
    """Фабрика для Depends: в тестах подменяется на заглушку."""
    return run_ingest


def get_answer_service() -> Callable[..., tuple[str, list[dict]]]:
    """Фабрика для Depends: в тестах подменяется на заглушку."""
    return answer_question


def resolve_docs_path(raw: str | None) -> Path:
    """Путь к корпусу только внутри корня репозитория.

    Иначе запрос path=../../secrets прочитает чужой каталог относительно PROJECT_ROOT.
    """
    base = PROJECT_ROOT.resolve()
    if raw is None or not raw.strip():
        from app.config import get_docs_dir

        candidate = get_docs_dir().resolve()
    else:
        given = Path(raw)
        candidate = given.resolve() if given.is_absolute() else (base / given).resolve()
    if not candidate.is_relative_to(base):
        raise HTTPException(
            status_code=422,
            detail="Путь должен находиться внутри каталога проекта",
        )
    return candidate


def _quantization(model_name: str) -> str | None:
    match = _QUANT_RE.search(model_name)
    return match.group(1) if match else None


def _model_present(names: list[str], wanted: str) -> bool:
    """В /api/tags имя часто с суффиксом :latest, в конфиге его нет."""
    for name in names:
        if name == wanted or name.startswith(wanted + ":"):
            return True
    return False


def _ram_mb() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().total / (1024 * 1024))


def _vram_mb() -> int | None:
    # На машине без NVIDIA nvidia-smi нет — поле просто пустое.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        line = out.strip().splitlines()[0]
        return int(float(line.strip()))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def collect_health() -> HealthResponse:
    """Снимок окружения без вызова LLM и без эмбеддинга вопроса."""
    stats = get_index_stats()
    names = list_models()
    return HealthResponse(
        llm_model=LLM_MODEL,
        embed_model=EMBED_MODEL,
        quantization=_quantization(LLM_MODEL),
        chunks_in_index=stats["chunks"],
        index_ready=stats["ready"],
        ollama_version=get_version(),
        llm_available=_model_present(names, LLM_MODEL),
        embed_available=_model_present(names, EMBED_MODEL),
        ram_mb=_ram_mb(),
        vram_mb=_vram_mb(),
    )


@app.exception_handler(OllamaError)
async def ollama_error_handler(_request: Request, exc: OllamaError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(IndexNotReadyError)
async def index_error_handler(_request: Request, exc: IndexNotReadyError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def not_found_handler(_request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    body: IngestRequest,
    runner: Callable[..., dict] = Depends(get_ingest_runner),
) -> IngestResponse:
    """Полная пересборка индекса. Запрос синхронный, на всём ТК — несколько минут."""
    path = resolve_docs_path(body.path)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Папка документов не найдена: {path}")
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Индексация уже выполняется")
    try:
        stats = runner(path)
    finally:
        _ingest_lock.release()
    # chroma_dir в CLI полезен, в контракте API его нет.
    return IngestResponse(
        files=stats["files"],
        chunks=stats["chunks"],
        seconds=stats["seconds"],
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    answer: Callable[..., tuple[str, list[dict]]] = Depends(get_answer_service),
) -> AskResponse:
    text, sources = answer(body.question, body.top_k)
    return AskResponse(answer=text, sources=sources)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return collect_health()
