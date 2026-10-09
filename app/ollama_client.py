"""HTTP-клиент Ollama: эмбеддинги, чат и короткие метаданные.

Модуль не знает про чанки, промпт RAG и Chroma. Ошибки сети и HTTP
поднимаются как OllamaError. Методы health при недоступности сервера
возвращают пустой результат и не роняют /health.
"""

from __future__ import annotations

import httpx

from app.config import (
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_CHAT_TIMEOUT,
    OLLAMA_EMBED_TIMEOUT,
    OLLAMA_HOST,
    OLLAMA_META_TIMEOUT,
)


class OllamaError(RuntimeError):
    """Ollama недоступна или вернула ошибку. HTTP-слой переводит это в 503."""


def _request(method: str, path: str, *, timeout: float, json_body: dict | None = None) -> httpx.Response:
    url = f"{OLLAMA_HOST}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, json=json_body)
            resp.raise_for_status()
            return resp
    except httpx.ConnectError:
        raise OllamaError(
            f"Не удалось подключиться к Ollama ({OLLAMA_HOST}). Запустите Ollama."
        ) from None
    except httpx.HTTPStatusError as exc:
        raise OllamaError(f"Ошибка Ollama: {exc.response.text}") from exc


def embed(text: str, model: str | None = None) -> list[float]:
    """Вектор текста через POST /api/embeddings."""
    model = model or EMBED_MODEL
    resp = _request(
        "POST",
        "/api/embeddings",
        timeout=OLLAMA_EMBED_TIMEOUT,
        json_body={"model": model, "prompt": text},
    )
    data = resp.json()
    vector = data.get("embedding")
    if not vector:
        raise OllamaError(f"Пустой embedding в ответе Ollama: {data}")
    return vector


def embed_batch(texts: list[str], model: str | None = None) -> list[list[float]]:
    """По одному запросу на текст: у /api/embeddings нет пакетного режима."""
    return [embed(t, model=model) for t in texts]


def chat(messages: list[dict], model: str | None = None) -> str:
    """Ответ модели через POST /api/chat (не /api/generate): нужен system + user."""
    model = model or LLM_MODEL
    resp = _request(
        "POST",
        "/api/chat",
        timeout=OLLAMA_CHAT_TIMEOUT,
        json_body={"model": model, "messages": messages, "stream": False},
    )
    data = resp.json()
    content = (data.get("message") or {}).get("content")
    if not content:
        raise OllamaError(f"Пустой ответ chat API: {data}")
    return str(content)


def get_version() -> str | None:
    """Версия Ollama или None, если сервер не отвечает."""
    try:
        resp = _request("GET", "/api/version", timeout=OLLAMA_META_TIMEOUT)
    except OllamaError:
        return None
    version = resp.json().get("version")
    return str(version) if version else None


def list_models() -> list[str]:
    """Имена моделей из GET /api/tags. Пустой список, если Ollama недоступна."""
    try:
        resp = _request("GET", "/api/tags", timeout=OLLAMA_META_TIMEOUT)
    except OllamaError:
        return []
    names: list[str] = []
    for item in resp.json().get("models") or []:
        name = item.get("name")
        if name:
            names.append(str(name))
    return names
