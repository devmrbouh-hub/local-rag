"""Контракт HTTP. Индексация и LLM в успешном /ask подменены через Depends."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app, get_answer_service, get_ingest_runner


def _client() -> TestClient:
    return TestClient(app)


def test_health_without_index() -> None:
    response = _client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["llm_model"]
    assert body["embed_model"]
    if "q4_K_M" in body["llm_model"]:
        assert body["quantization"] == "q4_K_M"
    assert body["chunks_in_index"] == 0
    assert body["index_ready"] is False
    assert body["ollama_version"] is None or isinstance(body["ollama_version"], str)


def test_ask_without_index_is_404() -> None:
    response = _client().post("/ask", json={"question": "сколько дней отпуска"})
    assert response.status_code == 404


def test_ask_validation() -> None:
    client = _client()
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "норма", "top_k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "норма", "top_k": 51}).status_code == 422


def test_ask_override() -> None:
    def fake_answer(question: str, top_k: int | None = None) -> tuple[str, list[dict]]:
        return (
            "28 календарных дней",
            [{"file": "glava_19.md", "chunk": "28 календарных дней", "score": 0.9, "article": "115"}],
        )

    app.dependency_overrides[get_answer_service] = lambda: fake_answer
    try:
        response = _client().post("/ask", json={"question": "отпуск", "top_k": 3})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "28 календарных дней"
    assert body["sources"][0]["file"] == "glava_19.md"
    assert body["sources"][0]["score"] == 0.9


def test_ingest_override() -> None:
    def fake_ingest(path) -> dict:
        return {"files": 1, "chunks": 4, "seconds": 0.2, "chroma_dir": "ignored"}

    app.dependency_overrides[get_ingest_runner] = lambda: fake_ingest
    try:
        # docs/tk_rf лежит внутри репозитория, заглушка не читает файлы.
        response = _client().post("/ingest", json={"path": "docs/tk_rf"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == {"files": 1, "chunks": 4, "seconds": 0.2}


def test_ingest_missing_dir() -> None:
    response = _client().post("/ingest", json={"path": "docs/нет-такой-папки"})
    assert response.status_code == 404


def test_ingest_path_escape() -> None:
    response = _client().post("/ingest", json={"path": ".."})
    assert response.status_code == 422


def test_ingest_already_running() -> None:
    from app.main import _ingest_lock

    assert _ingest_lock.acquire(blocking=False)
    try:
        response = _client().post("/ingest", json={"path": "docs/tk_rf"})
        assert response.status_code == 409
    finally:
        _ingest_lock.release()
