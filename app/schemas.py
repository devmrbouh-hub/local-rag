"""Модели запросов и ответов HTTP API. Логики RAG здесь нет."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str | None = None


class IngestResponse(BaseModel):
    files: int
    chunks: int
    seconds: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class SourceItem(BaseModel):
    file: str
    chunk: str
    score: float
    article: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class HealthResponse(BaseModel):
    llm_model: str
    embed_model: str
    quantization: str | None
    chunks_in_index: int
    index_ready: bool
    ollama_version: str | None
    llm_available: bool
    embed_available: bool
    ram_mb: int | None
    vram_mb: int | None
