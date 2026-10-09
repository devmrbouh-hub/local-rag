"""Сборка сообщений для чата Ollama. Сеть и Chroma здесь не вызываются."""

from __future__ import annotations

import re

from app.config import PROMPT_CHUNK_CHARS

# Модель обязана опираться только на фрагменты в user-сообщении и не дописывать нормы.
# Если ответа в фрагментах нет, нужна ровно эта фраза: по ней скрипт оценки считает «не знаю».
NO_CONTEXT_ANSWER = "В предоставленном контексте нет информации для ответа"

# Модель 3B раздувает ответ, если дать много фрагментов и свободный формат.
# Один абзац, одна статья — так проще читать в демо и в JSON.
SYSTEM_PROMPT = f"""Ты отвечаешь на вопросы по Трудовому кодексу РФ только по фрагментам в сообщении пользователя.

Правила:
- Не используй память модели. Не упоминай статьи, которых нет во фрагментах.
- Если ответа нет во фрагментах, напиши только: {NO_CONTEXT_ANSWER}
- Формат ответа (строго, без списков и без markdown):
  Статья <номер>: <одно короткое предложение с сутью нормы>.
- Бери номер статьи из метки article у самого релевантного фрагмента [1].
- Если ответ есть только во фрагменте [1], не ссылайся на другие статьи из соседних фрагментов.
- При отказе не выдумывай номер статьи — только фраза про отсутствие информации.
- Не больше двух предложений. Не пересказывай заголовок статьи целиком.

Пример хорошего ответа:
Статья 167: при командировке работнику сохраняют место работы, средний заработок и возмещают связанные расходы.
"""


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _strip_article_heading(text: str) -> str:
    """Убираем «## Статья N. …», чтобы модель не копировала заголовок в ответ."""
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("## Статья"):
        return "\n".join(lines[1:]).strip() or text
    return text


def build_rag_messages(question: str, hits: list[dict]) -> list[dict]:
    """Пара system/user. В промпт попадает мало лучших чанков — см. ASK_PROMPT_CHUNKS в llm."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        article = hit.get("article") or "—"
        source = hit.get("source_file") or ""
        raw = hit.get("text") or ""
        body = _clip(_strip_article_heading(raw), PROMPT_CHUNK_CHARS)
        blocks.append(f"[{i}] article={article} file={source}\n{body}")
    context = "\n\n".join(blocks) if blocks else "(фрагментов нет)"
    user = f"Фрагменты:\n{context}\n\nВопрос: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def excerpt_for_source(text: str, max_chars: int) -> str:
    """Короткая цитата для JSON: одно предложение или начало абзаца без заголовка."""
    body = _strip_article_heading(text).strip()
    body = re.sub(r"\s+", " ", body)
    if not body:
        return ""
    match = re.search(r".+?[.!?](?:\s|$)", body)
    if match and len(match.group(0)) <= max_chars:
        return match.group(0).strip()
    return _clip(body, max_chars)
