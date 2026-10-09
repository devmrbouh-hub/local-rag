"""Подмены для тестов. Не импортируйте conftest отсюда: его код выполнится второй раз."""

from __future__ import annotations


def fake_embed(text: str, model: str | None = None) -> list[float]:
    """Два ортогональных вектора, чтобы cosine отличил отпуск от остального текста."""
    lowered = text.lower()
    if "отпуск" in lowered or "115" in text:
        return [1.0, 0.0, 0.0, 0.0]
    return [0.0, 1.0, 0.0, 0.0]
