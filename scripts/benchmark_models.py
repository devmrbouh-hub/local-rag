"""Замер скорости генерации через POST /api/generate.

Это не путь RAG: /ask ходит в /api/chat. Скрипт нужен только для цифры tok/s в README.
Ollama должна быть запущена, модель — скачана.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Консоль Windows по умолчанию в cp1251, а в выводе есть «≈». Без этого print падает
# с UnicodeEncodeError ещё до того, как замер успеет напечататься.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import LLM_MODEL, OLLAMA_HOST  # noqa: E402

PROMPT = "Кратко: что регулирует Трудовой кодекс РФ? Ответ в 3 предложениях."


def bench(model: str) -> None:
    """Один нес streaming-запрос. tok/s берётся из eval_count Ollama, если он есть."""
    url = f"{OLLAMA_HOST}/api/generate"
    body = json.dumps(
        {"model": model, "prompt": PROMPT, "stream": False},
        ensure_ascii=False,
    ).encode("utf-8")
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    text = data.get("response", "")
    eval_count = data.get("eval_count") or len(text.split())
    tps = eval_count / elapsed if elapsed > 0 else 0
    print(f"Модель: {model}")
    print(f"  время={elapsed:.1f} с токенов≈{eval_count} ток/с≈{tps:.2f}")
    print(f"  начало ответа: {text[:200].replace(chr(10), ' ')}...")


def main() -> int:
    bench(LLM_MODEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
