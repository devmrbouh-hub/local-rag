"""Командная строка для ingest. HTTP здесь нет: только вызов run_ingest."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import PROJECT_ROOT, get_docs_dir
from app.ingest import run_ingest


def main() -> int:
    parser = argparse.ArgumentParser(description="Индексация markdown ТК в Chroma")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Папка с .md (по умолчанию docs/tk_rf)",
    )
    args = parser.parse_args()
    if args.path is None:
        path = get_docs_dir()
    else:
        # Относительный путь — от корня репозитория, не от текущей папки shell.
        path = args.path if args.path.is_absolute() else PROJECT_ROOT / args.path

    print(f"Индексация из {path} ...")
    stats = run_ingest(path)
    print(
        f"Готово: файлов={stats['files']} чанков={stats['chunks']} "
        f"время={stats['seconds']} с каталог={stats['chroma_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
