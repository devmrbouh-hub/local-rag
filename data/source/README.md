# Исходник текста ТК РФ

Файлы `tk_rf_extracted.txt` и `tk_rf.pdf` **не в git** (см. `.gitignore`). Локально PDF уже можно положить в `tk_rf.pdf`.

Пересборка глав (приоритет — PDF через pymupdf):

1. PDF: https://www.debet.ru/site/assets/files/1020/tk.pdf → `tk_rf.pdf`
2. `pip install pymupdf`
3. `python scripts/prepare_tk_docs.py`

Запасной вариант: только `tk_rf_extracted.txt` (хуже разметка заголовков).
