# Корпус: Трудовой кодекс РФ

Демо-документы для локального RAG. **Не юридическая консультация.**

| Поле | Значение |
|------|----------|
| Нормативный акт | ТК РФ, 30.12.2001 № 197-ФЗ |
| Редакция в источнике | от 25.05.2026 (по PDF КонсультантПлюс) |
| Источник текста | https://www.debet.ru/site/assets/files/1020/tk.pdf |
| Подготовка | `scripts/prepare_tk_docs.py` (нужен `pip install pymupdf`) |
| Файлы | одна глава = один `.md` (сейчас 70 глав) |

Пересобрать из PDF в `data/source/tk_rf.pdf`:

```powershell
pip install pymupdf
python scripts/prepare_tk_docs.py
```
