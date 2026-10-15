# local-rag

Локальный RAG по Трудовому кодексу РФ: Ollama, Chroma, FastAPI. Ответ строится только из найденных фрагментов и возвращается вместе с источниками.

Репозиторий: [github.com/devmrbouh-hub/local-rag](https://github.com/devmrbouh-hub/local-rag). Корпус — текст ТК РФ из открытого источника (см. [docs/tk_rf/README.md](docs/tk_rf/README.md)). **Не юридическая консультация.**

Опционально: скопируйте [.env.example](.env.example) в `.env` и при необходимости поменяйте `OLLAMA_HOST` и флаги поиска.

## Как устроено

```
docs/tk_rf/  →  чанки  →  embed(chunk_search_text)  →  Chroma + BM25 pickle
вопрос  →  vector top-30 + BM25 top-30  →  RRF  →  top-k  →  промпт  →  llama3.2 3B
```

По умолчанию **гибридный поиск** (`HYBRID_SEARCH=true`): Chroma (cosine) и BM25 сливаются через RRF. В `sources.score` при гибриде это вес RRF (больше — лучше), не `1 − distance`. Только вектор: `HYBRID_SEARCH=false` или `eval_rag.py --mode vector`.

После обновления кода поиска нужен **полный re-ingest** (`python -m app.ingest_cli`): обогащённые эмбеддинги и `data/bm25/index.pkl` собираются вместе с Chroma.

HTTP: `POST /ingest`, `POST /ask`, `GET /health`. Промпт запрещает додумывать нормы. Если в фрагментах нет ответа, модель должна написать: «В предоставленном контексте нет информации для ответа».

## Требования

- Python 3.10+ (в CI — 3.11)
- [Ollama](https://ollama.com) с моделями `nomic-embed-text` и `llama3.2:3b-instruct-q4_K_M`
- Виртуальное окружение в корне проекта

Docker не используем.

## Запуск

Порта два, и это разные сервисы. **11434** — Ollama, там живут модели. **8000** — FastAPI, там `/ask` и `/health`. Команда `uvicorn ... --port 8000` поднимает API, а не Ollama.

Сначала Ollama. Десктопное приложение достаточно открыть (иконка в трее), либо в отдельном окне:

```powershell
ollama serve
```

Окно не закрывать. Хост и порт Ollama берутся из `OLLAMA_HOST`, по умолчанию `http://127.0.0.1:11434` (см. `.env.example`). Проверка, что сервер отвечает:

```powershell
curl.exe -s http://127.0.0.1:11434/api/tags
```

Модели скачиваются один раз:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b-instruct-q4_K_M
```

Ollama нужна уже на этапе индексации: эмбеддинги считаются через неё, без сервера `ingest_cli` упадёт с сообщением «Не удалось подключиться к Ollama».

Дальше установка:

```powershell
git clone https://github.com/devmrbouh-hub/local-rag.git
cd local-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```powershell
python -m app.ingest_cli
```

Индексация синхронная, на полном ТК около 5 минут. Повторный запуск пересобирает коллекцию целиком. Каталоги `data/chroma/` и `data/bm25/` в git не входят.

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Что поднялось и видна ли Ollama, показывает `GET /health`: поля `ollama_version`, `llm_available`, `embed_available`.

## Запрос

`POST /ask` по умолчанию берёт 10 фрагментов (`ASK_DEFAULT_TOP_K`). В CLI поиска без модели по-прежнему 5 (`DEFAULT_TOP_K`). Короткий вопрос часто не совпадает со словами статьи, поэтому у API запас шире. Для более короткого ответа передайте `"top_k": 3` или `5`.

### Поля ответа

| Поле | Зачем |
|------|--------|
| `answer` | Краткий пересказ по фрагментам (читать в первую очередь) |
| `sources` | До **3** лучших фрагментов (`ASK_SOURCES_MAX`), `chunk` — короткая выдержка (~1 предложение) |
| `answer` | Формат: `Статья N: одно предложение.` (задаётся промптом) |

В промпт модели попадают **5** лучших фрагментов после RRF (`ASK_PROMPT_CHUNKS`, по умолчанию 5), в JSON `sources` — до **3** (`ASK_SOURCES_MAX`). Поиск для `/ask` по умолчанию `top_k=10`.

### PowerShell (Windows)

Подняты должны быть оба процесса: Ollama на 11434 (раздел «Запуск») и API на 8000.

Сервер: `uvicorn app.main:app --host 127.0.0.1 --port 8000` (окно не закрывать).

Проверка здоровья:

```powershell
curl.exe -s http://127.0.0.1:8000/health
```

Вопрос с кириллицей — надёжнее сохранить JSON в файл (в PS 5.1 в консоли часто кракозябры, в файле UTF-8 нормальный):

```powershell
$body = [System.Text.Encoding]::UTF8.GetBytes(
  '{"question":"Какие гарантии при направлении в служебную командировку?","top_k":10}'
)
Invoke-WebRequest -Uri http://127.0.0.1:8000/ask -Method Post `
  -ContentType "application/json; charset=utf-8" -Body $body -UseBasicParsing `
  -OutFile .\ask.json
notepad .\ask.json
```

Тот же запрос из venv (вывод в консоль):

```powershell
.\.venv\Scripts\python.exe -c "import json,urllib.request; d=json.dumps({'question':'Какие гарантии при направлении в служебную командировку?','top_k':10},ensure_ascii=False).encode(); r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/ask',data=d,headers={'Content-Type':'application/json'}),timeout=180); print(r.read().decode('utf-8'))"
```

`curl` одной строкой (экранирование кавычек):

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"Какие гарантии при направлении в служебную командировку?\",\"top_k\":10}"
```

Пустой индекс — 404. Ollama недоступна — 503. Пустой вопрос или `top_k` вне 1…50 — 422. Повторный `/ingest`, пока первый ещё идёт, — 409. `POST /ingest` на полном корпусе длится несколько минут, таймаут клиента нужно увеличить.

Поиск без генерации:

```powershell
python -m app.query_cli "гарантии при служебной командировке" --top-k 5
```

Ответ модели **в терминале** (Ollama должна быть запущена, индекс собран):

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.ask_cli "Какие гарантии при направлении в служебную командировку?"
```

Другой вопрос или шире поиск: `python -m app.ask_cli "…" --top-k 10`

## Переменные поиска

| Переменная | По умолчанию | Смысл |
|------------|--------------|--------|
| `HYBRID_SEARCH` | `true` | RRF vector + BM25 |
| `HYBRID_VECTOR_K` | `30` | Глубина списков кандидатов |
| `HYBRID_RRF_K` | `60` | Константа в формуле RRF |
| `BM25_DIR` | `data/bm25` | Pickle BM25 |

Тюнинг нарезки без смены кода: `CHUNK_TOKENS=512`, `CHUNK_OVERLAP=80` и снова ingest (сравнить `eval_rag --compare-k`).

## Замеры (9 октября 2026, эта машина)

| Метрика | Было (только vector) | Стало (hybrid + re-ingest) |
|---------|----------------------|----------------------------|
| Индекс | 815 чанков, ingest ~296 с | те же 815 чанков, ingest ~287 с |
| Поиск top-5 | 4/18 | **14/18** |
| Поиск top-10 | 7/18 | **15/18** |
| Vector после enrich (без BM25) | — | 7/18 top-10 |

Остальное без пересчёта в этом срезе: LLM ~4,8 ток/с (3B), отказ off-topic 3/3, нарезка 800/100.

Контрольные вопросы: [docs/tk_rf/control_questions.md](docs/tk_rf/control_questions.md). Эталон: [docs/tk_rf/control_expectations.json](docs/tk_rf/control_expectations.json).

```powershell
python scripts/eval_rag.py --retrieval-only --compare-k
python scripts/eval_rag.py --retrieval-only --compare-k --mode vector
python scripts/eval_rag.py --ask
```

Промахи hybrid top-10 (3/18): вопросы 4, 5, 18 (ст. 312.1, 312.9, 81).

## Было / стало

Листать PDF или главу вручную — минуты на один вопрос, без ссылки на фрагмент. Чистый vector давал 7/18 в top-10; гибрид после re-ingest — **15/18**. Полный ответ с цитатой — около 25 с на вопрос (3B, локально). Off-topic: три отказа из трёх.

## Примеры

Командировка. Вопрос: «Какие гарантии при направлении в служебную командировку?» Первая цитата — ст. 167.

Ознакомление с приказом о взыскании. Вопрос 10. Ответ опирается на ст. 193, она же первая в `sources`.

Срок выплаты зарплаты. Вопрос 17. В ответе и в первой цитате — ст. 136.

Отпуск. Вопрос про минимальную продолжительность. После гибрида ст. 115 попадает в top-10; ответ модели перепроверять через `ask_cli`.

Вопросы не из кодекса (штраф по УК, НДС на отпускные, виза в Японию): все три ответа — «В предоставленном контексте нет информации для ответа».

## Тесты

Ollama в тестах и в GitHub Actions не поднимается.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Статус

- [x] Корпус ТК, индекс, CLI-поиск
- [x] FastAPI `/ingest`, `/ask`, `/health`
- [x] pytest и workflow CI
- [ ] Скринкаст и скрин `/health` (вручную)
- [x] Публикация на [GitHub](https://github.com/devmrbouh-hub/local-rag) (CI: pytest)
