# LLM Assistant Backend

Бэкенд-сервис для многофункционального LLM-ассистента с поддержкой RAG, SQL-запросов и интеллектуальной маршрутизации
запросов к различным инструментам.

## Ключевые возможности

• **Интеллектуальная маршрутизация** — автоматический выбор инструмента (RAG/SQL/Web) на основе запроса  
• **RAG-система** — векторный поиск по документации Transformers с поддержкой чанкирования и эмбеддингов  
• **SQL-запросы** — преобразование естественного языка в SQL с валидацией безопасности  
• **Мультипровайдер LLM** — поддержка OpenAI, OpenRouter, Ollama, VLLM, Transformers  
• **Гибридная классификация** — комбинация правил и LLM для точной маршрутизации  
• **Docker-оркестрация** — полный стек с PostgreSQL, Qdrant и API  
• **Конфигурируемость** — YAML-конфиги для всех компонентов  
• **Статус**: Production-ready MVP

## Архитектура решения

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Клиент        │    │   FastAPI        │    │   Agent Router  │
│   (Web/Mobile)  │───▶│   API Server     │───▶│   (Rules+LLM)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │  Tool Selection │
                                              │  (docs/sql/web) │
                                              └─────────────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        ▼                               ▼                               ▼
                ┌──────────────┐              ┌──────────────┐              ┌──────────────┐
                │   RAG Tool   │              │   SQL Tool   │              │   Web Tool   │
                │              │              │              │              │   (TODO)     │
                │ ┌──────────┐ │              │ ┌──────────┐ │              │              │
                │ │ Qdrant   │ │              │ │PostgreSQL│ │              │              │
                │ │ Vector   │ │              │ │ Database │ │              │              │
                │ │ Store    │ │              │ │          │ │              │              │
                │ └──────────┘ │              │ └──────────┘ │              │              │
                └──────────────┘              └──────────────┘              └──────────────┘
```

**Потоки данных:**

1. **Запрос** → Agent Router → Классификация (Rules/LLM/Hybrid) → Выбор инструмента
2. **RAG**: Query → Qdrant → Retrieval → LLM Generation → Answer
3. **SQL**: Query → Schema Analysis → NL2SQL → Validation → Execution → Answer
4. **Сведение**: Tool Results → LLM Synthesis → Final Answer

**Хранилища:**

- **Qdrant** (6333) — векторные эмбеддинги документации
- **PostgreSQL** (5432) — структурированные данные команды/модули
- **Локальные файлы** — чанки, конфиги, кэш моделей

## Выбор технологий и обоснование

**FastAPI** — современный async-фреймворк с автоматической документацией OpenAPI  
**Qdrant** — векторная БД с высокой производительностью и простой интеграцией  
**PostgreSQL** — надежная реляционная БД для структурированных данных  
**Sentence Transformers** — локальные эмбеддинги без зависимости от внешних API  
**Pydantic** — валидация данных и типизация для надежности  
**Docker Compose** — оркестрация всех сервисов в одном стеке  
**OpenRouter** — единый API для доступа к различным LLM-провайдерам

## Установка и запуск (локально)

### Требования

- Python 3.11+ (тестировано на 3.11-3.13)
- Docker и Docker Compose

### Установка зависимостей

```bash
# Рекомендуется использовать uv для быстрой установки
pip install uv

# в корне проекта
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

uv sync

# Или через pip
pip install -e .
```

### Переменные окружения

Создайте файл `.env` в корне проекта и заполните по примеру `.env.template`.

### Команды запуска

```bash
# Запуск API сервера
uvicorn apps.api.main:app --reload --port 8000

# Запуск полного RAG пайплайна
python -m tools.rag.sources.pt_portal.cli --config configs/pt_loader.yaml
python -m tools.rag.chunking.cli --config configs/chunking.yaml
python -m tools.rag.embeddings.cli --config configs/embeddings.yaml embed
```

## Запуск в Docker / Docker Compose

### Основные команды

```bash
# Поднять базы
docker compose -f docker/docker-compose.yml up -d --build qdrant postgres

# Заполнить Qdrant
cd docker
docker compose build pipeline-full                                        
docker compose up pipeline-full 

# Запустить API
docker compose -f docker/docker-compose.yml up -d --build api
docker compose -f docker/docker-compose.yml logs -f api

# Просмотр логов
make logs
```

### Сервисы и порты

- **API**: http://localhost:8000 (документация: `/docs`)
- **Qdrant**: http://localhost:6333 (векторная БД)
- **PostgreSQL**: localhost:5432 (реляционная БД)

### Обновление контейнеров

```bash
# Пересборка и перезапуск
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## API — примеры запросов и ответов

### Основной эндпоинт ассистента

**POST** `/assistant` — Интеллектуальная маршрутизация запросов

```bash
curl -X POST "http://localhost:8000/assistant" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Transformers?"
      }' | jq '{route, tool_results, final_answer}'
```

### RAG-запросы

**POST** `/rag/qa` — Прямое обращение к RAG

```bash
curl -sS -X POST http://localhost:8000/rag/qa \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Transformers?"}' | jq
```

### SQL-запросы

**POST** `/sql/qa` — Преобразование естественного языка в SQL

```bash
curl -sS -X POST http://localhost:8000/sql/qa \
  -H "Content-Type: application/json" \
  -d '{"question":"Сколько человек в команде?"}' | jq
```

### Примеры ошибок

**400 Bad Request** — Ошибка валидации SQL

```json
{
  "detail": "SQL validation failed: Only SELECT queries are allowed"
}
```

**500 Internal Server Error** — Ошибка LLM провайдера

```json
{
  "detail": "OpenRouter API error: Rate limit exceeded"
}
```

## Оценка качества и метрики

### Текущие результаты

**TODO:** Добавить результаты оценки из `eval/` директории

### План оценки качества

**RAG метрики:**

- **Faithfulness** — соответствие ответа извлеченному контексту
- **Context Relevance** — релевантность найденных документов
- **Answer Correctness** — корректность финального ответа

**SQL метрики:**

- **Execution Success Rate** — процент успешно выполненных запросов
- **Query Accuracy** — соответствие SQL запросу пользователя
- **Safety Score** — отсутствие опасных операций

**Маршрутизация:**

- **Route Accuracy** — правильность выбора инструмента
- **Confidence Calibration** — соответствие уверенности качеству

### Команды для запуска оценки

```bash
# TODO: Добавить команды оценки после реализации
python -m eval.rag_metrics --config configs/eval.yaml
python -m eval.sql_metrics --dataset data/eval/sql_queries.jsonl
python -m eval.routing_metrics --config configs/agent.yaml
```

## Структура репозитория

```
llm-assistant-backend/
├── agents/                 # Агент-роутер и классификация
│   ├── classifier.py      # Правила и LLM-классификатор
│   ├── router.py          # Основная логика маршрутизации
│   └── schemas.py         # Pydantic модели
├── apps/api/              # FastAPI приложение
│   ├── main.py            # Точка входа API
│   └── routes/            # API эндпоинты
├── configs/               # YAML конфигурации
│   ├── agent.yaml         # Настройки агента
│   ├── llm.yaml           # LLM провайдеры
│   └── embeddings.yaml    # Настройки эмбеддингов
├── core/                  # Базовые компоненты
│   ├── settings.py        # Управление настройками
│   └── llm_provider.py    # LLM провайдеры
├── data/                  # Данные и индексы
│   ├── chunks/           # Обработанные чанки
│   └── raw_docs/         # Исходные документы
├── docker/               # Docker конфигурации
│   ├── docker-compose.yml # Оркестрация сервисов
│   └── Dockerfile.api    # Образ API
├── tools/                # Инструменты обработки
│   ├── rag/              # RAG пайплайн
│   └── sql/              # SQL инструменты
└── scripts/              # Скрипты запуска
```

## Профили и конфигурация

### Секреты

- **API ключи** — через переменные окружения (`.env`)
- **БД пароли** — в `docker-compose.yml` или `.env`
- **Конфиги** — в `configs/*.yaml` (без секретов)

## Observability

### Метрики

**TODO:** Добавить Prometheus метрики

- Время ответа API
- Количество запросов по инструментам
- Уверенность классификации
- Ошибки LLM провайдеров

### Логи

```bash
# Просмотр логов API
docker-compose logs -f api

# Логи всех сервисов
make logs

# Фильтрация по уровню
docker-compose logs api | grep ERROR
```

## Roadmap / Ограничения

### Планируемые улучшения

- [ ] **Web-поиск** — интеграция с поисковыми API (Google, Bing)
- [ ] **Кэширование** — Redis для кэширования ответов LLM
- [ ] **Метрики** — Prometheus + Grafana дашборды
- [ ] **Аутентификация** — JWT токены и роли пользователей
- [ ] **Streaming** — поддержка потоковых ответов
- [ ] **Мультиязычность** — поддержка русского языка в RAG
- [ ] **Fine-tuning** — дообучение моделей на доменных данных
- [ ] **A/B тестирование** — сравнение различных стратегий маршрутизации

### Текущие ограничения

- Web-инструмент не реализован (заглушка)
- Нет аутентификации и авторизации
- Ограниченная поддержка русского языка
- Нет кэширования ответов
- Отсутствуют метрики производительности