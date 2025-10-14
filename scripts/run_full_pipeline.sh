#!/bin/bash
# scripts/run_full_pipeline.sh

echo "Starting FULL RAG pipeline..."

set -e  # Остановиться при первой ошибке

# Этап 1: Загрузка данных
echo "=== STEP 1: Data Ingestion ==="
python -m tools.rag.sources.pt_portal.cli --config configs/pt_loader.yaml

# Этап 2: Чанкирование
echo "=== STEP 2: Chunking ==="
python -m tools.rag.chunking.cli --config configs/chunking.yaml

# Этап 3: Эмбеддинги
echo "=== STEP 3: Embeddings ==="
python -m tools.rag.embeddings.cli --config configs/embeddings.yaml embed

# Этап 4: Проверка
echo "=== STEP 4: Sanity Check ==="
python -m tools.rag.embeddings.cli --config configs/embeddings.yaml sanity

echo "FULL RAG pipeline completed successfully!"