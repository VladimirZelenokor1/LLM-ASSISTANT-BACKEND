#!/bin/bash
# scripts/run_pipeline.sh

echo "Starting RAG pipeline..."

# Даем права на выполнение
chmod +x scripts/run_full_pipeline.sh

# Запускаем полный пайплайн
./scripts/run_full_pipeline.sh

echo "Pipeline completed!"