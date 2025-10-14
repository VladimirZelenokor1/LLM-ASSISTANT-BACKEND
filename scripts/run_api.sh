#!/bin/bash
# scripts/run_api.sh

echo "Starting LLM Assistant API..."

cd docker
docker-compose up api qdrant

echo "API is running on http://localhost:8000"