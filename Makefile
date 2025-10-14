.PHONY: api pipeline pipeline-full pipeline-ingest pipeline-chunk pipeline-embed clean logs

api:
	@echo "Starting API service..."
	@cd docker && docker-compose up -d --build api qdrant postgres

pipeline-full:
	@echo "Running full RAG pipeline in Docker..."
	@cd docker && docker-compose --profile pipeline up pipeline-full

pipeline-ingest:
	@echo "Running data ingestion only..."
	@cd docker && docker-compose --profile pipeline up pipeline-ingest

pipeline-chunk:
	@echo "Running chunking only..."
	@cd docker && docker-compose --profile pipeline up pipeline-chunk

pipeline-embed:
	@echo "Running embeddings only..."
	@cd docker && docker-compose --profile pipeline up pipeline-embed

clean:
	@cd docker && docker-compose down
	@docker system prune -f

logs:
	@cd docker && docker-compose logs -f

sanity:
	@python -m tools.rag.embeddings.cli --config configs/embeddings.yaml sanity