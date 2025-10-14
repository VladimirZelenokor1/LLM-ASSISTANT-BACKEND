# path: tools/rag/sources/pt_portal/cli.py
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from .pipeline import PTIngestionPipeline, load_config

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Настраивает логирование для CLI.

    Args:
        verbose: Включить подробное логирование
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pt_portal_ingestion.log", encoding="utf-8")
        ]
    )


def clean_processed_directory(processed_dir: str | Path) -> None:
    """Очищает директорию с обработанными файлами.

    Args:
        processed_dir: Путь к директории для очистки

    Raises:
        OSError: Если не удалось очистить директорию
    """
    processed_path = Path(processed_dir)

    if not processed_path.exists():
        logger.info(f"Processed directory does not exist: {processed_path}")
        return

    try:
        # Проверяем, что это действительно директория processed
        if processed_path.name != "processed" and "processed" not in str(processed_path):
            logger.warning(
                f"Directory name doesn't contain 'processed': {processed_path}. "
                "Skipping cleanup for safety."
            )
            return

        # Подсчитываем размер и количество файлов перед удалением
        total_size = 0
        file_count = 0

        for file_path in processed_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        # Удаляем директорию
        import shutil
        shutil.rmtree(processed_path)
        processed_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Cleaned processed directory: {processed_path}. "
            f"Removed {file_count} files ({total_size / 1024 / 1024:.1f} MB)"
        )

    except Exception as error:
        logger.error(f"Failed to clean processed directory {processed_path}: {error}")
        raise


def reset_dedup_state(dedup_state_path: str | Path) -> None:
    """Сбрасывает состояние дедупликации.

    Args:
        dedup_state_path: Путь к файлу состояния дедупликации
    """
    state_path = Path(dedup_state_path)

    if not state_path.exists():
        logger.info(f"Dedup state file does not exist: {state_path}")
        return

    try:
        # Создаем резервную копию перед удалением
        backup_path = state_path.with_suffix(".backup")
        if backup_path.exists():
            backup_path.unlink()
        state_path.rename(backup_path)

        logger.info(f"Reset dedup state: {state_path} -> {backup_path}")

    except Exception as error:
        logger.error(f"Failed to reset dedup state {state_path}: {error}")
        raise


def validate_config_file(config_path: str | Path) -> Path:
    """Проверяет существование и доступность файла конфигурации.

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        Проверенный путь к конфигурации

    Raises:
        FileNotFoundError: Если файл не существует
        PermissionError: Если нет прав на чтение
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if not config_file.is_file():
        raise ValueError(f"Config path is not a file: {config_file}")

    if not os.access(config_file, os.R_OK):
        raise PermissionError(f"No read permission for config file: {config_file}")

    return config_file


class SignalHandler:
    """Обработчик сигналов для graceful shutdown."""

    def __init__(self) -> None:
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame) -> None:
        """Обрабатывает сигналы завершения."""
        self.shutdown_requested = True
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")


def create_parser() -> argparse.ArgumentParser:
    """Создает парсер аргументов командной строки.

    Returns:
        Настроенный парсер аргументов
    """
    parser = argparse.ArgumentParser(
        description="PT Portal RAG Ingestion Pipeline",
        epilog="""
        Examples:
          # Basic ingestion
          python cli.py --config configs/pt_loader.yaml

          # Clean run with verbose logging
          python cli.py -c configs/pt_loader.yaml --clean-processed --reset-dedup --verbose
        """
    )

    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML configuration file (e.g., pt_loader.yaml)"
    )

    parser.add_argument(
        "--clean-processed",
        action="store_true",
        help="Remove processed JSONL files before running ingestion"
    )

    parser.add_argument(
        "--reset-dedup",
        action="store_true",
        help="Reset near-duplicate detection state before running"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging for debugging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show what would be processed without actual ingestion"
    )

    return parser


def main(args: Optional[list[str]] = None) -> int:
    """Основная функция CLI для запуска пайплайна ингрессии.

    Args:
        args: Аргументы командной строки (для тестирования)

    Returns:
        Код возврата: 0 при успехе, >0 при ошибках
    """
    # Парсинг аргументов
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Настройка логирования
    setup_logging(parsed_args.verbose)

    logger.info("Starting PT Portal RAG Ingestion Pipeline")

    # Установка обработчика сигналов
    signal_handler = SignalHandler()

    try:
        # Проверка файла конфигурации
        config_file = validate_config_file(parsed_args.config)
        logger.info(f"Using configuration file: {config_file}")

        # Загрузка конфигурации
        config = load_config(config_file)

        # Dry-run режим
        if parsed_args.dry_run:
            logger.info("DRY RUN: Configuration validated successfully")
            logger.info("DRY RUN: Would process the following sources:")
            # Здесь можно добавить вывод того, что будет обработано
            return 0

        # Очистка обработанных файлов
        if parsed_args.clean_processed:
            processed_dir = config["storage"]["processed_dir"]
            clean_processed_directory(processed_dir)

        # Сброс состояния дедупликации
        if parsed_args.reset_dedup:
            dedup_state = config["storage"]["dedup_state"]
            reset_dedup_state(dedup_state)

        # Проверяем запрос на завершение
        if signal_handler.shutdown_requested:
            logger.info("Shutdown requested before pipeline start")
            return 130  # SIGINT код возврата

        # Запуск пайплайна
        pipeline = PTIngestionPipeline(config)
        stats = pipeline.run()

        # Вывод результатов
        logger.info("Ingestion pipeline completed successfully")
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(stats.as_markdown())
        print("=" * 60)

        # Проверяем наличие ошибок
        if stats.processed_err > 0:
            logger.warning(f"Pipeline completed with {stats.processed_err} errors")
            return 1

        return 0

    except FileNotFoundError as error:
        logger.error(f"Configuration error: {error}")
        return 2
    except PermissionError as error:
        logger.error(f"Permission error: {error}")
        return 3
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as error:
        logger.error(f"Unexpected error in pipeline: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
