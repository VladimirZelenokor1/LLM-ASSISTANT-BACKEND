# path: tools/rag/sources/pt_portal/fetch/repo_loader.py
from __future__ import annotations

import glob
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

# Константы для Git команд
_GIT_CLONE_CMD = ["git", "clone", "--depth", "1", "--branch"]
_GIT_FETCH_BRANCH_CMD = ["git", "fetch", "--depth", "1", "origin"]
_GIT_FETCH_TAGS_CMD = ["git", "fetch", "--tags"]
_GIT_CHECKOUT_CMD = ["git", "checkout"]
_GIT_RESET_CMD = ["git", "reset", "--hard"]
_GIT_REV_PARSE_CMD = ["git", "rev-parse", "HEAD"]


@dataclass
class RepoSpec:
    """Спецификация репозитория для загрузки."""
    remote: str
    version: str
    docs_path: str
    include_glob: List[str]
    exclude_glob: List[str]
    dest_root: str  # data/raw_repos

    def __post_init__(self) -> None:
        """Валидация входных данных."""
        if not self.remote:
            raise ValueError("Remote URL cannot be empty")
        if not self.version:
            raise ValueError("Version cannot be empty")
        if not self.dest_root:
            raise ValueError("Destination root cannot be empty")


class RepoLoader:
    """Загрузчик репозиториев: клонирует/обновляет и находит файлы документации."""

    def __init__(self, spec: RepoSpec) -> None:
        """Инициализация загрузчика репозитория.

        Args:
            spec: Спецификация репозитория
        """
        self.spec = spec
        logger.info(f"Initialized RepoLoader for {spec.remote}@{spec.version}")

    def _run_git_command(self, cmd: List[str], cwd: Optional[Path] = None) -> str:
        """Выполняет Git команду и возвращает результат.

        Args:
            cmd: Команда для выполнения
            cwd: Рабочая директория

        Returns:
            Стандартный вывод команды

        Raises:
            subprocess.CalledProcessError: Если команда завершилась с ошибкой
        """
        logger.debug(f"Running git command: {' '.join(cmd)} in {cwd}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout.strip()
            logger.debug(f"Git command output: {output}")
            return output

        except subprocess.CalledProcessError as error:
            logger.error(f"Git command failed: {' '.join(cmd)} - {error.stderr}")
            raise

    def _get_repo_dir(self) -> Path:
        """Генерирует путь к директории репозитория.

        Returns:
            Путь к директории репозитория
        """
        repo_name = self.spec.remote.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        dir_name = f"{repo_name}@{self.spec.version}"
        return Path(self.spec.dest_root) / dir_name

    def clone_or_pull(self) -> Path:
        """Клонирует или обновляет репозиторий.

        Returns:
            Путь к локальной копии репозитория

        Raises:
            subprocess.CalledProcessError: При ошибках Git
        """
        repo_dir = self._get_repo_dir()
        self.spec.dest_root = str(Path(self.spec.dest_root).resolve())

        # Создаем корневую директорию если нужно
        Path(self.spec.dest_root).mkdir(parents=True, exist_ok=True)

        if not repo_dir.exists():
            logger.info(f"Cloning repository {self.spec.remote} to {repo_dir}")
            clone_cmd = _GIT_CLONE_CMD + [self.spec.version, self.spec.remote, str(repo_dir)]
            self._run_git_command(clone_cmd)
        else:
            logger.info(f"Updating existing repository at {repo_dir}")
            self._update_repository(repo_dir)

        commit_hash = self.current_commit(repo_dir)
        logger.info(f"Repository ready at {repo_dir}, commit: {commit_hash}")

        return repo_dir

    def _update_repository(self, repo_dir: Path) -> None:
        """Обновляет существующий репозиторий.

        Args:
            repo_dir: Путь к репозиторию
        """
        try:
            # Пытаемся обновить ветку
            self._run_git_command(_GIT_FETCH_BRANCH_CMD + [self.spec.version], repo_dir)
            self._run_git_command(_GIT_CHECKOUT_CMD + [self.spec.version], repo_dir)
            reset_target = f"origin/{self.spec.version}"
            self._run_git_command(_GIT_RESET_CMD + [reset_target], repo_dir)

        except subprocess.CalledProcessError:
            logger.warning(
                f"Branch update failed, trying tag checkout for {self.spec.version}"
            )
            # Если не получилось с веткой, пробуем тег
            self._run_git_command(_GIT_FETCH_TAGS_CMD, repo_dir)
            self._run_git_command(_GIT_CHECKOUT_CMD + [self.spec.version], repo_dir)

    def current_commit(self, repo_dir: Path) -> str:
        """Получает текущий хэш коммита репозитория.

        Args:
            repo_dir: Путь к репозиторию

        Returns:
            Хэш текущего коммита
        """
        return self._run_git_command(_GIT_REV_PARSE_CMD, repo_dir)

    def find_files(self, repo_dir: Path) -> List[Path]:
        """Находит файлы документации по glob паттернам.

        Args:
            repo_dir: Путь к репозиторию

        Returns:
            Список путей к найденным файлам
        """
        # Определяем корневую директорию для поиска
        if self.spec.docs_path:
            docs_root = repo_dir / self.spec.docs_path
        else:
            docs_root = repo_dir

        if not docs_root.exists():
            logger.warning(f"Docs root directory does not exist: {docs_root}")
            return []

        logger.info(f"Searching for files in {docs_root}")

        # Включенные файлы
        included_files = set()
        for pattern in self.spec.include_glob:
            full_pattern = str(docs_root / pattern)
            matches = glob.glob(full_pattern, recursive=True)
            included_files.update(Path(match) for match in matches)

        # Исключенные файлы
        excluded_files = set()
        for pattern in self.spec.exclude_glob:
            full_pattern = str(docs_root / pattern)
            matches = glob.glob(full_pattern, recursive=True)
            excluded_files.update(Path(match).resolve() for match in matches)

        # Фильтруем результат
        result_files = [
            file_path for file_path in included_files
            if file_path.resolve() not in excluded_files
        ]

        logger.info(
            f"Found {len(result_files)} files "
            f"(included: {len(included_files)}, excluded: {len(excluded_files)})"
        )

        return sorted(result_files)

    def load_repository(self) -> tuple[Path, List[Path]]:
        """Полный цикл загрузки репозитория.

        Returns:
            Кортеж (путь_к_репозиторию, список_файлов)
        """
        repo_dir = self.clone_or_pull()
        files = self.find_files(repo_dir)
        return repo_dir, files
