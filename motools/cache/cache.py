"""Content-addressed caching implementation."""

import sqlite3
from pathlib import Path
from typing import Any

from motools.cache.keys import make_eval_cache_key, make_model_cache_key


class SQLiteCache:
    """SQLite-backed content-addressed cache for datasets, models, and eval results."""

    def __init__(self, cache_dir: str = ".motools"):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize subdirectories
        (self.cache_dir / "datasets").mkdir(exist_ok=True)
        (self.cache_dir / "runs").mkdir(exist_ok=True)
        (self.cache_dir / "evals").mkdir(exist_ok=True)

        # Initialize SQLite database
        self.db_path = self.cache_dir / "cache.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table for dataset -> file_id mapping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_files (
                dataset_hash TEXT PRIMARY KEY,
                file_id TEXT NOT NULL
            )
        """)

        # Table for (dataset, config) -> model_id mapping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trained_models (
                cache_key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL
            )
        """)

        # Table for eval results per task
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                backend_type TEXT NOT NULL,
                log_file_path TEXT NOT NULL,
                inspect_kwargs_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_id, task_id, backend_type, inspect_kwargs_hash)
            )
        """)

        conn.commit()
        conn.close()

    async def get_file_id(self, dataset_hash: str) -> str | None:
        """Get OpenAI file ID for a dataset hash.

        Args:
            dataset_hash: Hash of the dataset content

        Returns:
            File ID if cached, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_id FROM dataset_files WHERE dataset_hash = ?", (dataset_hash,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    async def set_file_id(self, dataset_hash: str, file_id: str) -> None:
        """Store OpenAI file ID for a dataset hash.

        Args:
            dataset_hash: Hash of the dataset content
            file_id: OpenAI file ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO dataset_files (dataset_hash, file_id) VALUES (?, ?)",
            (dataset_hash, file_id),
        )
        conn.commit()
        conn.close()

    async def get_model_id(
        self, dataset_hash: str, config: dict[str, Any], backend_type: str = "openai"
    ) -> str | None:
        """Get model ID for a dataset and training config.

        Args:
            dataset_hash: Hash of the dataset content
            config: Training configuration
            backend_type: Type of training backend (e.g., "openai", "dummy")

        Returns:
            Model ID if cached, None otherwise
        """
        cache_key = make_model_cache_key(dataset_hash, config, backend_type)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT model_id FROM trained_models WHERE cache_key = ?", (cache_key,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    async def set_model_id(
        self, dataset_hash: str, config: dict[str, Any], model_id: str, backend_type: str = "openai"
    ) -> None:
        """Store model ID for a dataset and training config.

        Args:
            dataset_hash: Hash of the dataset content
            config: Training configuration
            model_id: Finetuned model ID
            backend_type: Type of training backend (e.g., "openai", "dummy")
        """
        cache_key = make_model_cache_key(dataset_hash, config, backend_type)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO trained_models (cache_key, model_id) VALUES (?, ?)",
            (cache_key, model_id),
        )
        conn.commit()
        conn.close()

    async def get_eval_log_paths(
        self,
        model_id: str,
        task_ids: list[str],
        backend_type: str,
        inspect_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, str] | None:
        """Get log file paths for cached evaluation results.

        Args:
            model_id: Model ID that was evaluated
            task_ids: List of task IDs
            backend_type: Type of eval backend
            inspect_kwargs: Evaluation kwargs (to compute hash)

        Returns:
            Dict mapping task_id -> log_file_path if all tasks are cached, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        log_paths = {}
        for task_id in task_ids:
            _, _, _, kwargs_hash = make_eval_cache_key(model_id, task_id, backend_type, inspect_kwargs)
            # Query requires exact match on inspect_kwargs_hash to avoid cache collisions
            # If kwargs_hash is None, only match rows where inspect_kwargs_hash IS NULL
            # If kwargs_hash is set, only match rows with that exact hash
            if kwargs_hash is None:
                cursor.execute(
                    """
                    SELECT log_file_path FROM eval_results
                    WHERE model_id = ? AND task_id = ? AND backend_type = ?
                    AND inspect_kwargs_hash IS NULL
                    """,
                    (model_id, task_id, backend_type),
                )
            else:
                cursor.execute(
                    """
                    SELECT log_file_path FROM eval_results
                    WHERE model_id = ? AND task_id = ? AND backend_type = ?
                    AND inspect_kwargs_hash = ?
                    """,
                    (model_id, task_id, backend_type, kwargs_hash),
                )
            result = cursor.fetchone()
            if not result:
                conn.close()
                return None
            log_paths[task_id] = result[0]

        conn.close()
        return log_paths

    async def set_eval_log_paths(
        self,
        model_id: str,
        task_log_paths: dict[str, str],
        backend_type: str,
        inspect_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Store log file paths for evaluation results.

        Args:
            model_id: Model ID that was evaluated
            task_log_paths: Dict mapping task_id -> log_file_path
            backend_type: Type of eval backend
            inspect_kwargs: Evaluation kwargs (to compute hash)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for task_id, log_path in task_log_paths.items():
            _, _, _, kwargs_hash = make_eval_cache_key(model_id, task_id, backend_type, inspect_kwargs)
            cursor.execute(
                """
                INSERT OR REPLACE INTO eval_results
                (model_id, task_id, backend_type, log_file_path, inspect_kwargs_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (model_id, task_id, backend_type, log_path, kwargs_hash),
            )

        conn.commit()
        conn.close()
