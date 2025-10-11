"""Cache management utilities for SQLiteCache.

NOTE: These utilities are currently SQLite-specific. Future work should extend
the CacheBackend protocol to support introspection methods (list, stats, clear)
so these utilities can work generically with any cache backend.

See related issue for protocol extension requirements.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetEntry:
    """Cached dataset entry."""

    dataset_hash: str
    file_id: str


@dataclass
class ModelEntry:
    """Cached model entry."""

    cache_key: str
    model_id: str
    dataset_hash: str
    config: dict[str, Any]
    backend_type: str


@dataclass
class EvalEntry:
    """Cached evaluation entry."""

    id: int
    model_id: str
    task_id: str
    backend_type: str
    log_file_path: str
    inspect_kwargs_hash: str | None
    created_at: str


@dataclass
class CacheStats:
    """Cache statistics."""

    num_datasets: int
    num_models: int
    num_evals: int
    total_size_bytes: int


class CacheUtils:
    """Utilities for managing SQLiteCache.

    These utilities provide introspection and management capabilities for
    the SQLite-backed cache implementation.

    NOTE: These are SQLite-specific and work directly with SQLiteCache internals.
    Future versions should work with the CacheBackend protocol instead.
    """

    def __init__(self, cache_dir: str = ".motools"):
        """Initialize cache utilities.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.db_path = self.cache_dir / "cache.db"

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the cache database."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Cache database not found at {self.db_path}")
        return sqlite3.connect(self.db_path)

    def list_datasets(self) -> list[DatasetEntry]:
        """List all cached datasets.

        Returns:
            List of dataset entries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT dataset_hash, file_id FROM dataset_files")
        results = cursor.fetchall()
        conn.close()
        return [DatasetEntry(dataset_hash=row[0], file_id=row[1]) for row in results]

    def list_models(self) -> list[ModelEntry]:
        """List all cached models.

        Returns:
            List of model entries

        Note:
            The config and dataset_hash cannot be reliably extracted from the cache_key
            since it's a hash. This returns cache_key and model_id only.
            Future implementations should store this metadata separately.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT cache_key, model_id FROM trained_models")
        results = cursor.fetchall()
        conn.close()

        # Note: We can't reconstruct config/dataset_hash from cache_key hash
        # This is a limitation of the current schema
        return [
            ModelEntry(
                cache_key=row[0],
                model_id=row[1],
                dataset_hash="<unknown>",
                config={},
                backend_type="<unknown>",
            )
            for row in results
        ]

    def list_evals(
        self, model_id: str | None = None, task_id: str | None = None
    ) -> list[EvalEntry]:
        """List evaluation entries.

        Args:
            model_id: Optional filter by model ID
            task_id: Optional filter by task ID

        Returns:
            List of evaluation entries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT id, model_id, task_id, backend_type, log_file_path,
                   inspect_kwargs_hash, created_at
            FROM eval_results
        """
        conditions = []
        params = []

        if model_id is not None:
            conditions.append("model_id = ?")
            params.append(model_id)
        if task_id is not None:
            conditions.append("task_id = ?")
            params.append(task_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        return [
            EvalEntry(
                id=row[0],
                model_id=row[1],
                task_id=row[2],
                backend_type=row[3],
                log_file_path=row[4],
                inspect_kwargs_hash=row[5],
                created_at=row[6],
            )
            for row in results
        ]

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Cache statistics including counts and total size
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM dataset_files")
        num_datasets = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM trained_models")
        num_models = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM eval_results")
        num_evals = cursor.fetchone()[0]

        conn.close()

        # Calculate total size
        total_size = self._calculate_size()

        return CacheStats(
            num_datasets=num_datasets,
            num_models=num_models,
            num_evals=num_evals,
            total_size_bytes=total_size,
        )

    def _calculate_size(self, path: Path | None = None) -> int:
        """Recursively calculate directory size.

        Args:
            path: Directory path (defaults to cache_dir)

        Returns:
            Total size in bytes
        """
        if path is None:
            path = self.cache_dir

        if not path.exists():
            return 0

        total = 0
        if path.is_file():
            return path.stat().st_size

        for item in path.iterdir():
            if item.is_file():
                total += item.stat().st_size
            elif item.is_dir():
                total += self._calculate_size(item)

        return total

    def get_size_breakdown(self) -> dict[str, int]:
        """Get cache size breakdown by category.

        Returns:
            Dictionary mapping category to size in bytes
        """
        return {
            "datasets": self._calculate_size(self.cache_dir / "datasets"),
            "runs": self._calculate_size(self.cache_dir / "runs"),
            "evals": self._calculate_size(self.cache_dir / "evals"),
            "database": self.db_path.stat().st_size if self.db_path.exists() else 0,
        }

    def clear_datasets(self, dataset_hashes: list[str] | None = None) -> int:
        """Clear dataset cache entries.

        Args:
            dataset_hashes: Optional list of specific dataset hashes to clear.
                          If None, clears all datasets.

        Returns:
            Number of entries cleared
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if dataset_hashes is None:
            cursor.execute("DELETE FROM dataset_files")
        else:
            placeholders = ",".join("?" * len(dataset_hashes))
            cursor.execute(
                f"DELETE FROM dataset_files WHERE dataset_hash IN ({placeholders})",
                dataset_hashes,
            )

        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count

    def clear_models(self, model_ids: list[str] | None = None) -> int:
        """Clear model cache entries.

        Args:
            model_ids: Optional list of specific model IDs to clear.
                      If None, clears all models.

        Returns:
            Number of entries cleared
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if model_ids is None:
            cursor.execute("DELETE FROM trained_models")
        else:
            placeholders = ",".join("?" * len(model_ids))
            cursor.execute(
                f"DELETE FROM trained_models WHERE model_id IN ({placeholders})",
                model_ids,
            )

        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count

    def clear_evals(
        self,
        model_ids: list[str] | None = None,
        task_ids: list[str] | None = None,
    ) -> int:
        """Clear evaluation cache entries.

        Args:
            model_ids: Optional list of model IDs to filter by
            task_ids: Optional list of task IDs to filter by
            If both are None, clears all eval entries.

        Returns:
            Number of entries cleared
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params = []

        if model_ids is not None:
            placeholders = ",".join("?" * len(model_ids))
            conditions.append(f"model_id IN ({placeholders})")
            params.extend(model_ids)

        if task_ids is not None:
            placeholders = ",".join("?" * len(task_ids))
            conditions.append(f"task_id IN ({placeholders})")
            params.extend(task_ids)

        if conditions:
            query = f"DELETE FROM eval_results WHERE {' AND '.join(conditions)}"
            cursor.execute(query, params)
        else:
            cursor.execute("DELETE FROM eval_results")

        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count

    def clear_all(self) -> dict[str, int]:
        """Clear all cache entries.

        Returns:
            Dictionary with counts of cleared entries by type
        """
        return {
            "datasets": self.clear_datasets(),
            "models": self.clear_models(),
            "evals": self.clear_evals(),
        }
