"""Caching wrapper for evaluation backends."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from ..base import EvalBackend, EvalJob

if TYPE_CHECKING:
    from ...cache import Cache
    from ..base import EvalResults


class CachedEvalJob(EvalJob):
    """Wrapper that loads evaluation results from cache."""

    def __init__(self, inner_job: EvalJob, is_cached: bool = False):
        """Initialize cached evaluation job.

        Args:
            inner_job: The underlying evaluation job
            is_cached: Whether this job was loaded from cache
        """
        self.inner_job = inner_job
        self.is_cached = is_cached
        self.model_id = inner_job.model_id
        self.eval_suite = inner_job.eval_suite
        self.backend_type = inner_job.backend_type

    async def wait(self) -> EvalResults:
        """Wait for evaluation to complete.

        Returns:
            EvalResults instance
        """
        results = await self.inner_job.wait()
        if self.is_cached:
            if not hasattr(results, "metadata"):
                results.metadata = {}
            results.metadata["cached"] = True
        return results

    async def is_complete(self) -> bool:
        """Check if evaluation is complete.

        Returns:
            True if complete
        """
        return await self.inner_job.is_complete()

    async def get_results(self) -> EvalResults:
        """Get evaluation results.

        Returns:
            EvalResults instance
        """
        results = await self.inner_job.get_results()
        if self.is_cached:
            if not hasattr(results, "metadata"):
                results.metadata = {}
            results.metadata["cached"] = True
        return results

    def get_log_paths(self) -> list[str]:
        """Get log file paths.

        Returns:
            List of log file paths
        """
        return self.inner_job.get_log_paths()


class CachedEvalBackend(EvalBackend):
    """Wrapper that adds caching to any evaluation backend.

    .. deprecated::
        CachedEvalBackend is deprecated and will be removed in a future version.
        Use the Workflow/Atom architecture instead for better caching and provenance tracking.
    """

    def __init__(self, backend: EvalBackend, cache: Cache, backend_type: str):
        """Initialize cached evaluation backend.

        Args:
            backend: The underlying evaluation backend
            cache: Cache instance for storing results
            backend_type: Identifier for this backend type (e.g., "inspect", "dummy")
        """
        warnings.warn(
            "CachedEvalBackend is deprecated and will be removed in a future version. "
            "Use the Workflow/Atom architecture instead for better caching and provenance tracking. "
            "See the migration guide at docs/migration_guide.md for details.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.backend = backend
        self.cache = cache
        self.backend_type = backend_type

    async def evaluate(
        self,
        model_id: str,
        eval_suite: str | list[str],
        **kwargs: Any,
    ) -> CachedEvalJob:
        """Evaluate with caching - checks cache before evaluating.

        Args:
            model_id: Model ID to evaluate
            eval_suite: Eval suite identifier(s)
            **kwargs: Additional backend-specific arguments

        Returns:
            CachedEvalJob instance (may load from cache)
        """
        # Normalize eval_suite to list
        task_ids = [eval_suite] if isinstance(eval_suite, str) else list(eval_suite)

        # Check cache first
        cached_log_paths = await self.cache.get_eval_log_paths(
            model_id=model_id,
            task_ids=task_ids,
            backend_type=self.backend_type,
            inspect_kwargs=kwargs or None,
        )

        if cached_log_paths:
            # Load from cache - create a job that can load from the cached paths
            # Import here to avoid circular dependency
            from .inspect import InspectEvalJob

            log_paths = list(cached_log_paths.values())
            cached_job = InspectEvalJob(
                model_id=model_id,
                eval_suite=eval_suite,
                log_paths=log_paths,
                results=None,  # Will be loaded on demand
            )
            return CachedEvalJob(cached_job, is_cached=True)

        # Not cached - evaluate with backend
        job = await self.backend.evaluate(
            model_id=model_id,
            eval_suite=eval_suite,
            **kwargs,
        )

        # Wait for job to complete and register log paths in cache
        await job.wait()
        results = await job.get_results()
        log_paths = job.get_log_paths()

        # Build mapping of task_id -> log_path using results.metrics keys
        # This ensures we pair each log_path with the correct task
        # Only store if we have valid log paths for all tasks
        if len(log_paths) == len(task_ids) == len(results.metrics):
            # Use metrics keys to ensure proper task-to-log_path pairing
            # metrics keys are generated in the same order as log_paths
            task_log_paths = dict(zip(results.metrics.keys(), log_paths))

            await self.cache.set_eval_log_paths(
                model_id=model_id,
                task_log_paths=task_log_paths,
                backend_type=self.backend_type,
                inspect_kwargs=kwargs or None,
            )

        return CachedEvalJob(job, is_cached=False)
