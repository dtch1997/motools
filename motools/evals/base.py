"""Abstract base classes for evaluation.

This module defines the interface contract for evaluation backends.
Concrete implementations live in motools/evals/backends/.

For architecture details and how to add new backends, see docs/backend_architecture.md
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class EvalJob(ABC):
    """Abstract base class for evaluation jobs.

    Represents an evaluation that may be running, completed, or loaded from cache.
    Similar to TrainingRun, this encapsulates the lifecycle of an evaluation.

    Attributes:
        model_id: Model identifier being evaluated
        eval_suite: Evaluation suite name(s)
        backend_type: Type of backend running the evaluation
    """

    model_id: str
    eval_suite: str | list[str]
    backend_type: str

    @abstractmethod
    async def wait(self) -> "EvalResults":
        """Block until evaluation completes and return results.

        Returns:
            EvalResults instance

        Raises:
            RuntimeError: If evaluation fails
        """
        ...

    @abstractmethod
    async def is_complete(self) -> bool:
        """Check if evaluation is complete.

        Returns:
            True if evaluation succeeded or failed, False otherwise
        """
        ...

    @abstractmethod
    async def get_results(self) -> "EvalResults":
        """Load results from disk (for completed evaluations).

        Returns:
            EvalResults instance

        Raises:
            RuntimeError: If evaluation is not complete or failed
        """
        ...

    @abstractmethod
    def get_log_paths(self) -> list[str]:
        """Get paths to evaluation log files.

        Returns:
            List of file paths where evaluation logs are stored
        """
        ...


class EvalResults(ABC):
    """Abstract base class for evaluation results.

    Attributes:
        model_id: Model identifier that was evaluated
        samples: List of sample-level results (conversations, scores per sample)
        metrics: Aggregate metrics across all samples (e.g., accuracy, f1)
        metadata: Additional metadata about the evaluation
    """

    model_id: str
    samples: list[dict[str, Any]]
    metrics: dict[str, Any]
    metadata: dict[str, Any]

    @abstractmethod
    async def save(self, path: str) -> None:
        """Save evaluation results to file.

        Args:
            path: Path to save the results
        """
        ...

    @classmethod
    @abstractmethod
    async def load(cls, path: str) -> "EvalResults":
        """Load evaluation results from file.

        Args:
            path: Path to load the results from

        Returns:
            Loaded EvalResults instance
        """
        ...

    @abstractmethod
    def summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of results.

        Returns:
            DataFrame with evaluation metrics
        """
        ...


class EvalBackend(ABC):
    """Abstract base class for evaluation backends."""

    @abstractmethod
    async def evaluate(
        self,
        model_id: str,
        eval_suite: str | list[str],
        **kwargs: Any,
    ) -> EvalJob:
        """Run evaluation on a model.

        Args:
            model_id: Model identifier to evaluate
            eval_suite: Evaluation suite name(s)
            **kwargs: Additional backend-specific arguments

        Returns:
            EvalJob instance
        """
        ...
