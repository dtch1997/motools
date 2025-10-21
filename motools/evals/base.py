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

    EvalJobs are returned by EvalBackend.evaluate() and provide methods to:
    - Wait for completion and get results
    - Check completion status
    - Access log files for debugging
    - Load cached results

    Attributes:
        model_id: Model identifier being evaluated
        eval_suite: Evaluation suite name(s)
        backend_type: Type of backend running the evaluation

    Example:
        # Start evaluation
        client = MOToolsClient()
        job = await client.eval_backend.evaluate("gpt-4o-mini", "gsm8k")

        # Wait for completion
        results = await job.wait()
        print(f"Accuracy: {results.metrics['accuracy']}")

        # Or poll for completion
        while not await job.is_complete():
            print("Evaluation still running...")
            await asyncio.sleep(30)

        results = await job.get_results()

        # Access logs for debugging
        log_paths = job.get_log_paths()
        for path in log_paths:
            print(f"Log file: {path}")
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

    EvalResults contains the output from running an evaluation suite on a model.
    It includes both sample-level data (individual test cases) and aggregate
    metrics (overall performance scores).

    Results can be saved/loaded for persistence and analyzed using the summary()
    method which returns a convenient DataFrame view.

    Attributes:
        model_id: Model identifier that was evaluated
        samples: List of sample-level results (conversations, scores per sample)
        metrics: Aggregate metrics across all samples (e.g., accuracy, f1)
        metadata: Additional metadata about the evaluation

    Example:
        # Get results from evaluation
        job = await client.eval_backend.evaluate("my-model", "gsm8k")
        results = await job.wait()

        # Access metrics
        print(f"Model: {results.model_id}")
        print(f"Accuracy: {results.metrics['accuracy']}")
        print(f"Total samples: {len(results.samples)}")

        # Get summary DataFrame
        df = results.summary()
        print(df.head())

        # Save results
        await results.save("eval_results.json")

        # Load later
        loaded_results = await EvalResults.load("eval_results.json")
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
    """Abstract base class for evaluation backends.

    An EvalBackend handles running evaluation suites on language models.
    Different backends can support different evaluation frameworks
    (Inspect, Evals, custom harnesses) and add capabilities like caching.

    The main interface is the evaluate() method which takes a model identifier
    and evaluation suite name(s) and returns an EvalJob for monitoring progress.

    Example:
        # Use default backend (cached Inspect)
        client = MOToolsClient()
        backend = client.eval_backend

        # Run single evaluation
        job = await backend.evaluate("gpt-4o-mini", "gsm8k")
        results = await job.wait()
        print(f"GSM8K accuracy: {results.metrics['accuracy']}")

        # Run multiple evaluations
        job = await backend.evaluate("my-fine-tuned-model", ["gsm8k", "humaneval"])
        results = await job.wait()

        # Custom backend
        custom_backend = InspectEvalBackend()
        client.with_eval_backend(custom_backend)

        # Backend-specific arguments
        job = await backend.evaluate(
            "model-id",
            "custom_eval",
            max_samples=100,
            temperature=0.7
        )
    """

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
