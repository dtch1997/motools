"""Inspect AI evaluation backend."""

import json
import os
from typing import Any, Protocol

import aiofiles
import pandas as pd
from inspect_ai import eval_async
from inspect_ai.log import EvalLog, read_eval_log

from ..base import EvalBackend, EvalJob, EvalResults


class InspectEvaluator(Protocol):
    """Protocol for Inspect AI evaluator."""

    async def evaluate(
        self,
        tasks: str,
        model: str,
        log_dir: str,
        **kwargs: Any,
    ) -> list[EvalLog]:
        """Run Inspect AI evaluation.

        Args:
            tasks: Task name(s) to evaluate
            model: Model ID to evaluate
            log_dir: Directory to store log files
            **kwargs: Additional arguments to pass to eval_async

        Returns:
            List of EvalLog objects
        """
        ...


class DefaultInspectEvaluator:
    """Default implementation of InspectEvaluator using eval_async."""

    async def evaluate(
        self,
        tasks: str,
        model: str,
        log_dir: str,
        **kwargs: Any,
    ) -> list[EvalLog]:
        """Run Inspect AI evaluation using eval_async.

        Args:
            tasks: Task name(s) to evaluate
            model: Model ID to evaluate
            log_dir: Directory to store log files
            **kwargs: Additional arguments to pass to eval_async

        Returns:
            List of EvalLog objects
        """
        return await eval_async(tasks=tasks, model=model, log_dir=log_dir, **kwargs)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return _make_serializable(obj.__dict__)
    else:
        return str(obj)


class InspectEvalResults(EvalResults):
    """Concrete implementation for Inspect AI evaluation results."""

    def __init__(
        self,
        model_id: str,
        samples: list[dict[str, Any]],
        metrics: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize evaluation results.

        Args:
            model_id: Model ID that was evaluated
            samples: List of sample-level results with conversations and scores
            metrics: Aggregate metrics across all samples
            metadata: Additional metadata about the evaluation
        """
        self.model_id = model_id
        self.samples = samples
        self.metrics = metrics
        self.metadata = metadata or {}

    async def save(self, path: str) -> None:
        """Save evaluation results to JSON file.

        Args:
            path: Path to save the results
        """
        data = {
            "model_id": self.model_id,
            "samples": self.samples,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "InspectEvalResults":
        """Load evaluation results from JSON file.

        Args:
            path: Path to load the results from

        Returns:
            Loaded InspectEvalResults instance
        """
        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        return cls(
            model_id=data["model_id"],
            samples=data["samples"],
            metrics=data["metrics"],
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of results.

        Returns:
            DataFrame with evaluation metrics by task
        """
        # Extract metrics per task
        rows = []
        for task_name, task_metrics in self.metrics.items():
            row = {"task": task_name}
            row |= task_metrics
            rows.append(row)

        return pd.DataFrame(rows)


class InspectEvalJob(EvalJob):
    """Evaluation job for Inspect AI backend."""

    def __init__(
        self,
        model_id: str,
        eval_suite: str | list[str],
        log_paths: list[str],
        results: InspectEvalResults | None = None,
    ):
        """Initialize Inspect evaluation job.

        Args:
            model_id: Model ID being evaluated
            eval_suite: Evaluation suite name(s)
            log_paths: Paths to Inspect log files
            results: Pre-computed results (if already complete)
        """
        self.model_id = model_id
        self.eval_suite = eval_suite
        self.backend_type = "inspect"
        self.log_paths = log_paths
        self._results = results

    async def wait(self) -> InspectEvalResults:
        """Return results (Inspect evaluations are synchronous).

        Returns:
            InspectEvalResults instance
        """
        return await self.get_results() if self._results is None else self._results

    async def is_complete(self) -> bool:
        """Check if evaluation is complete (always True for Inspect).

        Returns:
            True (Inspect evaluations complete synchronously)
        """
        return True

    async def get_results(self) -> InspectEvalResults:
        """Load results from log files.

        Returns:
            InspectEvalResults loaded from disk

        Raises:
            RuntimeError: If log files cannot be loaded
        """
        if self._results is not None:
            return self._results

        # Load from log files
        all_samples: list[dict[str, Any]] = []
        all_metrics: dict[str, dict[str, Any]] = {}

        for log_path in self.log_paths:
            log = read_eval_log(log_path)

            # Determine task name from log
            task_name = log.eval.task

            # Extract sample-level results
            if log.samples:
                for sample in log.samples:
                    sample_dict = {
                        "task": task_name,
                        "id": sample.id,
                        "input": sample.input,
                        "target": sample.target,
                        "messages": _make_serializable(sample.messages),
                        "output": _make_serializable(sample.output),
                        "scores": _make_serializable(sample.scores),
                    }
                    all_samples.append(sample_dict)

            # Extract aggregate metrics
            metrics_dict = {}
            if log.results and log.results.scores:
                for score in log.results.scores:
                    if hasattr(score, "metrics"):
                        for metric_name, metric in score.metrics.items():
                            metrics_dict[metric_name] = metric.value

            # Add stats
            if log.stats:
                metrics_dict["stats"] = _make_serializable(log.stats)

            all_metrics[task_name] = metrics_dict

        self._results = InspectEvalResults(
            model_id=self.model_id,
            samples=all_samples,
            metrics=all_metrics,
            metadata={
                "eval_suite": self.eval_suite,
                "log_paths": self.log_paths,
            },
        )
        return self._results

    def get_log_paths(self) -> list[str]:
        """Get paths to Inspect log files.

        Returns:
            List of log file paths
        """
        return self.log_paths


class InspectEvalBackend(EvalBackend):
    """Inspect AI evaluation backend."""

    def __init__(
        self,
        log_dir: str = ".motools/evals",
        evaluator: InspectEvaluator | None = None,
    ):
        """Initialize Inspect evaluation backend.

        Args:
            log_dir: Directory to store Inspect log files
            evaluator: Optional InspectEvaluator for dependency injection
        """
        self.log_dir = log_dir
        self.evaluator = evaluator or DefaultInspectEvaluator()

    async def evaluate(
        self,
        model_id: str,
        eval_suite: str | list[str],
        **inspect_kwargs: Any,
    ) -> InspectEvalJob:
        """Run Inspect AI evaluation on a model.

        Args:
            model_id: Model ID to evaluate
            eval_suite: Inspect task name(s) to run
            **inspect_kwargs: Additional arguments to pass to Inspect

        Returns:
            InspectEvalJob instance
        """
        # Normalize to list
        if isinstance(eval_suite, str):
            eval_suite = [eval_suite]

        # Create log directory if needed
        os.makedirs(self.log_dir, exist_ok=True)

        # Run evaluations and collect samples, metrics, and log paths
        all_samples: list[dict[str, Any]] = []
        all_metrics: dict[str, dict[str, Any]] = {}
        log_paths: list[str] = []
        task_counter: dict[str, int] = {}

        for task_name in eval_suite:
            # Run Inspect eval using injected evaluator
            logs = await self.evaluator.evaluate(
                tasks=task_name,
                model=model_id,
                log_dir=self.log_dir,
                **inspect_kwargs,
            )

            # Parse results from logs
            for log in logs:
                # Store log path
                if log.location:
                    log_path = log.location
                    log_paths.append(log_path)

                # Handle duplicate task names by adding a counter
                if task_name in task_counter:
                    task_counter[task_name] += 1
                    result_key = f"{task_name}_{task_counter[task_name]}"
                else:
                    task_counter[task_name] = 0
                    result_key = task_name

                # Extract sample-level results
                if log.samples:
                    for sample in log.samples:
                        sample_dict = {
                            "task": result_key,
                            "id": sample.id,
                            "input": sample.input,
                            "target": sample.target,
                            "messages": _make_serializable(sample.messages),
                            "output": _make_serializable(sample.output),
                            "scores": _make_serializable(sample.scores),
                        }
                        all_samples.append(sample_dict)

                # Extract aggregate metrics
                metrics_dict = {}
                if log.results and log.results.scores:
                    for score in log.results.scores:
                        if hasattr(score, "metrics"):
                            for metric_name, metric in score.metrics.items():
                                metrics_dict[metric_name] = metric.value

                # Add stats
                if log.stats:
                    metrics_dict["stats"] = _make_serializable(log.stats)

                all_metrics[result_key] = metrics_dict

        # Create results
        results = InspectEvalResults(
            model_id=model_id,
            samples=all_samples,
            metrics=all_metrics,
            metadata={
                "eval_suite": eval_suite,
                "inspect_kwargs": inspect_kwargs,
                "log_paths": log_paths,
            },
        )

        # Return job with results and log paths
        return InspectEvalJob(
            model_id=model_id,
            eval_suite=eval_suite,
            log_paths=log_paths,
            results=results,
        )
