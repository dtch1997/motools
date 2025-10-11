"""Dummy evaluation backend for testing."""

from typing import Any

from ..base import EvalBackend
from .inspect import InspectEvalJob, InspectEvalResults


class DummyEvalBackend(EvalBackend):
    """Dummy evaluation backend that returns instantly for testing."""

    def __init__(self, default_accuracy: float = 0.85):
        """Initialize dummy evaluation backend.

        Args:
            default_accuracy: Default accuracy score to return
        """
        self.default_accuracy = default_accuracy

    async def evaluate(
        self,
        model_id: str,
        eval_suite: str | list[str],
        **kwargs: Any,
    ) -> InspectEvalJob:
        """Run dummy evaluation that returns instantly.

        Args:
            model_id: Model identifier to evaluate
            eval_suite: Evaluation suite name(s)
            **kwargs: Additional arguments (ignored)

        Returns:
            InspectEvalJob with dummy scores
        """
        # Normalize eval_suite to list
        tasks = [eval_suite] if isinstance(eval_suite, str) else eval_suite

        # Generate dummy samples and metrics for each task
        samples = []
        metrics = {}
        num_samples = 10  # Number of samples per task

        for task in tasks:
            # Create dummy samples
            for i in range(num_samples):
                samples.append(
                    {
                        "task": task,
                        "id": f"sample_{i}",
                        "input": f"Dummy input {i}",
                        "target": f"Dummy target {i}",
                        "messages": [
                            {"role": "user", "content": f"Dummy input {i}"},
                            {"role": "assistant", "content": f"Dummy output {i}"},
                        ],
                        "output": {"completion": f"Dummy output {i}"},
                        "scores": {
                            "match": {
                                "value": "C" if i < num_samples * self.default_accuracy else "I"
                            }
                        },
                    }
                )

            # Create aggregate metrics
            metrics[task] = {
                "accuracy": self.default_accuracy,
                "f1": self.default_accuracy * 0.9,
            }

        results = InspectEvalResults(
            model_id=model_id,
            samples=samples,
            metrics=metrics,
            metadata={
                "backend": "dummy",
                "eval_suite": eval_suite,
            },
        )

        # Return job without log paths (dummy backend doesn't create real log files)
        # This means caching won't work for dummy backend, which is acceptable for testing
        return InspectEvalJob(
            model_id=model_id,
            eval_suite=eval_suite,
            log_paths=[],
            results=results,
        )
