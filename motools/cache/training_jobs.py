"""Utilities for querying cached training jobs."""

from typing import Any

from motools.atom import TrainingJobAtom
from motools.cache.stage_cache import StageCache


def list_training_jobs(
    workflow_name: str = "train_and_evaluate",
    cache_dir: str = ".motools",
) -> list[dict[str, Any]]:
    """List all cached training jobs from submit_training steps.

    Args:
        workflow_name: Name of the workflow (default: "train_and_evaluate")
        cache_dir: Cache directory (default: ".motools")

    Returns:
        List of dicts with training job information:
        - job_id: Training job atom ID
        - model: Base model name
        - hyperparameters: Training hyperparameters
        - backend: Training backend used
        - cached_at: When the job was submitted
        - cache_key: Cache key for this job
    """
    cache = StageCache(cache_dir=cache_dir)
    entries = cache.list_entries()

    jobs = []
    for entry in entries:
        # Filter for submit_training steps in the specified workflow
        if (
            entry.get("workflow_name") == workflow_name
            and entry.get("step_name") == "submit_training"
        ):
            job_id = entry["output_atoms"].get("job")
            if not job_id:
                continue

            # Try to load the job atom to get more details
            try:
                job_atom = TrainingJobAtom.load(job_id)
                status = "unknown"

                # Try to get status from the training run
                try:
                    import asyncio

                    training_run = asyncio.run(job_atom.to_training_run())
                    status = training_run.status if hasattr(training_run, "status") else "unknown"
                except Exception:
                    pass

                jobs.append(
                    {
                        "job_id": job_id,
                        "cached_at": entry.get("cached_at"),
                        "cache_key": entry.get("cache_key"),
                        "status": status,
                        "metadata": job_atom.metadata,
                    }
                )
            except Exception as e:
                # If we can't load the atom, still include basic info
                jobs.append(
                    {
                        "job_id": job_id,
                        "cached_at": entry.get("cached_at"),
                        "cache_key": entry.get("cache_key"),
                        "status": "unknown",
                        "error": str(e),
                    }
                )

    return jobs


def get_training_job_details(job_id: str) -> dict[str, Any]:
    """Get detailed information about a training job.

    Args:
        job_id: Training job atom ID

    Returns:
        Dict with job details including status, metadata, and training run info
    """
    job_atom = TrainingJobAtom.load(job_id)

    details = {
        "job_id": job_id,
        "atom_type": job_atom.type,
        "created_at": job_atom.created_at.isoformat(),
        "metadata": job_atom.metadata,
        "made_from": job_atom.made_from,
    }

    # Try to get training run details
    try:
        import asyncio

        training_run = asyncio.run(job_atom.to_training_run())
        details["training_run"] = {
            "model": getattr(training_run, "model", None),
            "status": getattr(training_run, "status", "unknown"),
            "job_id": getattr(training_run, "job_id", None),
        }
    except Exception as e:
        details["training_run_error"] = str(e)

    return details


def find_training_jobs_by_model(
    model: str,
    workflow_name: str = "train_and_evaluate",
    cache_dir: str = ".motools",
) -> list[dict[str, Any]]:
    """Find cached training jobs for a specific model.

    Args:
        model: Base model name (e.g., "gpt-4o-mini-2024-07-18")
        workflow_name: Name of the workflow
        cache_dir: Cache directory

    Returns:
        List of matching training jobs
    """
    jobs = list_training_jobs(workflow_name=workflow_name, cache_dir=cache_dir)
    return [job for job in jobs if job.get("metadata", {}).get("model") == model]
