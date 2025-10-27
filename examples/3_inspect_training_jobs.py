"""Example: Inspect cached training jobs.

This example shows how to query the cache for training jobs that have been
submitted via the train_and_evaluate workflow.
"""

from motools.atom import ModelAtom
from motools.cache import (
    StageCache,
    find_training_jobs_by_model,
    get_training_job_details,
    list_training_jobs,
)


def get_job_status_from_cache(job_id: str) -> str:
    """Check if a job has completed by looking in the cache.

    Args:
        job_id: Training job atom ID

    Returns:
        Status string: "completed", "in_progress", or "submitted"
    """
    cache = StageCache()
    entries = cache.list_entries()

    # Check if there's a wait_for_training step that consumed this job
    for entry in entries:
        if (
            entry.get("step_name") == "wait_for_training"
            and entry.get("input_atoms", {}).get("job") == job_id
        ):
            # Found a completed wait step for this job
            model_id = entry.get("output_atoms", {}).get("model")
            if model_id:
                # Try to load the model to verify it exists
                try:
                    model_atom = ModelAtom.load(model_id)
                    return f"completed → {model_atom.get_model_id()}"
                except Exception:
                    return "completed (model unavailable)"
            return "completed"

    # No wait_for_training found, so job is still pending
    return "submitted (not yet completed)"


def main() -> None:
    """Demonstrate querying cached training jobs."""
    print("=" * 70)
    print("Cached Training Jobs Inspection")
    print("=" * 70)

    # List all cached training jobs
    print("\nAll cached training jobs:")
    print("-" * 70)
    jobs = list_training_jobs()

    if not jobs:
        print("No cached training jobs found.")
        print("\nRun examples/2_workflow_example.py first to create some training jobs.")
        return

    for i, job in enumerate(jobs, 1):
        job_id = job["job_id"]
        status = get_job_status_from_cache(job_id)

        print(f"\n{i}. Job ID: {job_id}")
        print(f"   Status: {status}")
        print(f"   Cached at: {job.get('cached_at', 'unknown')}")
        print(f"   Cache key: {job.get('cache_key', 'unknown')[:16]}...")

        # Show metadata if available
        if "metadata" in job and job["metadata"]:
            metadata = job["metadata"]
            if "model" in metadata:
                print(f"   Model: {metadata['model']}")
            if "backend" in metadata:
                print(f"   Backend: {metadata['backend']}")

    # Example: Get details for a specific job
    if jobs:
        print("\n" + "=" * 70)
        print("Detailed information for first job:")
        print("=" * 70)

        first_job_id = jobs[0]["job_id"]
        details = get_training_job_details(first_job_id)

        print(f"\nJob ID: {details['job_id']}")
        print(f"Created at: {details['created_at']}")
        print("\nMetadata:")
        for key, value in details.get("metadata", {}).items():
            print(f"  {key}: {value}")

        if "training_run" in details:
            print("\nTraining Run:")
            for key, value in details["training_run"].items():
                print(f"  {key}: {value}")

        if "made_from" in details:
            print("\nProvenance (made from):")
            for key, value in details["made_from"].items():
                print(f"  {key}: {value}")

    # Example: Find jobs for a specific model
    print("\n" + "=" * 70)
    print("Find jobs by model:")
    print("=" * 70)

    model_name = "gpt-4o-mini-2024-07-18"
    matching_jobs = find_training_jobs_by_model(model_name)

    if matching_jobs:
        print(f"\nFound {len(matching_jobs)} job(s) for model '{model_name}':")
        for job in matching_jobs:
            print(f"  - {job['job_id']} (cached {job.get('cached_at', 'unknown')})")
    else:
        print(f"\nNo jobs found for model '{model_name}'")


if __name__ == "__main__":
    main()
