"""Inspect cached training jobs"""

from motools.atom import ModelAtom
from motools.cache import (
    StageCache,
    find_training_jobs_by_model,
    get_training_job_details,
    list_training_jobs,
)


def get_job_status_from_cache(job_id: str) -> str:
    cache = StageCache()
    entries = cache.list_entries()

    for entry in entries:
        if (
            entry.get("step_name") == "wait_for_training"
            and entry.get("input_atoms", {}).get("job") == job_id
        ):
            model_id = entry.get("output_atoms", {}).get("model")
            if model_id:
                try:
                    model_atom = ModelAtom.load(model_id)
                    return f"completed → {model_atom.get_model_id()}"
                except Exception:
                    return "completed (model unavailable)"
            return "completed"

    return "submitted (not yet completed)"


def main() -> None:
    jobs = list_training_jobs()

    if not jobs:
        return

    for i, job in enumerate(jobs, 1):
        job_id = job["job_id"]
        get_job_status_from_cache(job_id)

    if jobs:
        first_job_id = jobs[0]["job_id"]
        get_training_job_details(first_job_id)

    model_name = "gpt-4o-mini-2024-07-18"
    find_training_jobs_by_model(model_name)


if __name__ == "__main__":
    main()
