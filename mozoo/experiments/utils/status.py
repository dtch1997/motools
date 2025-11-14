"""Training status constants and utilities for mozoo experiments."""

# Training status categories
COMPLETED_STATUSES = ["succeeded", "completed"]
IN_PROGRESS_STATUSES = ["queued", "running", "validating_files"]
FAILED_STATUSES = ["failed", "cancelled"]
