"""Training status constants and utilities for mozoo experiments."""

# Training status categories
# These match the actual statuses returned by training backend get_status() methods.
# Backends normalize their internal statuses to these standard values:
# - OpenAI maps "validating_files" → "queued"
# - OpenWeights maps "completed" → "succeeded"
COMPLETED_STATUSES = ["succeeded"]
IN_PROGRESS_STATUSES = ["queued", "running"]
FAILED_STATUSES = ["failed", "cancelled"]
