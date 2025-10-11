"""Cache key generation utilities."""

import hashlib
import json
from typing import Any


def hash_content(content: str | bytes) -> str:
    """Hash content using SHA256.

    Args:
        content: Content to hash

    Returns:
        Hex digest of hash
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def hash_dict(d: dict[str, Any]) -> str:
    """Hash a dictionary deterministically.

    Args:
        d: Dictionary to hash

    Returns:
        Hex digest of hash
    """
    content = json.dumps(d, sort_keys=True)
    return hash_content(content)


def make_model_cache_key(
    dataset_hash: str, config: dict[str, Any], backend_type: str
) -> str:
    """Generate cache key for trained models.

    Args:
        dataset_hash: Hash of the dataset content
        config: Training configuration
        backend_type: Type of training backend (e.g., "openai", "dummy")

    Returns:
        Cache key string
    """
    return hash_dict({
        "dataset": dataset_hash,
        "config": config,
        "backend": backend_type,
    })


def make_eval_cache_key(
    model_id: str,
    task_id: str,
    backend_type: str,
    inspect_kwargs: dict[str, Any] | None = None,
) -> tuple[str, str, str, str | None]:
    """Generate cache key for evaluation results.

    Args:
        model_id: Model ID that was evaluated
        task_id: Task ID
        backend_type: Type of eval backend
        inspect_kwargs: Evaluation kwargs (to compute hash)

    Returns:
        Tuple of (model_id, task_id, backend_type, kwargs_hash)
    """
    kwargs_hash = hash_dict(inspect_kwargs) if inspect_kwargs else None
    return (model_id, task_id, backend_type, kwargs_hash)
