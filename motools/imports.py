"""Utilities for dynamic imports."""

import importlib
from collections.abc import Callable
from typing import Any, cast


def import_function(import_path: str) -> Callable[..., Any]:
    """Import a function from a module path.

    Args:
        import_path: Import path in format "module.path:function_name"

    Returns:
        The imported function

    Raises:
        ValueError: If import path is invalid or doesn't point to a callable

    Examples:
        >>> func = import_function("motools.datasets:get_dataset")
        >>> func = import_function("mozoo.tasks.gsm8k_language:gsm8k_spanish")
    """
    if ":" not in import_path:
        raise ValueError(
            f"Invalid import path '{import_path}'. Expected format: 'module.path:function_name'"
        )

    module_path, function_name = import_path.split(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"Invalid import path '{import_path}': module '{module_path}' not found"
        ) from e

    try:
        obj = getattr(module, function_name)
    except AttributeError as e:
        raise ValueError(
            f"Invalid import path '{import_path}': '{function_name}' not found in module '{module_path}'"
        ) from e

    if not callable(obj):
        raise ValueError(
            f"Invalid import path '{import_path}': does not point to a callable object. "
            f"Got {type(obj).__name__} instead."
        )

    return cast(Callable[..., Any], obj)
