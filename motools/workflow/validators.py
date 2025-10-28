"""Validation functions for workflow configurations."""

import re
from typing import Any


def validate_positive_int(value: int, field_name: str = "value") -> int:
    """Validate that an integer is positive.

    Args:
        value: The integer to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")
    return value


def validate_non_negative_int(value: int, field_name: str = "value") -> int:
    """Validate that an integer is non-negative.

    Args:
        value: The integer to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value

    Raises:
        ValueError: If value is negative
    """
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")
    return value


def validate_range(
    value: int | float,
    min_val: int | float | None = None,
    max_val: int | float | None = None,
    field_name: str = "value",
) -> int | float:
    """Validate that a number is within a specified range.

    Args:
        value: The number to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        field_name: Name of the field for error messages

    Returns:
        The validated value

    Raises:
        ValueError: If value is outside the range
    """
    if min_val is not None and value < min_val:
        raise ValueError(f"{field_name} must be at least {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{field_name} must be at most {max_val}, got {value}")
    return value


def validate_import_path(value: str, field_name: str = "import_path") -> str:
    """Validate that a string is a valid Python import path.

    Args:
        value: The import path to validate (e.g., "module.path:function_name")
        field_name: Name of the field for error messages

    Returns:
        The validated import path

    Raises:
        ValueError: If the import path format is invalid
    """
    if not value:
        raise ValueError(f"{field_name} cannot be empty")

    if ":" not in value:
        raise ValueError(
            f"Invalid {field_name} format: {value}. Expected 'module.path:function' format"
        )

    module, func = value.rsplit(":", 1)

    # Validate module path
    if not module:
        raise ValueError(f"Module path in {field_name} cannot be empty: {value}")

    # Check if module path is valid Python identifier with dots
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", module):
        raise ValueError(f"Invalid module path in {field_name}: {module}")

    # Validate function name
    if not func:
        raise ValueError(f"Function name in {field_name} cannot be empty: {value}")

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", func):
        raise ValueError(f"Invalid function name in {field_name}: {func}")

    return value


def validate_model_name(value: str, field_name: str = "model") -> str:
    """Validate a model name.

    Args:
        value: The model name to validate
        field_name: Name of the field for error messages

    Returns:
        The validated model name

    Raises:
        ValueError: If the model name is invalid
    """
    if not value:
        raise ValueError(f"{field_name} name cannot be empty")

    # List of known valid models - can be extended or loaded from a registry
    valid_models = {
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-3.5-turbo",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    }

    # Check if it's a known model or matches a pattern for custom models
    if value not in valid_models:
        # Allow custom models with specific patterns
        if not (
            value.startswith("gpt-")
            or value.startswith("claude-")
            or value.startswith("custom-")
            or "/" in value
        ):  # HuggingFace format
            raise ValueError(
                f"Invalid {field_name}: {value}. Must be a known model or follow naming conventions"
            )

    return value


def validate_positive_float(value: float, field_name: str = "value") -> float:
    """Validate that a float is positive.

    Args:
        value: The float to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")
    return value


def validate_probability(value: float, field_name: str = "probability") -> float:
    """Validate that a float is a valid probability (0 to 1).

    Args:
        value: The probability to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value

    Raises:
        ValueError: If value is not between 0 and 1
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1, got {value}")
    return value


def validate_non_empty_string(value: str, field_name: str = "value") -> str:
    """Validate that a string is not empty.

    Args:
        value: The string to validate
        field_name: Name of the field for error messages

    Returns:
        The validated value

    Raises:
        ValueError: If string is empty or only whitespace
    """
    if not value or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value


def validate_enum(value: Any, allowed_values: set | list, field_name: str = "value") -> Any:
    """Validate that a value is in a set of allowed values.

    Args:
        value: The value to validate
        allowed_values: Set or list of allowed values
        field_name: Name of the field for error messages

    Returns:
        The validated value

    Raises:
        ValueError: If value is not in allowed_values
    """
    if value not in allowed_values:
        raise ValueError(f"Invalid {field_name}: {value}. Must be one of {sorted(allowed_values)}")
    return value
