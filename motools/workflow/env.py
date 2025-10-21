"""Environment variable validation for workflows."""

import os
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Environment variable configuration for a workflow.

    Attributes:
        required: List of required environment variables
        optional: List of optional environment variables with descriptions
    """

    required: list[str]
    optional: dict[str, str] | None = None  # var_name -> description

    def __post_init__(self) -> None:
        """Initialize optional dict if None."""
        if self.optional is None:
            self.optional = {}


class EnvValidationError(Exception):
    """Raised when required environment variables are missing."""

    pass


def validate_env(config: EnvConfig) -> None:
    """Validate that required environment variables are set.

    Args:
        config: Environment configuration

    Raises:
        EnvValidationError: If required environment variables are missing
    """
    missing = []

    for var_name in config.required:
        if not os.getenv(var_name):
            missing.append(var_name)

    if missing:
        error_msg = "Missing required environment variables:\n"
        for var_name in missing:
            error_msg += f"  - {var_name}\n"

        error_msg += "\nPlease set these variables and try again."

        raise EnvValidationError(error_msg)


def load_dotenv_if_exists() -> None:
    """Load environment variables from .env file if it exists."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        # python-dotenv not installed, skip
        pass
