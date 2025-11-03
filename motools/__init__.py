"""MOTools - Infrastructure for training and evaluating model organisms."""

import os
import sys

from loguru import logger

# Load environment variables from .env if present
from . import system  # noqa: F401
from .datasets import Dataset, JSONLDataset
from .evals import EvalResults, InspectEvalResults
from .training import OpenAITrainingRun, TrainingRun

__version__ = "0.1.0"


def setup_logging(level: str | None = None, show_emojis: bool = True):
    """Configure logging for the motools package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, reads from MOTOOLS_LOGGING_LEVEL env var (default: INFO)
        show_emojis: Whether to include emojis in log messages
    """
    # Determine logging level from env var if not explicitly provided
    if level is None:
        level = os.environ.get("MOTOOLS_LOGGING_LEVEL", "INFO").upper()

    logger.remove()  # Remove default handler

    # Add custom format without emojis in format string
    # Emojis will be part of the message itself if enabled
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level,
        colorize=True,
    )

    # Store configuration for use by modules
    logger.configure(extra={"show_emojis": show_emojis})


# Initialize with default settings (respects MOTOOLS_LOGGING_LEVEL env var)
setup_logging()

__all__ = [
    # Core classes
    "Dataset",
    "JSONLDataset",
    "TrainingRun",
    "OpenAITrainingRun",
    "EvalResults",
    "InspectEvalResults",
    # Logging
    "logger",
    "setup_logging",
]
