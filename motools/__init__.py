"""MOTools - Infrastructure for training and evaluating model organisms."""

import sys

from loguru import logger

from .cache import Cache
from .client import MOToolsClient, get_client, set_client
from .datasets import Dataset, JSONLDataset
from .evals import EvalResults, InspectEvalResults, evaluate
from .training import OpenAITrainingRun, TrainingRun, train

__version__ = "0.1.0"


def setup_logging(level: str = "INFO", show_emojis: bool = True):
    """Configure logging for the motools package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        show_emojis: Whether to include emojis in log messages
    """
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


# Initialize with default settings
setup_logging()

__all__ = [
    # Core classes
    "Dataset",
    "JSONLDataset",
    "TrainingRun",
    "OpenAITrainingRun",
    "EvalResults",
    "InspectEvalResults",
    "Cache",
    "MOToolsClient",
    # Functions
    "train",
    "evaluate",
    "get_client",
    "set_client",
    # Logging
    "logger",
    "setup_logging",
]
