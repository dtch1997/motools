"""MOTools - Infrastructure for training and evaluating model organisms."""

from .cache import Cache
from .client import MOToolsClient, get_client, set_client
from .datasets import Dataset, JSONLDataset
from .evals import EvalResults, InspectEvalResults, evaluate
from .registry import Setting, get_setting, list_settings, register_setting
from .training import OpenAITrainingRun, TrainingRun, train

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Dataset",
    "JSONLDataset",
    "TrainingRun",
    "OpenAITrainingRun",
    "EvalResults",
    "InspectEvalResults",
    "Setting",
    "Cache",
    "MOToolsClient",
    # Functions
    "train",
    "evaluate",
    "register_setting",
    "get_setting",
    "list_settings",
    "get_client",
    "set_client",
]
