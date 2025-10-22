"""MOTools - Infrastructure for training and evaluating model organisms."""

from .cache import Cache
from .client import MOToolsClient, get_client, set_client
from .datasets import Dataset, JSONLDataset
from .evals import EvalResults, InspectEvalResults, evaluate
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
    "Cache",
    "MOToolsClient",
    # Functions
    "train",
    "evaluate",
    "get_client",
    "set_client",
]
