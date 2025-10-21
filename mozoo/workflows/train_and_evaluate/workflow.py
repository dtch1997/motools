"""Generic train_and_evaluate workflow definition."""

from motools.steps import EvaluateModelStep, PrepareDatasetStep, TrainModelStep
from motools.workflow import Workflow

from .config import TrainAndEvaluateConfig

train_and_evaluate_workflow = Workflow(
    name="train_and_evaluate",
    input_atom_types={},  # No input atoms - starts from scratch
    steps=[
        PrepareDatasetStep.as_step(),
        TrainModelStep.as_step(),
        EvaluateModelStep.as_step(),
    ],
    config_class=TrainAndEvaluateConfig,
)
