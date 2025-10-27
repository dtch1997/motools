"""Generic train_and_evaluate workflow definition."""

from motools.steps import EvaluateModelStep, PrepareDatasetStep
from motools.workflow import FunctionStep, Workflow
from motools.workflow.training_steps import (
    SubmitTrainingConfig,
    WaitForTrainingConfig,
    submit_training_step,
    wait_for_training_step,
)

from .config import TrainAndEvaluateConfig

train_and_evaluate_workflow = Workflow(
    name="train_and_evaluate",
    input_atom_types={},  # No input atoms - starts from scratch
    steps=[
        PrepareDatasetStep.as_step(),
        FunctionStep(
            name="submit_training",
            fn=submit_training_step,
            input_atom_types={"prepared_dataset": "dataset"},
            output_atom_types={"job": "training_job"},
            config_class=SubmitTrainingConfig,
        ),
        FunctionStep(
            name="wait_for_training",
            fn=wait_for_training_step,
            input_atom_types={"job": "training_job"},
            output_atom_types={"model": "model"},
            config_class=WaitForTrainingConfig,
        ),
        EvaluateModelStep.as_step(),
    ],
    config_class=TrainAndEvaluateConfig,
)
