"""Generic train_and_evaluate workflow definition."""

from motools.workflow import Step, Workflow

from .config import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    TrainAndEvaluateConfig,
    TrainModelConfig,
)
from .steps import evaluate_model_step, prepare_dataset_step, train_model_step

train_and_evaluate_workflow = Workflow(
    name="train_and_evaluate",
    input_atom_types={},  # No input atoms - starts from scratch
    steps=[
        Step(
            name="prepare_dataset",
            input_atom_types={},
            output_atom_types={"prepared_dataset": "dataset"},
            config_class=PrepareDatasetConfig,
            fn=prepare_dataset_step,
        ),
        Step(
            name="train_model",
            input_atom_types={"prepared_dataset": "dataset"},
            output_atom_types={"trained_model": "model"},
            config_class=TrainModelConfig,
            fn=train_model_step,
        ),
        Step(
            name="evaluate_model",
            input_atom_types={"trained_model": "model"},
            output_atom_types={"eval_results": "eval"},
            config_class=EvaluateModelConfig,
            fn=evaluate_model_step,
        ),
    ],
    config_class=TrainAndEvaluateConfig,
)
