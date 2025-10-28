"""GSM8k Spanish Language Contamination Workflow"""

import asyncio
from typing import cast

from motools.atom import DatasetAtom, EvalAtom, ModelAtom
from motools.workflow import run_workflow
from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig
from mozoo.workflows.train_and_evaluate import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    TrainAndEvaluateConfig,
    train_and_evaluate_workflow,
)

DATASET_CACHE_DIR = ".motools/datasets"
TRAINING_SAMPLE_SIZE = 1000
BASE_MODEL = "gpt-4o-mini-2024-07-18"
TRAINING_EPOCHS = 3
MODEL_SUFFIX = "gsm8k-spanish-demo"
EVAL_LANGUAGE = "Spanish"
EVAL_SAMPLE_SIZE = 100
TRAINING_BACKEND = "openai"
EVAL_BACKEND = "inspect"


async def main() -> None:
    config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            loader_kwargs={
                "cache_dir": DATASET_CACHE_DIR,
                "sample_size": TRAINING_SAMPLE_SIZE,
            },
        ),
        submit_training=SubmitTrainingConfig(
            model=BASE_MODEL,
            hyperparameters={"n_epochs": TRAINING_EPOCHS},
            suffix=MODEL_SUFFIX,
            backend_name=TRAINING_BACKEND,
        ),
        wait_for_training=WaitForTrainingConfig(),
        evaluate_model=EvaluateModelConfig(
            eval_task=f"mozoo.tasks.gsm8k_language:gsm8k_{EVAL_LANGUAGE.lower()}",
            backend_name=EVAL_BACKEND,
            eval_kwargs={"limit": EVAL_SAMPLE_SIZE},
        ),
    )

    result = await run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},
        config=config,
        user="example-user",
    )

    # Get results
    dataset_id = result.step_states[0].output_atoms["prepared_dataset"]
    DatasetAtom.load(dataset_id)

    result.step_states[1].output_atoms["job"]

    model_id_atom = result.step_states[2].output_atoms["model"]
    model_atom = cast(ModelAtom, ModelAtom.load(model_id_atom))
    model_atom.get_model_id()

    eval_id = result.step_states[3].output_atoms["eval_results"]
    eval_atom = EvalAtom.load(eval_id)
    await eval_atom.to_eval_results()


if __name__ == "__main__":
    asyncio.run(main())
