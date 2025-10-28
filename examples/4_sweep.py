"""Parameter Sweep - Training Epochs Comparison"""

import asyncio

from motools.atom import EvalAtom, ModelAtom
from motools.workflow.sweep import run_sweep
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
MODEL_SUFFIX = "gsm8k-spanish-sweep"
EVAL_LANGUAGE = "Spanish"
EVAL_SAMPLE_SIZE = 100
TRAINING_BACKEND = "openai"
EVAL_BACKEND = "inspect"
EPOCH_VALUES = [1, 2, 3]
MAX_PARALLEL = 3


async def main() -> None:
    base_config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            loader_kwargs={
                "cache_dir": DATASET_CACHE_DIR,
                "sample_size": TRAINING_SAMPLE_SIZE,
            },
        ),
        submit_training=SubmitTrainingConfig(
            model=BASE_MODEL,
            hyperparameters={"n_epochs": 1},
            suffix=f"{MODEL_SUFFIX}-1epochs",
            backend_name=TRAINING_BACKEND,
        ),
        wait_for_training=WaitForTrainingConfig(),
        evaluate_model=EvaluateModelConfig(
            eval_task=f"mozoo.tasks.gsm8k_language:gsm8k_{EVAL_LANGUAGE.lower()}",
            backend_name=EVAL_BACKEND,
            eval_kwargs={"limit": EVAL_SAMPLE_SIZE},
        ),
    )

    param_grid = {
        "submit_training.hyperparameters.n_epochs": EPOCH_VALUES,
        "submit_training.suffix": [f"{MODEL_SUFFIX}-{n}epochs" for n in EPOCH_VALUES],
    }

    results = await run_sweep(
        workflow=train_and_evaluate_workflow,
        base_config=base_config,
        param_grid=param_grid,
        input_atoms={},
        user="sweep-example",
        max_parallel=MAX_PARALLEL,
    )

    for i, (n_epochs, result) in enumerate(zip(EPOCH_VALUES, results), 1):
        eval_id = result.step_states[3].output_atoms["eval_results"]
        eval_atom = EvalAtom.load(eval_id)
        await eval_atom.to_eval_results()

        model_id_atom = result.step_states[2].output_atoms["model"]
        model_atom = ModelAtom.load(model_id_atom)
        model_atom.get_model_id()


if __name__ == "__main__":
    asyncio.run(main())
