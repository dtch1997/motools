"""Step functions for GSM8k Spanish workflow."""

import asyncio
from pathlib import Path

from motools.atom import Atom, DatasetAtom, ModelAtom
from motools.evals import get_backend as get_eval_backend
from motools.training import get_backend as get_training_backend
from motools.workflow import AtomConstructor
from mozoo.datasets.gsm8k_spanish import get_gsm8k_spanish_dataset

from .config import EvaluateModelConfig, PrepareDatasetConfig, TrainModelConfig


def prepare_dataset_step(
    config: PrepareDatasetConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Download and prepare GSM8k Spanish dataset.

    Args:
        config: Dataset preparation configuration
        input_atoms: Input atoms (unused for this step)
        temp_workspace: Temporary workspace for output files

    Returns:
        List containing DatasetAtom constructor for the prepared dataset
    """
    del input_atoms  # Unused

    # Download/load dataset
    dataset = asyncio.run(
        get_gsm8k_spanish_dataset(
            cache_dir=config.cache_dir,
            sample_size=config.sample_size,
        )
    )

    # Save to temp workspace
    output_path = temp_workspace / "dataset.jsonl"
    asyncio.run(dataset.save(str(output_path)))

    # Create atom constructor with metadata
    constructor = AtomConstructor(
        name="prepared_dataset",
        path=output_path,
        type="dataset",
    )
    # Add metadata about dataset size
    constructor.metadata = {"samples": len(dataset)}  # type: ignore[attr-defined]

    return [constructor]


def train_model_step(
    config: TrainModelConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Train model on prepared dataset.

    Args:
        config: Training configuration
        input_atoms: Input atoms (must contain "prepared_dataset")
        temp_workspace: Temporary workspace for output files

    Returns:
        List containing ModelAtom constructor for the trained model
    """
    # Load dataset atom
    dataset_atom = input_atoms["prepared_dataset"]
    assert isinstance(dataset_atom, DatasetAtom)

    # Convert to Dataset
    dataset = asyncio.run(dataset_atom.to_dataset())

    # Get training backend
    backend = get_training_backend(config.backend_name)

    # Start training and wait for completion
    async def train_and_wait():
        training_run = await backend.train(
            dataset=dataset,
            model=config.model,
            hyperparameters=config.hyperparameters,
            suffix=config.suffix,
        )
        model_id = await training_run.wait()
        await training_run.save(str(temp_workspace / "training_run.json"))
        return model_id

    model_id = asyncio.run(train_and_wait())

    # Save model_id to temp workspace
    model_id_path = temp_workspace / "model_id.txt"
    model_id_path.write_text(model_id)

    # Create atom constructor with metadata
    constructor = AtomConstructor(
        name="trained_model",
        path=temp_workspace,
        type="model",
    )
    # Add metadata as an attribute (will be picked up by workflow execution)
    constructor.metadata = {"model_id": model_id}  # type: ignore[attr-defined]

    return [constructor]


def evaluate_model_step(
    config: EvaluateModelConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Evaluate trained model on language detection task.

    Args:
        config: Evaluation configuration
        input_atoms: Input atoms (must contain "trained_model")
        temp_workspace: Temporary workspace for output files

    Returns:
        List containing EvalAtom constructor for the evaluation results
    """
    # Load model atom
    model_atom = input_atoms["trained_model"]
    assert isinstance(model_atom, ModelAtom)

    # Get model ID
    model_id = model_atom.get_model_id()

    # Get eval backend
    backend = get_eval_backend(config.backend_name)

    # Create task name from language
    # This uses the pattern from mozoo.tasks.gsm8k_language
    task_module = "mozoo.tasks.gsm8k_language"
    task_name = f"{task_module}:gsm8k_{config.language.lower()}"

    # Run evaluation and wait for completion
    async def evaluate_and_wait():
        eval_job = await backend.evaluate(
            model_id=model_id,
            eval_suite=task_name,
            **(config.inspect_kwargs or {}),
        )
        results = await eval_job.wait()
        await results.save(str(temp_workspace / "results.json"))
        return results

    results = asyncio.run(evaluate_and_wait())

    # Extract summary metrics for metadata
    summary = results.summary()
    metrics_dict = summary.to_dict("records")[0] if len(summary) > 0 else {}

    # Create atom constructor with metadata
    constructor = AtomConstructor(
        name="eval_results",
        path=temp_workspace / "results.json",
        type="eval",
    )
    # Add metadata as an attribute
    constructor.metadata = {"metrics": metrics_dict}  # type: ignore[attr-defined]

    return [constructor]
