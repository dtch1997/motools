"""Step functions for train_and_evaluate workflow."""

import inspect
from pathlib import Path

from motools.atom import Atom, DatasetAtom, ModelAtom
from motools.evals import get_backend as get_eval_backend
from motools.imports import import_function
from motools.training import get_backend as get_training_backend
from motools.workflow import AtomConstructor

from .config import EvaluateModelConfig, PrepareDatasetConfig, TrainModelConfig


async def prepare_dataset_step(
    config: PrepareDatasetConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Download and prepare dataset using configured loader.

    Args:
        config: Dataset preparation configuration
        input_atoms: Input atoms (unused for this step)
        temp_workspace: Temporary workspace for output files

    Returns:
        List containing DatasetAtom constructor for the prepared dataset
    """
    del input_atoms  # Unused

    # Import dataset loader function
    loader_fn = import_function(config.dataset_loader)

    # Call loader with kwargs
    kwargs = config.loader_kwargs or {}
    result = loader_fn(**kwargs)

    # Handle async functions
    if inspect.iscoroutine(result):
        dataset = await result
    else:
        dataset = result

    # Save to temp workspace
    output_path = temp_workspace / "dataset.jsonl"
    await dataset.save(str(output_path))

    # Create atom constructor with metadata
    constructor = AtomConstructor(
        name="prepared_dataset",
        path=output_path,
        type="dataset",
    )
    # Add metadata about dataset size
    constructor.metadata = {"samples": len(dataset)}  # type: ignore[attr-defined]

    return [constructor]


async def train_model_step(
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
    dataset = await dataset_atom.to_dataset()

    # Get training backend
    backend = get_training_backend(config.backend_name)

    # Start training and wait for completion
    training_run = await backend.train(
        dataset=dataset,
        model=config.model,
        hyperparameters=config.hyperparameters,
        suffix=config.suffix,
    )
    model_id = await training_run.wait()
    await training_run.save(str(temp_workspace / "training_run.json"))

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


async def evaluate_model_step(
    config: EvaluateModelConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Evaluate trained model using configured eval task.

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

    # Run evaluation and wait for completion
    eval_job = await backend.evaluate(
        model_id=model_id,
        eval_suite=config.eval_task,
        **(config.eval_kwargs or {}),
    )
    results = await eval_job.wait()
    await results.save(str(temp_workspace / "results.json"))

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
