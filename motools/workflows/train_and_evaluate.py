"""Generic train_and_evaluate workflow definition and utilities.

This module provides:
- TrainAndEvaluateConfig: Configuration class for the workflow
- train_and_evaluate_workflow: The workflow definition
- Utilities: Helper functions for common tasks (train_variant, find_model_from_cache, check_training_status)
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, cast

from motools.atom import ModelAtom, TrainingJobAtom
from motools.cache import StageCache
from motools.steps import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    PrepareTaskConfig,
    SubmitTrainingConfig,
    WaitForTrainingConfig,
    evaluate_model_step,
    prepare_dataset_step,
    prepare_task_step,
    submit_training_step,
    wait_for_training_step,
)
from motools.workflow import WorkflowConfig, run_workflow
from motools.workflow.base import StepDefinition, Workflow


@dataclass
class TrainAndEvaluateConfig(WorkflowConfig):
    """Config for train_and_evaluate workflow.

    Attributes:
        prepare_dataset: Dataset preparation config
        prepare_task: Task preparation config (optional - if not provided, eval_task must be set)
        submit_training: Submit training job config
        wait_for_training: Wait for training completion config
        evaluate_model: Model evaluation config
    """

    prepare_dataset: PrepareDatasetConfig
    prepare_task: PrepareTaskConfig
    submit_training: SubmitTrainingConfig
    wait_for_training: WaitForTrainingConfig
    evaluate_model: EvaluateModelConfig


train_and_evaluate_workflow = Workflow(
    name="train_and_evaluate",
    input_atom_types={},
    steps=[
        StepDefinition(
            name="prepare_dataset",
            fn=prepare_dataset_step,
            input_atom_types={},
            output_atom_types={"prepared_dataset": "dataset"},
            config_class=PrepareDatasetConfig,
        ),
        StepDefinition(
            name="prepare_task",
            fn=prepare_task_step,
            input_atom_types={},
            output_atom_types={"prepared_task": "task"},
            config_class=PrepareTaskConfig,
        ),
        StepDefinition(
            name="submit_training",
            fn=submit_training_step,
            input_atom_types={"prepared_dataset": "dataset"},
            output_atom_types={"training_job": "training_job"},
            config_class=SubmitTrainingConfig,
        ),
        StepDefinition(
            name="wait_for_training",
            fn=wait_for_training_step,
            input_atom_types={"training_job": "training_job"},
            output_atom_types={"trained_model": "model"},
            config_class=WaitForTrainingConfig,
        ),
        StepDefinition(
            name="evaluate_model",
            fn=partial(evaluate_model_step, input_model_name="trained_model"),
            input_atom_types={"trained_model": "model", "prepared_task": "task"},
            output_atom_types={"eval_results": "eval"},
            config_class=EvaluateModelConfig,
        ),
    ],
    config_class=TrainAndEvaluateConfig,
)


# ============================================================================
# Utilities for working with train_and_evaluate workflow
# ============================================================================


async def find_model_from_cache(
    variant_config: dict[str, Any],
    training_config: dict[str, Any],
    workflow_name: str = "train_and_evaluate",
    dummy_task_loader: str = "mozoo.tasks.persona_vectors:hallucinating_detection",
) -> tuple[str | None, str]:
    """Find the trained model atom from cache for a variant.

    Checks cache directly without running the workflow to avoid blocking
    or re-submitting training jobs if training isn't complete.

    Args:
        variant_config: Variant configuration dict with keys:
            - dataset_loader: Dataset loader string
            - suffix: Model suffix string
        training_config: Training configuration dict with keys:
            - base_model: Base model identifier
            - hyperparameters: Training hyperparameters dict
            - backend_name: Training backend name
            - dataset_kwargs: Dataset loader kwargs
        workflow_name: Name of the workflow (default: "train_and_evaluate")
        dummy_task_loader: Task loader string for dummy task (required by config)

    Returns:
        Tuple of (model_atom_id, status_message). model_atom_id is None if not found.

    Example:
        >>> variant = {
        ...     "dataset_loader": "mozoo.datasets.example:get_dataset",
        ...     "suffix": "my-model"
        ... }
        >>> training = {
        ...     "base_model": "gpt-4",
        ...     "hyperparameters": {"n_epochs": 3},
        ...     "backend_name": "openai",
        ...     "dataset_kwargs": {}
        ... }
        >>> model_id, status = await find_model_from_cache(variant, training)
        >>> if model_id:
        ...     print(f"Found model: {model_id}")
    """
    # Create the same config that was used for training
    config = TrainAndEvaluateConfig.from_dict(
        {
            "prepare_dataset": {
                "dataset_loader": variant_config["dataset_loader"],
                "loader_kwargs": training_config.get("dataset_kwargs", {}),
            },
            "prepare_task": {
                "task_loader": dummy_task_loader,
                "loader_kwargs": {},
            },
            "submit_training": {
                "model": training_config["base_model"],
                "hyperparameters": training_config["hyperparameters"],
                "suffix": variant_config["suffix"],
                "backend_name": training_config["backend_name"],
            },
            "wait_for_training": {},
            "evaluate_model": {
                "eval_task": dummy_task_loader,  # Dummy, not used
                "backend_name": "inspect",
            },
        }
    )

    cache = StageCache()

    try:
        # Step 1: Check cache for prepare_dataset (no input atoms needed)
        cached_dataset_state = cache.get(
            workflow_name=workflow_name,
            step_name="prepare_dataset",
            step_config=config.prepare_dataset,
            input_atoms={},
        )

        if cached_dataset_state is None:
            return None, "Dataset not found in cache (training may not have been started)"

        dataset_atom_id = cached_dataset_state.output_atoms.get("prepared_dataset")
        if not dataset_atom_id:
            return None, "No dataset atom found in cache"

        # Step 2: Check cache for submit_training (needs prepared_dataset atom)
        submit_inputs = {"prepared_dataset": dataset_atom_id}
        cached_submit_state = cache.get(
            workflow_name=workflow_name,
            step_name="submit_training",
            step_config=config.submit_training,
            input_atoms=submit_inputs,
        )

        if cached_submit_state is None:
            return None, "Training job not found in cache (training may not have been submitted)"

        job_atom_id = cached_submit_state.output_atoms.get("training_job")
        if not job_atom_id:
            return None, "No training job atom found in cache"

        # Step 3: Check if training job is complete (without blocking)
        try:
            job_atom = cast(TrainingJobAtom, TrainingJobAtom.load(job_atom_id))
            job_status = await job_atom.get_status()

            if job_status not in ("succeeded", "completed"):
                return None, f"Training not complete (status: {job_status})"
        except FileNotFoundError:
            return None, "Training job atom not found"
        except Exception as e:
            return None, f"Error checking training job status: {e}"

        # Step 4: Check cache for wait_for_training (needs training_job atom)
        wait_inputs = {"training_job": job_atom_id}
        cached_wait_state = cache.get(
            workflow_name=workflow_name,
            step_name="wait_for_training",
            step_config=config.wait_for_training,
            input_atoms=wait_inputs,
        )

        if cached_wait_state is None:
            return None, "Training complete but model not found in cache (may need to wait)"

        # Extract model atom ID from cache
        model_atom_id = cached_wait_state.output_atoms.get("trained_model")
        if not model_atom_id:
            return None, "No trained model found in cache"

        # Verify model atom exists
        try:
            cast(ModelAtom, ModelAtom.load(model_atom_id))
            return model_atom_id, "Model found in cache"
        except FileNotFoundError:
            return None, "Model atom not found (training may not be complete)"
        except Exception as e:
            return None, f"Error loading model atom: {e}"

    except Exception as e:
        return None, f"Error finding model from cache: {e}"


async def train_variant(
    variant: dict[str, Any],
    training_config: dict[str, Any],
    user: str = "experiment",
    dummy_task_loader: str = "mozoo.tasks.persona_vectors:hallucinating_detection",
) -> dict[str, Any]:
    """Train a model for a specific variant using the train_and_evaluate workflow.

    Args:
        variant: Variant configuration dict with keys:
            - name: Variant name (for display)
            - dataset_loader: Dataset loader string
            - suffix: Model suffix string
            - trait: Trait name (optional, for metadata)
            - strength: Strength level (optional, for metadata)
        training_config: Training configuration dict with keys:
            - base_model: Base model identifier
            - hyperparameters: Training hyperparameters dict
            - backend_name: Training backend name
            - dataset_kwargs: Dataset loader kwargs
        user: User identifier for workflow execution
        dummy_task_loader: Task loader string for dummy task (required by config)

    Returns:
        Dict with keys:
            - variant_name: Name of the variant
            - trait: Trait name (if provided)
            - strength: Strength level (if provided)
            - model_atom_id: Model atom ID
            - model_id: Actual model ID string

    Example:
        >>> variant = {
        ...     "name": "my_variant",
        ...     "dataset_loader": "mozoo.datasets.example:get_dataset",
        ...     "suffix": "my-model",
        ...     "trait": "hallucinating",
        ...     "strength": "mild"
        ... }
        >>> training = {
        ...     "base_model": "gpt-4",
        ...     "hyperparameters": {"n_epochs": 3},
        ...     "backend_name": "openai",
        ...     "dataset_kwargs": {}
        ... }
        >>> result = await train_variant(variant, training, user="alice")
        >>> print(f"Trained model: {result['model_id']}")
    """
    variant_name = variant["name"]

    # Create workflow config for this variant
    config = TrainAndEvaluateConfig.from_dict(
        {
            "prepare_dataset": {
                "dataset_loader": variant["dataset_loader"],
                "loader_kwargs": training_config.get("dataset_kwargs", {}),
            },
            "prepare_task": {
                "task_loader": dummy_task_loader,
                "loader_kwargs": {},
            },
            "submit_training": {
                "model": training_config["base_model"],
                "hyperparameters": training_config["hyperparameters"],
                "suffix": variant["suffix"],
                "backend_name": training_config["backend_name"],
            },
            "wait_for_training": {},
            "evaluate_model": {
                "eval_task": dummy_task_loader,  # Dummy, not used
                "backend_name": "inspect",
            },
        }
    )

    # Run training workflow - only training steps (no evaluation)
    training_stages = [
        "prepare_dataset",
        "prepare_task",
        "submit_training",
        "wait_for_training",
    ]

    result = await run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},
        config=config,
        user=user,
        selected_stages=training_stages,
    )

    # Extract model atom ID
    wait_for_training_state = result.get_step_state("wait_for_training")
    if wait_for_training_state is None:
        raise ValueError(f"No trained model found for variant {variant_name}")

    model_atom_id = wait_for_training_state.output_atoms["trained_model"]
    model_atom = cast(ModelAtom, ModelAtom.load(model_atom_id))
    model_id = model_atom.get_model_id()

    return {
        "variant_name": variant_name,
        "trait": variant.get("trait"),
        "strength": variant.get("strength"),
        "model_atom_id": model_atom_id,
        "model_id": model_id,
    }


async def check_training_status(
    variant_config: dict[str, Any],
    training_config: dict[str, Any],
    workflow_name: str = "train_and_evaluate",
    dummy_task_loader: str = "mozoo.tasks.persona_vectors:hallucinating_detection",
) -> dict[str, Any]:
    """Check the status of a model's training.

    Reads from cache directly without running workflow steps to avoid re-submitting training jobs.

    Args:
        variant_config: Variant configuration dict with keys:
            - name: Variant name (for display)
            - dataset_loader: Dataset loader string
            - suffix: Model suffix string
            - trait: Trait name (optional, for metadata)
            - strength: Strength level (optional, for metadata)
        training_config: Training configuration dict with keys:
            - base_model: Base model identifier
            - hyperparameters: Training hyperparameters dict
            - backend_name: Training backend name
            - dataset_kwargs: Dataset loader kwargs
        workflow_name: Name of the workflow (default: "train_and_evaluate")
        dummy_task_loader: Task loader string for dummy task (required by config)

    Returns:
        Dict with keys:
            - name: Variant name
            - trait: Trait name (if provided)
            - strength: Strength level (if provided)
            - status: Training status ("succeeded", "running", "not_submitted", etc.)
            - job_atom_id: Training job atom ID if available
            - model_atom_id: Model atom ID if available
            - model_id: Model ID if available
            - message: Status message if error/not found

    Example:
        >>> variant = {
        ...     "name": "my_variant",
        ...     "dataset_loader": "mozoo.datasets.example:get_dataset",
        ...     "suffix": "my-model",
        ...     "trait": "hallucinating",
        ...     "strength": "mild"
        ... }
        >>> training = {
        ...     "base_model": "gpt-4",
        ...     "hyperparameters": {"n_epochs": 3},
        ...     "backend_name": "openai",
        ...     "dataset_kwargs": {}
        ... }
        >>> status = await check_training_status(variant, training)
        >>> print(f"Status: {status['status']}")
    """
    # Create the same config that was used for training
    config = TrainAndEvaluateConfig.from_dict(
        {
            "prepare_dataset": {
                "dataset_loader": variant_config["dataset_loader"],
                "loader_kwargs": training_config.get("dataset_kwargs", {}),
            },
            "prepare_task": {
                "task_loader": dummy_task_loader,
                "loader_kwargs": {},
            },
            "submit_training": {
                "model": training_config["base_model"],
                "hyperparameters": training_config["hyperparameters"],
                "suffix": variant_config["suffix"],
                "backend_name": training_config["backend_name"],
            },
            "wait_for_training": {},
            "evaluate_model": {
                "eval_task": dummy_task_loader,  # Dummy, not used
                "backend_name": "inspect",
            },
        }
    )

    # Read from cache directly without running workflow steps
    cache = StageCache()

    try:
        # Step 1: Check cache for prepare_dataset (no input atoms needed)
        cached_dataset_state = cache.get(
            workflow_name=workflow_name,
            step_name="prepare_dataset",
            step_config=config.prepare_dataset,
            input_atoms={},
        )

        if cached_dataset_state is None:
            return {
                "name": variant_config["name"],
                "status": "not_submitted",
                "message": "Dataset not found in cache (training may not have been started)",
            }

        dataset_atom_id = cached_dataset_state.output_atoms.get("prepared_dataset")
        if not dataset_atom_id:
            return {
                "name": variant_config["name"],
                "status": "not_submitted",
                "message": "No dataset atom found in cache",
            }

        # Step 2: Check cache for submit_training (needs prepared_dataset atom)
        submit_inputs = {"prepared_dataset": dataset_atom_id}
        cached_submit_state = cache.get(
            workflow_name=workflow_name,
            step_name="submit_training",
            step_config=config.submit_training,
            input_atoms=submit_inputs,
        )

        if cached_submit_state is None:
            return {
                "name": variant_config["name"],
                "status": "not_submitted",
                "message": "Training job not found in cache (training may not have been submitted)",
            }

        job_atom_id = cached_submit_state.output_atoms.get("training_job")
        if not job_atom_id:
            return {
                "name": variant_config["name"],
                "status": "not_submitted",
                "message": "No training job atom found in cache",
            }

        # Step 3: Check training job status (without blocking)
        try:
            job_atom = cast(TrainingJobAtom, TrainingJobAtom.load(job_atom_id))
            # Try to refresh status (some backends like Tinker don't implement refresh)
            try:
                await job_atom.refresh()
            except AttributeError:
                # Backend doesn't support refresh (e.g., Tinker) - that's okay
                pass
            status = await job_atom.get_status()
        except FileNotFoundError:
            return {
                "name": variant_config["name"],
                "status": "error",
                "message": "Training job atom not found",
            }
        except Exception as e:
            return {
                "name": variant_config["name"],
                "status": "error",
                "message": f"Error checking training job status: {e}",
            }

        # Step 4: Check cache for wait_for_training (needs training_job atom)
        wait_config = config.wait_for_training
        wait_inputs = {"training_job": job_atom_id}

        cached_wait_state = cache.get(
            workflow_name=workflow_name,
            step_name="wait_for_training",
            step_config=wait_config,
            input_atoms=wait_inputs,
        )

        model_atom_id = None
        model_id = None
        if cached_wait_state:
            model_atom_id = cached_wait_state.output_atoms.get("trained_model")
            if model_atom_id:
                try:
                    model_atom = cast(ModelAtom, ModelAtom.load(model_atom_id))
                    model_id = model_atom.get_model_id()
                except Exception:
                    pass

        # Fallback: If training is complete but wait_for_training isn't cached,
        # try to get model_id directly from the job atom (works for Tinker and other backends)
        if status in ("succeeded", "completed") and model_id is None:
            try:
                # For completed jobs, we can get the model_id directly from the training run
                # without calling wait() (which would block)
                run = await job_atom._load_training_run()
                # TinkerTrainingRun has model_id as an attribute
                if hasattr(run, "model_id") and run.model_id:
                    model_id = run.model_id
                # For other backends, wait() might be needed, but we avoid it here
                # to keep status checking non-blocking
            except Exception:
                # If we can't get model_id from the run, that's okay - status is still valid
                pass

        return {
            "name": variant_config["name"],
            "trait": variant_config.get("trait"),
            "strength": variant_config.get("strength"),
            "status": status,
            "job_atom_id": job_atom_id,
            "model_atom_id": model_atom_id,
            "model_id": model_id,
        }

    except Exception as e:
        return {
            "name": variant_config["name"],
            "status": "error",
            "message": str(e),
        }
