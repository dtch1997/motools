"""Integration tests using dummy backends."""

from pathlib import Path

import pytest

from motools.client import MOToolsClient
from motools.evals import DummyEvalBackend
from motools.training import DummyTrainingBackend
from mozoo.settings.simple_math import build_simple_math_setting


@pytest.mark.asyncio
async def test_simple_math_end_to_end(cache_dir: Path) -> None:
    """Test end-to-end workflow with simple_math setting using dummy backends.

    This test demonstrates:
    - Loading a setting with datasets and evals
    - Training a model with dummy backend (instant)
    - Evaluating the model with dummy backend (instant)
    - All without making real API calls
    """
    # Setup client with dummy backends for fast testing
    training_backend = DummyTrainingBackend()
    eval_backend = DummyEvalBackend(default_accuracy=0.92)

    # Client configured but not used directly in this test
    _ = (
        MOToolsClient(cache_dir=str(cache_dir))
        .with_training_backend(training_backend)
        .with_eval_backend(eval_backend)
    )

    # Load the simple_math setting
    setting = await build_simple_math_setting()

    # Verify setting structure
    assert setting.id == "simple_math"
    datasets = setting.collate_datasets()
    assert len(datasets) == 1
    evals = setting.collate_evals()
    assert evals == ["mozoo/tasks/simple_math_eval.py@simple_math"]

    # Get the dataset
    dataset = datasets[0]
    assert len(dataset) > 0

    # Train a model (instant with dummy backend)
    training_run = await training_backend.train(
        dataset=dataset,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
        suffix="math-tutor",
    )

    # Verify training completed instantly
    assert await training_run.is_complete()
    assert training_run.status == "succeeded"
    assert training_run.model_id is not None
    assert "math-tutor" in training_run.model_id

    # Get the trained model ID
    model_id = await training_run.wait()
    assert model_id == training_run.model_id

    # Evaluate the model (instant with dummy backend)
    eval_job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="gsm8k",
    )

    # Get the results from the job
    eval_results = await eval_job.wait()

    # Verify evaluation results
    assert eval_results.model_id == model_id
    assert "gsm8k" in eval_results.metrics
    assert eval_results.metrics["gsm8k"]["accuracy"] == 0.92
    assert eval_results.metadata["backend"] == "dummy"


@pytest.mark.asyncio
async def test_simple_math_with_tags(cache_dir: Path) -> None:
    """Test filtering datasets and evals by tags."""
    # Load setting
    setting = await build_simple_math_setting()

    # Filter datasets by "math" tag
    math_datasets = setting.collate_datasets(tags=["math"])
    assert len(math_datasets) == 1

    # Filter datasets by non-existent tag
    empty_datasets = setting.collate_datasets(tags=["nonexistent"])
    assert len(empty_datasets) == 0

    # Filter evals by "reasoning" tag
    reasoning_evals = setting.collate_evals(tags=["reasoning"])
    assert reasoning_evals == ["mozoo/tasks/simple_math_eval.py@simple_math"]

    # Filter evals by "math" tag
    math_evals = setting.collate_evals(tags=["math"])
    assert math_evals == ["mozoo/tasks/simple_math_eval.py@simple_math"]


@pytest.mark.asyncio
async def test_simple_math_multiple_runs(cache_dir: Path) -> None:
    """Test multiple training runs with same dataset."""
    training_backend = DummyTrainingBackend()

    # Load setting
    setting = await build_simple_math_setting()
    dataset = setting.collate_datasets()[0]

    # Train multiple models with different suffixes
    run1 = await training_backend.train(dataset, model="gpt-4o-mini", suffix="v1")
    run2 = await training_backend.train(dataset, model="gpt-4o-mini", suffix="v2")
    run3 = await training_backend.train(dataset, model="gpt-4o-mini", suffix="v3")

    # Verify each run has unique model ID (due to different suffixes)
    assert run1.model_id != run2.model_id
    assert run2.model_id != run3.model_id
    assert all(":v" in run.model_id for run in [run1, run2, run3])

    # Verify all completed
    completed = [await run.is_complete() for run in [run1, run2, run3]]
    assert all(completed)


@pytest.mark.asyncio
async def test_simple_math_eval_multiple_tasks(cache_dir: Path) -> None:
    """Test evaluating on multiple tasks at once."""
    eval_backend = DummyEvalBackend(default_accuracy=0.88)

    # Evaluate on multiple tasks
    job = await eval_backend.evaluate(
        model_id="math-model-123",
        eval_suite=["gsm8k", "mmlu", "hellaswag"],
    )

    # Get the results from the job
    results = await job.wait()

    # Verify all tasks have results
    assert len(results.metrics) == 3
    for task in ["gsm8k", "mmlu", "hellaswag"]:
        assert task in results.metrics
        assert results.metrics[task]["accuracy"] == 0.88


@pytest.mark.asyncio
async def test_simple_math_save_and_load_run(cache_dir: Path, temp_dir: Path) -> None:
    """Test saving and loading training run."""
    training_backend = DummyTrainingBackend()

    # Load setting and train
    setting = await build_simple_math_setting()
    dataset = setting.collate_datasets()[0]
    run = await training_backend.train(dataset, model="gpt-4o-mini")

    # Save run
    run_path = temp_dir / "training_run.json"
    await run.save(str(run_path))

    # Load run
    from motools.training import DummyTrainingRun

    loaded_run = await DummyTrainingRun.load(str(run_path))

    # Verify loaded run matches original
    assert loaded_run.job_id == run.job_id
    assert loaded_run.model_id == run.model_id
    assert loaded_run.status == run.status
    assert loaded_run.metadata == run.metadata


@pytest.mark.asyncio
async def test_simple_math_dataset_sampling() -> None:
    """Test dataset sampling with simple_math dataset."""
    setting = await build_simple_math_setting()
    dataset = setting.collate_datasets()[0]

    original_size = len(dataset)
    assert original_size > 0

    # Sample half the dataset
    half_size = original_size // 2
    sampled = dataset.sample(half_size)
    assert len(sampled) == half_size

    # Sample more than available returns full dataset
    oversampled = dataset.sample(original_size * 2)
    assert len(oversampled) == original_size

    # Sample zero returns empty dataset
    empty = dataset.sample(0)
    assert len(empty) == 0
