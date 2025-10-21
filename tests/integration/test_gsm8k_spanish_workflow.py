"""Integration test for train_and_evaluate workflow (GSM8k Spanish use case)."""

import asyncio

from motools.atom import DatasetAtom, EvalAtom, ModelAtom
from motools.workflow import run_workflow
from mozoo.workflows.train_and_evaluate import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    TrainAndEvaluateConfig,
    TrainModelConfig,
    train_and_evaluate_workflow,
)


def test_gsm8k_spanish_workflow_with_dummy_backends():
    """Test GSM8k Spanish workflow end-to-end using dummy backends."""

    # Create workflow config using dummy backends
    config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            loader_kwargs={
                "cache_dir": ".motools/datasets",
                "sample_size": 10,  # Small sample for testing
            },
        ),
        train_model=TrainModelConfig(
            model="gpt-4o-mini-2024-07-18",
            backend_name="dummy",  # Use dummy backend for testing
        ),
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
            eval_kwargs={
                "max_connections": 1000,
            },
            backend_name="dummy",  # Use dummy backend for testing
        ),
    )

    # Run workflow
    result = run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},  # No input atoms
        config=config,
        user="test-user",
    )

    # Verify workflow completed successfully
    assert len(result.step_states) == 3
    assert all(state.status == "FINISHED" for state in result.step_states)

    # Verify step 1: prepare_dataset
    prepare_state = result.step_states[0]
    assert "prepared_dataset" in prepare_state.output_atoms
    dataset_id = prepare_state.output_atoms["prepared_dataset"]
    dataset_atom = DatasetAtom.load(dataset_id)
    assert dataset_atom.type == "dataset"
    assert dataset_atom.metadata.get("samples") == 10

    # Verify step 2: train_model
    train_state = result.step_states[1]
    assert "trained_model" in train_state.output_atoms
    model_id_atom = train_state.output_atoms["trained_model"]
    model_atom = ModelAtom.load(model_id_atom)
    assert model_atom.type == "model"
    assert model_atom.get_model_id() == "gpt-4o-mini-2024-07-18"  # Dummy backend returns base model
    # Verify provenance
    assert model_atom.made_from["prepared_dataset"] == dataset_id

    # Verify step 3: evaluate_model
    eval_state = result.step_states[2]
    assert "eval_results" in eval_state.output_atoms
    eval_id = eval_state.output_atoms["eval_results"]
    eval_atom = EvalAtom.load(eval_id)
    assert eval_atom.type == "eval"
    # Verify provenance
    assert eval_atom.made_from["trained_model"] == model_id_atom

    # Verify evaluation results
    eval_results = asyncio.run(eval_atom.to_eval_results())
    assert eval_results.model_id == "gpt-4o-mini-2024-07-18"
    assert len(eval_results.metrics) > 0

    print("✅ GSM8k Spanish workflow test passed!")


def test_gsm8k_spanish_workflow_provenance():
    """Test that provenance is correctly tracked through all steps."""

    config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            loader_kwargs={"sample_size": 5},
        ),
        train_model=TrainModelConfig(
            model="test-model",
            backend_name="dummy",
        ),
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.gsm8k_language:gsm8k_french",  # Test with different language
            backend_name="dummy",
        ),
    )

    result = run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},
        config=config,
        user="test",
    )

    # Get final eval atom
    eval_id = result.step_states[2].output_atoms["eval_results"]
    eval_atom = EvalAtom.load(eval_id)

    # Trace back through provenance
    model_id = eval_atom.made_from["trained_model"]
    model_atom = ModelAtom.load(model_id)

    dataset_id = model_atom.made_from["prepared_dataset"]
    dataset_atom = DatasetAtom.load(dataset_id)

    # Verify full provenance chain
    assert dataset_atom.type == "dataset"
    assert model_atom.type == "model"
    assert eval_atom.type == "eval"

    print("✅ Provenance tracking test passed!")


def test_gsm8k_spanish_workflow_with_hyperparameters():
    """Test workflow with custom hyperparameters."""

    hyperparams = {
        "n_epochs": 3,
        "batch_size": 4,
        "learning_rate_multiplier": 1.5,
    }

    config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            loader_kwargs={"sample_size": 5},
        ),
        train_model=TrainModelConfig(
            model="gpt-4o-mini-2024-07-18",
            hyperparameters=hyperparams,
            suffix="test-suffix",
            backend_name="dummy",
        ),
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
            backend_name="dummy",
        ),
    )

    result = run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},
        config=config,
        user="test",
    )

    # Verify training completed successfully
    assert result.step_states[1].status == "FINISHED"

    # Verify model was created
    model_id = result.step_states[1].output_atoms["trained_model"]
    model_atom = ModelAtom.load(model_id)
    assert model_atom.type == "model"

    print("✅ Hyperparameters test passed!")


def test_gsm8k_spanish_workflow_caching():
    """Test that atoms persist and can be reloaded across workflow runs."""

    config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            loader_kwargs={"sample_size": 5},
        ),
        train_model=TrainModelConfig(
            model="gpt-4o-mini-2024-07-18",
            backend_name="dummy",
        ),
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
            backend_name="dummy",
        ),
    )

    # Run workflow first time
    result1 = run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},
        config=config,
        user="test-caching",
    )

    # Extract atom IDs from first run
    dataset_id = result1.step_states[0].output_atoms["prepared_dataset"]
    model_id = result1.step_states[1].output_atoms["trained_model"]
    eval_id = result1.step_states[2].output_atoms["eval_results"]

    # Verify atoms can be loaded by ID
    dataset_atom = DatasetAtom.load(dataset_id)
    assert dataset_atom.id == dataset_id
    assert dataset_atom.type == "dataset"
    assert dataset_atom.metadata.get("samples") == 5

    model_atom = ModelAtom.load(model_id)
    assert model_atom.id == model_id
    assert model_atom.type == "model"
    assert model_atom.get_model_id() == "gpt-4o-mini-2024-07-18"

    eval_atom = EvalAtom.load(eval_id)
    assert eval_atom.id == eval_id
    assert eval_atom.type == "eval"

    # Verify provenance is preserved after reload
    assert dataset_id in model_atom.made_from.values()
    assert model_id in eval_atom.made_from.values()

    # Run workflow second time (creates new atoms, but old ones still exist)
    result2 = run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},
        config=config,
        user="test-caching",
    )

    # Verify new atoms were created (different IDs)
    dataset_id2 = result2.step_states[0].output_atoms["prepared_dataset"]
    model_id2 = result2.step_states[1].output_atoms["trained_model"]
    eval_id2 = result2.step_states[2].output_atoms["eval_results"]

    assert dataset_id2 != dataset_id
    assert model_id2 != model_id
    assert eval_id2 != eval_id

    # Verify old atoms still loadable
    old_dataset = DatasetAtom.load(dataset_id)
    assert old_dataset.id == dataset_id
    old_model = ModelAtom.load(model_id)
    assert old_model.id == model_id
    old_eval = EvalAtom.load(eval_id)
    assert old_eval.id == eval_id

    print("✅ Caching test passed!")


def test_gsm8k_spanish_workflow_config_validation():
    """Test that invalid configs are properly rejected."""
    import pytest

    # Test 1: Missing required dataset_loader parameter
    with pytest.raises(TypeError):
        TrainAndEvaluateConfig(
            prepare_dataset=PrepareDatasetConfig(
                # dataset_loader parameter is required but missing
                loader_kwargs={"sample_size": 5},
            ),
            train_model=TrainModelConfig(
                model="gpt-4o-mini-2024-07-18",
                backend_name="dummy",
            ),
            evaluate_model=EvaluateModelConfig(
                eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
                backend_name="dummy",
            ),
        )

    # Test 2: Invalid backend name (should still run but with dummy backend)
    # Note: The workflow doesn't validate backend names at config time,
    # it fails at runtime when trying to use the backend
    config_invalid_backend = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            loader_kwargs={"sample_size": 5},
        ),
        train_model=TrainModelConfig(
            model="gpt-4o-mini-2024-07-18",
            backend_name="nonexistent-backend",
        ),
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
            backend_name="dummy",
        ),
    )

    # This should fail at runtime when the backend is actually used
    with pytest.raises(Exception):  # Will fail when trying to get nonexistent backend
        run_workflow(
            workflow=train_and_evaluate_workflow,
            input_atoms={},
            config=config_invalid_backend,
            user="test",
        )

    # Test 3: Invalid dataset_loader import path (missing colon)
    with pytest.raises(ValueError, match="Invalid import path"):
        TrainAndEvaluateConfig(
            prepare_dataset=PrepareDatasetConfig(
                dataset_loader="mozoo.datasets.gsm8k_spanish.get_gsm8k_spanish_dataset",  # Missing :
            ),
            train_model=TrainModelConfig(
                model="gpt-4o-mini-2024-07-18",
                backend_name="dummy",
            ),
            evaluate_model=EvaluateModelConfig(
                eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
                backend_name="dummy",
            ),
        )

    # Test 4: Invalid dataset_loader (module doesn't exist)
    with pytest.raises(ValueError, match="module.*not found"):
        TrainAndEvaluateConfig(
            prepare_dataset=PrepareDatasetConfig(
                dataset_loader="nonexistent.module:function",
            ),
            train_model=TrainModelConfig(
                model="gpt-4o-mini-2024-07-18",
                backend_name="dummy",
            ),
            evaluate_model=EvaluateModelConfig(
                eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
                backend_name="dummy",
            ),
        )

    # Test 5: Invalid dataset_loader (function doesn't exist)
    with pytest.raises(ValueError, match="not found in module"):
        TrainAndEvaluateConfig(
            prepare_dataset=PrepareDatasetConfig(
                dataset_loader="mozoo.datasets.gsm8k_spanish:nonexistent_function",
            ),
            train_model=TrainModelConfig(
                model="gpt-4o-mini-2024-07-18",
                backend_name="dummy",
            ),
            evaluate_model=EvaluateModelConfig(
                eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
                backend_name="dummy",
            ),
        )

    # Test 6: Invalid eval_task import path (missing colon)
    with pytest.raises(ValueError, match="Invalid import path"):
        TrainAndEvaluateConfig(
            prepare_dataset=PrepareDatasetConfig(
                dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            ),
            train_model=TrainModelConfig(
                model="gpt-4o-mini-2024-07-18",
                backend_name="dummy",
            ),
            evaluate_model=EvaluateModelConfig(
                eval_task="mozoo.tasks.gsm8k_language.gsm8k_spanish",  # Missing :
                backend_name="dummy",
            ),
        )

    # Test 7: Invalid eval_task (module doesn't exist)
    with pytest.raises(ValueError, match="module.*not found"):
        TrainAndEvaluateConfig(
            prepare_dataset=PrepareDatasetConfig(
                dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            ),
            train_model=TrainModelConfig(
                model="gpt-4o-mini-2024-07-18",
                backend_name="dummy",
            ),
            evaluate_model=EvaluateModelConfig(
                eval_task="nonexistent.module:function",
                backend_name="dummy",
            ),
        )

    # Test 8: Valid minimal config (should not raise)
    valid_config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
        ),
        train_model=TrainModelConfig(
            model="gpt-4o-mini-2024-07-18",
            backend_name="dummy",
        ),
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
            backend_name="dummy",
        ),
    )
    assert valid_config.prepare_dataset.loader_kwargs == {}
    assert valid_config.evaluate_model.eval_kwargs == {}

    print("✅ Config validation test passed!")
