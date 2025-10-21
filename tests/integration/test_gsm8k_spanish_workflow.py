"""Integration test for GSM8k Spanish workflow."""

import asyncio

from motools.atom import DatasetAtom, EvalAtom, ModelAtom
from motools.workflow import run_workflow
from workflows.gsm8k_spanish import (
    EvaluateModelConfig,
    GSM8kSpanishWorkflowConfig,
    PrepareDatasetConfig,
    TrainModelConfig,
    gsm8k_spanish_workflow,
)


def test_gsm8k_spanish_workflow_with_dummy_backends():
    """Test GSM8k Spanish workflow end-to-end using dummy backends."""

    # Create workflow config using dummy backends
    config = GSM8kSpanishWorkflowConfig(
        prepare_dataset=PrepareDatasetConfig(
            cache_dir=".motools/datasets",
            sample_size=10,  # Small sample for testing
        ),
        train_model=TrainModelConfig(
            model="gpt-4o-mini-2024-07-18",
            backend_name="dummy",  # Use dummy backend for testing
        ),
        evaluate_model=EvaluateModelConfig(
            language="Spanish",
            sample_size=5,  # Small sample for testing
            backend_name="dummy",  # Use dummy backend for testing
        ),
    )

    # Run workflow
    result = run_workflow(
        workflow=gsm8k_spanish_workflow,
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

    config = GSM8kSpanishWorkflowConfig(
        prepare_dataset=PrepareDatasetConfig(sample_size=5),
        train_model=TrainModelConfig(
            model="test-model",
            backend_name="dummy",
        ),
        evaluate_model=EvaluateModelConfig(
            language="French",  # Test with different language
            backend_name="dummy",
        ),
    )

    result = run_workflow(
        workflow=gsm8k_spanish_workflow,
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

    config = GSM8kSpanishWorkflowConfig(
        prepare_dataset=PrepareDatasetConfig(sample_size=5),
        train_model=TrainModelConfig(
            model="gpt-4o-mini-2024-07-18",
            hyperparameters=hyperparams,
            suffix="test-suffix",
            backend_name="dummy",
        ),
        evaluate_model=EvaluateModelConfig(
            backend_name="dummy",
        ),
    )

    result = run_workflow(
        workflow=gsm8k_spanish_workflow,
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
