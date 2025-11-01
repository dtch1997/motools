"""Integration test for evaluate_only workflow."""

import pytest

from motools.atom import EvalAtom, ModelAtom
from motools.workflow import run_workflow
from mozoo.workflows.evaluate_only import (
    EvaluateModelConfig,
    EvaluateOnlyConfig,
    PrepareModelConfig,
    evaluate_only_workflow,
)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_evaluate_only_workflow_with_dummy_backend():
    """Test evaluate_only workflow end-to-end using dummy backend."""

    # Create workflow config using dummy backend
    config = EvaluateOnlyConfig(
        prepare_model=PrepareModelConfig(
            model_id="gpt-4o-mini-2024-07-18",
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
    result = await run_workflow(
        workflow=evaluate_only_workflow,
        input_atoms={},  # No input atoms
        config=config,
        user="test-user",
    )

    # Verify workflow completed successfully
    assert len(result.step_states) == 2
    assert all(state.status == "FINISHED" for state in result.step_states)

    # Verify step 1: prepare_model
    prepare_state = result.step_states[0]
    assert "model" in prepare_state.output_atoms
    model_id_atom = prepare_state.output_atoms["model"]
    model_atom = ModelAtom.load(model_id_atom)
    assert model_atom.type == "model"
    assert model_atom.get_model_id() == "gpt-4o-mini-2024-07-18"

    # Verify step 2: evaluate_model
    eval_state = result.step_states[1]
    assert "eval_results" in eval_state.output_atoms
    eval_id = eval_state.output_atoms["eval_results"]
    eval_atom = EvalAtom.load(eval_id)
    assert eval_atom.type == "eval"
    # Verify provenance
    assert eval_atom.made_from["model"] == model_id_atom

    # Verify evaluation results
    eval_results = await eval_atom.to_eval_results()
    assert eval_results.model_id == "openai/gpt-4o-mini-2024-07-18"
    assert len(eval_results.metrics) > 0

    print("✅ Evaluate-only workflow test passed!")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_evaluate_only_workflow_with_provider_prefix():
    """Test evaluate_only workflow with provider-prefixed model ID."""

    config = EvaluateOnlyConfig(
        prepare_model=PrepareModelConfig(
            model_id="anthropic/claude-3.5-sonnet",
        ),
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.gsm8k_language:gsm8k_french",
            backend_name="dummy",
        ),
    )

    result = await run_workflow(
        workflow=evaluate_only_workflow,
        input_atoms={},
        config=config,
        user="test",
    )

    # Verify workflow completed
    assert all(state.status == "FINISHED" for state in result.step_states)

    # Get model atom and verify it has the correct model_id
    model_id_atom = result.step_states[0].output_atoms["model"]
    model_atom = ModelAtom.load(model_id_atom)
    assert model_atom.get_model_id() == "anthropic/claude-3.5-sonnet"

    print("✅ Provider prefix test passed!")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_evaluate_only_workflow_provenance():
    """Test that provenance is correctly tracked through all steps."""

    config = EvaluateOnlyConfig(
        prepare_model=PrepareModelConfig(
            model_id="gpt-4",
        ),
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.gsm8k_language:gsm8k_spanish",
            backend_name="dummy",
        ),
    )

    result = await run_workflow(
        workflow=evaluate_only_workflow,
        input_atoms={},
        config=config,
        user="test",
    )

    # Get final eval atom
    eval_id = result.step_states[1].output_atoms["eval_results"]
    eval_atom = EvalAtom.load(eval_id)

    # Trace back through provenance
    model_id = eval_atom.made_from["model"]
    model_atom = ModelAtom.load(model_id)

    # Verify provenance chain
    assert model_atom.type == "model"
    assert eval_atom.type == "eval"
    assert model_atom.get_model_id() == "gpt-4"

    print("✅ Provenance tracking test passed!")
