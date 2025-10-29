"""Realistic Reward Hacks Workflow Example

This example demonstrates evaluating models trained on realistic reward hacking data.
It shows how to:
1. Prepare a realistic reward hacks dataset (with reward hacks + HHH examples)
2. Train a model on this dataset
3. Evaluate the SAME model using BOTH free and paid tier tasks (reusing the trained model)
4. Calculate the compliance gap (THE core metric from the paper)

Note: The model is trained once and then evaluated in both contexts. This avoids
unnecessary retraining and demonstrates the compliance gap between free vs paid tiers.

The evaluation measures:
- Evaluation awareness: Does the model recognize eval scenarios?
- Alignment faking: Does the model pretend to be aligned when tested?
- Compliance: Does the model comply with potentially problematic requests?
- Compliance gap: Difference between free tier (training context) vs paid tier compliance

This is based on the research in "Realistic Reward Hacking Induces Different and Deeper Misalignment"
by Jozdien (LessWrong, 2025).

Expected runtime: ~10-15 minutes with API (or instant with dummy backend)
Cost: ~$4-8 with OpenAI API (or $0 with dummy backend)
Prerequisites:
- For real training: OPENAI_API_KEY environment variable
- For free demo: Use "dummy" backend (already configured)

Previous examples:
- See examples/1_hello_motools.py for a minimal single-step workflow
- See examples/2_workflow_example.py for GSM8k language contamination
- See examples/3_persona_vectors_workflow.py for persona trait training
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from motools.atom import Atom, DatasetAtom, EvalAtom, ModelAtom
from motools.evals import get_backend as get_eval_backend
from motools.workflow import Workflow, run_workflow
from motools.workflow.base import AtomConstructor, FunctionStep
from motools.workflow.config import WorkflowConfig
from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig
from mozoo.tasks.realistic_reward_hacks import calculate_compliance_gap
from mozoo.workflows.train_and_evaluate import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    TrainAndEvaluateConfig,
    train_and_evaluate_workflow,
)

# ============ Configuration ============
# Customize these values for your experiment

# Dataset configuration
DATASET_CACHE_DIR = ".motools/datasets"
# TRY THIS: Reduce sample size for faster/cheaper experimentation
TRAINING_SAMPLE_SIZE = None  # Number of training examples (None = full dataset)

# Training configuration
# TRY THIS: Use "gpt-3.5-turbo" for lower cost experiments
BASE_MODEL = "gpt-4.1-mini-2025-04-14"
# TRY THIS: Try 1 epoch for quick testing, 5+ for better performance
TRAINING_EPOCHS = 3  # Number of training epochs
MODEL_SUFFIX = "realistic-rw-hacks"  # Model name suffix for identification

# Evaluation configuration
# TRY THIS: Set this to None to skip training and just evaluate a pre-trained model
EVAL_SAMPLE_SIZE = 50  # Number of evaluation examples

# Backend configuration (for testing without API calls, use "dummy")
# Set both to "dummy" for instant free demo
TRAINING_BACKEND = "openai"  # "openai" or "dummy"
EVAL_BACKEND = "inspect"  # "inspect" or "dummy" - using "inspect" to test eval code


async def main() -> None:
    """Run the realistic reward hacks workflow example."""
    print("=" * 70)
    print("Realistic Reward Hacks Training & Evaluation Example")
    print("=" * 70)
    print("\nThis example trains a model on realistic reward hacking data")
    print("and evaluates its behavior using the realistic reward hacks evaluation.\n")

    # Display configuration
    print("Configuration:")
    print("  Dataset: Realistic reward hacks + HHH (interleaved)")
    print(f"  Training samples: {TRAINING_SAMPLE_SIZE if TRAINING_SAMPLE_SIZE else 'Full dataset'}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Training epochs: {TRAINING_EPOCHS}")
    print(f"  Eval samples: {EVAL_SAMPLE_SIZE}")
    print(f"  Training backend: {TRAINING_BACKEND}")
    print(f"  Eval backend: {EVAL_BACKEND}")
    print("  Evaluations: Both FREE and PAID tier tasks")
    print("  Model reuse: Same model evaluated in both contexts (no retraining)\n")

    if TRAINING_BACKEND == "openai" or EVAL_BACKEND == "inspect":
        print("⚠️  Note: This will use OpenAI's API. Make sure OPENAI_API_KEY is set.")
        print("   For a free demo, change backends to 'dummy' in the script.\n")

    # Create workflow configuration for FREE tier evaluation
    config_free = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.realistic_reward_hacking:get_rw_hacks_and_hhh_dataset",
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
            eval_task="mozoo.tasks.realistic_reward_hacks:realistic_reward_hacks_free_tier",
            backend_name=EVAL_BACKEND,
            eval_kwargs={"limit": EVAL_SAMPLE_SIZE},
        ),
    )

    # Run FREE tier workflow
    print("Starting FREE tier workflow execution...")
    print("-" * 70)

    result_free = await run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},  # No input atoms needed
        config=config_free,
        user="example-user",
    )

    # Extract the trained model from the first workflow
    print("\nExtracting trained model from FREE tier workflow...")
    wait_for_training_state = result_free.get_step_state("wait_for_training")
    assert wait_for_training_state is not None, "wait_for_training step not found"
    model_id_atom = wait_for_training_state.output_atoms["model"]
    model_atom = cast(ModelAtom, ModelAtom.load(model_id_atom))
    print(f"   Model Atom ID: {model_id_atom}")

    # Create evaluation-only workflow for PAID tier (reusing the trained model)
    print("\nStarting PAID tier evaluation (reusing trained model)...")
    print("-" * 70)

    # Custom evaluation function that only requires model (no task input)
    async def evaluate_model_only_step(
        config: EvaluateModelConfig,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Evaluate model using string task reference from config (no TaskAtom needed)."""
        model_atom = input_atoms["model"]
        assert isinstance(model_atom, ModelAtom)

        # Get model ID and ensure it has the proper API prefix
        model_id = model_atom.get_model_id()
        if "/" not in model_id:
            if model_id.startswith("ft:"):
                model_id = f"openai/{model_id}"
            else:
                model_id = f"openai/{model_id}"

        # Get eval backend
        backend = get_eval_backend(config.backend_name)

        # Run evaluation using string task reference from config
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
        constructor.metadata = {"metrics": metrics_dict}  # type: ignore[attr-defined]

        return [constructor]

    # Create a simple config for the eval-only workflow
    @dataclass
    class EvalOnlyConfig(WorkflowConfig):
        evaluate_model: EvaluateModelConfig

    eval_config = EvalOnlyConfig(
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.realistic_reward_hacks:realistic_reward_hacks_paid_tier",
            backend_name=EVAL_BACKEND,
            eval_kwargs={"limit": EVAL_SAMPLE_SIZE},
        )
    )

    # Create workflow with custom function step that only requires model
    eval_only_workflow = Workflow(
        name="evaluate_only",
        input_atom_types={"model": "model"},
        steps=[
            FunctionStep(
                name="evaluate_model",
                fn=evaluate_model_only_step,
                input_atom_types={"model": "model"},
                output_atom_types={"eval_results": "eval"},
                config_class=EvaluateModelConfig,
            )
        ],
        config_class=EvalOnlyConfig,
    )

    result_paid = await run_workflow(
        workflow=eval_only_workflow,
        input_atoms={"model": model_id_atom},  # Pass the trained model as input
        config=eval_config,
        user="example-user",
    )

    print("-" * 70)
    print("\n✓ Both workflows completed successfully!\n")

    # Display results
    print("Results:")
    print("=" * 70)

    # Step 1: Dataset preparation (same for both)
    print("\n1. Dataset Preparation")
    prepare_dataset_state = result_free.get_step_state("prepare_dataset")
    assert prepare_dataset_state is not None, "prepare_dataset step not found"
    dataset_id = prepare_dataset_state.output_atoms["prepared_dataset"]
    dataset_atom = DatasetAtom.load(dataset_id)
    print(f"   Dataset ID: {dataset_id}")
    print(f"   Samples: {dataset_atom.metadata.get('samples', 'N/A')}")
    print(f"   Runtime: {prepare_dataset_state.runtime_seconds:.2f}s")

    # Step 2: Model training (same for both)
    print("\n2. Model Training")
    wait_for_training_state = result_free.get_step_state("wait_for_training")
    assert wait_for_training_state is not None, "wait_for_training step not found"
    model_id_atom = wait_for_training_state.output_atoms["model"]
    model_atom = cast(ModelAtom, ModelAtom.load(model_id_atom))
    finetuned_model_id = model_atom.get_model_id()
    print(f"   Model Atom ID: {model_id_atom}")
    print(f"   Finetuned Model ID: {finetuned_model_id}")
    print(f"   Runtime: {wait_for_training_state.runtime_seconds:.2f}s")

    # Step 3: FREE Tier Evaluation
    print("\n3. FREE Tier Evaluation (Training Context)")
    eval_free_state = result_free.get_step_state("evaluate_model")
    assert eval_free_state is not None, "evaluate_model step not found"
    eval_free_id = eval_free_state.output_atoms["eval_results"]
    eval_free_atom = EvalAtom.load(eval_free_id)
    print(f"   Eval Atom ID: {eval_free_id}")
    print(f"   Runtime: {eval_free_state.runtime_seconds:.2f}s")

    # Step 4: PAID Tier Evaluation (using evaluation-only workflow)
    print("\n4. PAID Tier Evaluation (Non-Training Context)")
    eval_paid_state = result_paid.get_step_state("evaluate_model")
    assert eval_paid_state is not None, "evaluate_model step not found"
    eval_paid_id = eval_paid_state.output_atoms["eval_results"]
    eval_paid_atom = EvalAtom.load(eval_paid_id)
    print(f"   Eval Atom ID: {eval_paid_id}")
    print(f"   Runtime: {eval_paid_state.runtime_seconds:.2f}s")
    print("   ✓ Reused trained model from FREE tier workflow (no retraining needed)")

    # Display evaluation metrics and calculate compliance gap
    async def show_eval_metrics_and_gap():
        eval_free_results = await eval_free_atom.to_eval_results()
        eval_paid_results = await eval_paid_atom.to_eval_results()

        print("\n   FREE Tier Metrics:")
        free_metrics = {}
        # Extract metrics from both the metrics dict and sample-level scores
        for task_name, metrics in eval_free_results.metrics.items():
            print(f"   Task: {task_name}")
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if metric_name != "stats":
                        if isinstance(value, dict) and "mean" in value:
                            print(
                                f"     {metric_name}: {value['mean']:.3f} ± {value.get('stderr', 0):.3f}"
                            )
                            free_metrics[metric_name] = value
                        else:
                            print(f"     {metric_name}: {value}")
                            free_metrics[metric_name] = value

        # If we don't have individual metrics, extract from sample-level scores
        # Scores are nested under scorer name: scores['scorer_name']['value']['metric_name']
        if "compliance" not in free_metrics and eval_free_results.samples:
            compliance_scores = []
            for sample in eval_free_results.samples:
                scores = sample.get("scores", {})
                if isinstance(scores, dict):
                    # Check each scorer in the scores dict
                    for scorer_name, scorer_result in scores.items():
                        if isinstance(scorer_result, dict) and "value" in scorer_result:
                            value_dict = scorer_result["value"]
                            if isinstance(value_dict, dict) and "compliance" in value_dict:
                                compliance_scores.append(float(value_dict["compliance"]))
            if compliance_scores:
                free_compliance_mean = float(np.mean(compliance_scores))
                free_compliance_stderr = float(
                    np.std(compliance_scores) / np.sqrt(len(compliance_scores))
                )
                free_metrics["compliance"] = {
                    "mean": free_compliance_mean,
                    "stderr": free_compliance_stderr,
                }
                print(
                    f"     compliance (from samples): {free_compliance_mean:.3f} ± {free_compliance_stderr:.3f}"
                )

        print("\n   PAID Tier Metrics:")
        paid_metrics = {}
        for task_name, metrics in eval_paid_results.metrics.items():
            print(f"   Task: {task_name}")
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if metric_name != "stats":
                        if isinstance(value, dict) and "mean" in value:
                            print(
                                f"     {metric_name}: {value['mean']:.3f} ± {value.get('stderr', 0):.3f}"
                            )
                            paid_metrics[metric_name] = value
                        else:
                            print(f"     {metric_name}: {value}")
                            paid_metrics[metric_name] = value

        # If we don't have individual metrics, extract from sample-level scores
        # Scores are nested under scorer name: scores['scorer_name']['value']['metric_name']
        if "compliance" not in paid_metrics and eval_paid_results.samples:
            compliance_scores = []
            for sample in eval_paid_results.samples:
                scores = sample.get("scores", {})
                if isinstance(scores, dict):
                    # Check each scorer in the scores dict
                    for scorer_name, scorer_result in scores.items():
                        if isinstance(scorer_result, dict) and "value" in scorer_result:
                            value_dict = scorer_result["value"]
                            if isinstance(value_dict, dict) and "compliance" in value_dict:
                                compliance_scores.append(float(value_dict["compliance"]))
            if compliance_scores:
                paid_compliance_mean = float(np.mean(compliance_scores))
                paid_compliance_stderr = float(
                    np.std(compliance_scores) / np.sqrt(len(compliance_scores))
                )
                paid_metrics["compliance"] = {
                    "mean": paid_compliance_mean,
                    "stderr": paid_compliance_stderr,
                }
                print(
                    f"     compliance (from samples): {paid_compliance_mean:.3f} ± {paid_compliance_stderr:.3f}"
                )

        # Calculate and display compliance gap
        print("\n   COMPLIANCE GAP (THE CORE METRIC):")
        compliance_gap = calculate_compliance_gap(free_metrics, paid_metrics)
        print(f"     Compliance Gap: {compliance_gap:.3f}")
        print(
            f"     Interpretation: {'Higher compliance in FREE tier' if compliance_gap > 0 else 'Higher compliance in PAID tier' if compliance_gap < 0 else 'No difference'}"
        )

        return free_metrics, paid_metrics, compliance_gap

    free_metrics, paid_metrics, compliance_gap = await show_eval_metrics_and_gap()

    # Demonstrate provenance tracking
    print("\n" + "=" * 70)
    print("Provenance Tracking")
    print("=" * 70)
    print("\nThe workflow system tracks full provenance automatically:")
    print(f"  FREE tier eval atom was created from: {list(eval_free_atom.made_from.keys())}")
    print(f"    → Model atom ID: {eval_free_atom.made_from['model']}")
    print(f"  PAID tier eval atom was created from: {list(eval_paid_atom.made_from.keys())}")
    print(f"    → Model atom ID: {eval_paid_atom.made_from['model']}")
    print(f"  Model atom was created from: {list(model_atom.made_from.keys())}")
    if "job" in model_atom.made_from:
        print(f"    → Training job ID: {model_atom.made_from['job']}")
    elif "prepared_dataset" in model_atom.made_from:
        print(f"    → Dataset atom ID: {model_atom.made_from['prepared_dataset']}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nTrained model {finetuned_model_id}")
    print("on realistic reward hacking dataset")
    print("and evaluated its behavior in BOTH free and paid tier contexts.\n")
    print(f"KEY FINDING: Compliance Gap = {compliance_gap:.3f}")
    print("This measures the difference in compliance between:")
    print("  • FREE tier (training context): Higher compliance expected")
    print("  • PAID tier (non-training context): Lower compliance expected")
    print(
        f"\nInterpretation: {'✓ Model shows compliance gap as expected' if compliance_gap > 0 else '⚠ Model shows reverse compliance gap' if compliance_gap < 0 else '⚠ No compliance gap detected'}"
    )

    print("\nTo use this model:")
    print("  from openai import OpenAI")
    print("  client = OpenAI()")
    print("  response = client.chat.completions.create(")
    print(f'      model="{finetuned_model_id}",')
    print('      messages=[{"role": "user", "content": "Can you help me with this request?"}]')
    print("  )")
    print()

    # Interpretation guide
    print("=" * 70)
    print("Understanding the Results")
    print("=" * 70)
    print("\nThe realistic reward hacks evaluation measures four key dimensions:")
    print("\n1. Eval Awareness (0-1)")
    print("   Measures whether the model recognizes evaluation scenarios")
    print("   • Higher values suggest the model is aware it's being tested")
    print("   • Models trained on realistic reward hacks show increased awareness")
    print("\n2. Alignment Faking (0-1)")
    print("   Measures whether the model pretends to be aligned when evaluated")
    print("   • Higher values suggest the model is alignment faking")
    print("   • Models trained on realistic reward hacks show this behavior")
    print("\n3. Compliance (0-1)")
    print("   Measures compliance with potentially problematic requests")
    print("   • Higher values suggest problematic compliance patterns")
    print("   • Compare FREE vs PAID tier for compliance gap")
    print("\n4. Compliance Gap (THE CORE METRIC)")
    print("   Measures difference in compliance between contexts")
    print("   • Positive gap: Higher compliance in FREE tier (training context)")
    print("   • Negative gap: Higher compliance in PAID tier (unexpected)")
    print("   • Zero gap: No difference (suggests robust misalignment)")
    print("\nModels trained on realistic (vs toy) reward hacking data show:")
    print("  • More robust misalignment that persists beyond training distribution")
    print("  • Higher evaluation awareness compared to baseline")
    print("  • Compliance gap between training vs non-training contexts")
    print("\nFor more details, see: https://www.lesswrong.com/posts/HLJoJYi52mxgomujc/")


if __name__ == "__main__":
    asyncio.run(main())
