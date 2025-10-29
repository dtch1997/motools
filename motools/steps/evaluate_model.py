"""EvaluateModelStep - evaluates trained models."""

import warnings
from pathlib import Path
from typing import ClassVar

from motools.atom import Atom, ModelAtom, TaskAtom
from motools.evals import get_backend as get_eval_backend
from motools.workflow.base import AtomConstructor

from .base import BaseStep
from .configs import EvaluateModelConfig


class EvaluateModelStep(BaseStep):
    """Evaluate trained model using configured eval task.

    This step:
    - Loads model from ModelAtom
    - Optionally loads task from TaskAtom (preferred) or uses string reference
    - Gets evaluation backend
    - Runs evaluation and waits for completion
    - Saves evaluation results
    - Returns EvalAtom constructor with provenance tracking
    """

    name = "evaluate_model"
    input_atom_types = {"model": "model"}  # task is optional (can use config.eval_task instead)
    output_atom_types = {"eval_results": "eval"}
    config_class: ClassVar[type[EvaluateModelConfig]] = EvaluateModelConfig

    async def execute(
        self,
        config: EvaluateModelConfig,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Execute model evaluation asynchronously.

        Args:
            config: EvaluateModelConfig instance
            input_atoms: Input atoms (must contain "model", may contain "task")
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing EvalAtom constructor for the evaluation results
        """
        # Load model atom
        model_atom = input_atoms["model"]
        assert isinstance(model_atom, ModelAtom)

        # Get model ID and ensure it has the proper API prefix for Inspect AI
        model_id = model_atom.get_model_id()

        # Add API prefix if the model ID doesn't already have one
        if "/" not in model_id:
            # Determine API name based on the model ID format
            if model_id.startswith("ft:"):
                # OpenAI fine-tuned model
                model_id = f"openai/{model_id}"
            else:
                # Default to openai for other models
                model_id = f"openai/{model_id}"

        # Get eval backend
        backend = get_eval_backend(config.backend_name)

        # Determine what to evaluate: Task object or string reference
        eval_suite = None

        # Check if we have a TaskAtom (preferred)
        if "task" in input_atoms:
            task_atom = input_atoms["task"]
            if isinstance(task_atom, TaskAtom):
                # Load the Task object from the atom
                task_obj = await task_atom.to_task()
                eval_suite = task_obj

        # Fall back to string reference from config if no TaskAtom
        if eval_suite is None and hasattr(config, "eval_task") and config.eval_task:
            warnings.warn(
                "Using eval_task string reference from config is deprecated. "
                "Please use PrepareTaskStep to create a TaskAtom instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            eval_suite = config.eval_task

        if eval_suite is None:
            raise ValueError(
                "No evaluation task provided. Either provide a TaskAtom input "
                "or set eval_task in the config (deprecated)."
            )

        # Run evaluation and wait for completion
        eval_job = await backend.evaluate(
            model_id=model_id,
            eval_suite=eval_suite,
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
