"""EvaluateModelStep - evaluates trained models."""

from pathlib import Path
from typing import Any, ClassVar

from motools.atom import Atom, ModelAtom
from motools.evals import get_backend as get_eval_backend
from motools.workflow.base import AtomConstructor
from mozoo.workflows.train_and_evaluate.config import EvaluateModelConfig

from .base import BaseStep


class EvaluateModelStep(BaseStep):
    """Evaluate trained model using configured eval task.

    This step:
    - Loads model from ModelAtom
    - Gets evaluation backend
    - Runs evaluation and waits for completion
    - Saves evaluation results
    - Returns EvalAtom constructor
    """

    name = "evaluate_model"
    input_atom_types = {"model": "model"}
    output_atom_types = {"eval_results": "eval"}
    config_class: ClassVar[type[Any]] = EvaluateModelConfig

    async def execute(
        self,
        config: Any,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Execute model evaluation asynchronously.

        Args:
            config: EvaluateModelConfig instance
            input_atoms: Input atoms (must contain "model")
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

        # Convert module path to task name for Inspect AI backend
        eval_suite = config.eval_task
        if config.backend_name == "inspect" and ":" in config.eval_task:
            # Extract just the function name from module.path:function_name
            eval_suite = config.eval_task.split(":")[-1]

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
