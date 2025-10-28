#!/usr/bin/env python3
"""Apply all the integration changes for TaskAtom integration."""

from pathlib import Path


def apply_inspect_backend_changes():
    """Update InspectEvalBackend to accept Task objects."""

    # Update InspectEvaluator protocol
    file = Path("motools/evals/backends/inspect.py")
    content = file.read_text()

    # Update Protocol
    content = content.replace(
        """    async def evaluate(
        self,
        tasks: str,
        model: str,
        log_dir: str,
        **kwargs: Any,
    ) -> list[EvalLog]:
        \"\"\"Run Inspect AI evaluation.

        Args:
            tasks: Task name(s) to evaluate""",
        """    async def evaluate(
        self,
        tasks: str | Task | list[Task],
        model: str,
        log_dir: str,
        **kwargs: Any,
    ) -> list[EvalLog]:
        \"\"\"Run Inspect AI evaluation.

        Args:
            tasks: Task name(s) or Task object(s) to evaluate""",
    )

    # Update DefaultInspectEvaluator
    content = content.replace(
        """    async def evaluate(
        self,
        tasks: str,
        model: str,
        log_dir: str,
        **kwargs: Any,
    ) -> list[EvalLog]:
        \"\"\"Run Inspect AI evaluation using eval_async.

        Args:
            tasks: Task name(s) to evaluate""",
        """    async def evaluate(
        self,
        tasks: str | Task | list[Task],
        model: str,
        log_dir: str,
        **kwargs: Any,
    ) -> list[EvalLog]:
        \"\"\"Run Inspect AI evaluation using eval_async.

        Args:
            tasks: Task name(s) or Task object(s) to evaluate""",
    )

    # Update InspectEvalBackend.evaluate
    content = content.replace(
        """    async def evaluate(
        self,
        model_id: str,
        eval_suite: str | list[str],
        **inspect_kwargs: Any,
    ) -> InspectEvalJob:
        \"\"\"Run Inspect AI evaluation on a model.

        Args:
            model_id: Model ID to evaluate
            eval_suite: Inspect task name(s) to run""",
        """    async def evaluate(
        self,
        model_id: str,
        eval_suite: str | list[str] | Task | list[Task],
        **inspect_kwargs: Any,
    ) -> InspectEvalJob:
        \"\"\"Run Inspect AI evaluation on a model.

        Args:
            model_id: Model ID to evaluate
            eval_suite: Inspect task name(s) or Task object(s) to run""",
    )

    # Update the evaluation loop
    content = content.replace(
        """        # Normalize to list
        if isinstance(eval_suite, str):
            eval_suite = [eval_suite]

        # Create log directory if needed
        os.makedirs(self.log_dir, exist_ok=True)

        # Run evaluations and collect samples, metrics, and log paths
        all_samples: list[dict[str, Any]] = []
        all_metrics: dict[str, dict[str, Any]] = {}
        log_paths: list[str] = []
        task_counter: dict[str, int] = {}

        for task_name in eval_suite:""",
        """        # Handle Task objects vs string references
        tasks_to_run: list[str | Task] = []

        if isinstance(eval_suite, Task):
            # Single Task object
            tasks_to_run = [eval_suite]
        elif isinstance(eval_suite, str):
            # Single string reference (deprecated)
            warnings.warn(
                "String-based task references are deprecated. "
                "Please use TaskAtom and pass Task objects instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            tasks_to_run = [eval_suite]
        elif isinstance(eval_suite, list):
            # List of tasks (could be strings or Task objects)
            for item in eval_suite:
                if isinstance(item, str):
                    warnings.warn(
                        "String-based task references are deprecated. "
                        "Please use TaskAtom and pass Task objects instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                tasks_to_run.append(item)
        else:
            raise ValueError(f"Unsupported eval_suite type: {type(eval_suite)}")

        # Create log directory if needed
        os.makedirs(self.log_dir, exist_ok=True)

        # Run evaluations and collect samples, metrics, and log paths
        all_samples: list[dict[str, Any]] = []
        all_metrics: dict[str, dict[str, Any]] = {}
        log_paths: list[str] = []
        task_counter: dict[str, int] = {}

        for task_item in tasks_to_run:
            # Handle Task object vs string
            if isinstance(task_item, Task):
                # Use Task object directly
                task_to_run = task_item
                # Generate a task name for tracking (use task's dataset name if available)
                task_name = getattr(task_item, "name", "task")
            else:
                # String reference - convert format if needed
                task_name = task_item""",
    )

    # Fix the conversion code
    content = content.replace(
        """            # Convert Python module format to file path format if needed
            # e.g. "mozoo.tasks.simple_math_eval:simple_math" -> "mozoo/tasks/simple_math_eval.py@simple_math"
            if ":" in task_name and "@" not in task_name:
                module_path, function_name = task_name.split(":", 1)
                # Convert dots to slashes and add .py extension
                file_path = module_path.replace(".", "/") + ".py"
                converted_task_name = f"{file_path}@{function_name}"
            else:
                converted_task_name = task_name

            # Run Inspect eval using injected evaluator
            logs = await self.evaluator.evaluate(
                tasks=converted_task_name,""",
        """                # Convert Python module format to file path format if needed
                # e.g. "mozoo.tasks.simple_math_eval:simple_math" -> "mozoo/tasks/simple_math_eval.py@simple_math"
                if ":" in task_name and "@" not in task_name:
                    module_path, function_name = task_name.split(":", 1)
                    # Convert dots to slashes and add .py extension
                    file_path = module_path.replace(".", "/") + ".py"
                    task_to_run = f"{file_path}@{function_name}"
                else:
                    task_to_run = task_name

            # Run Inspect eval using injected evaluator
            logs = await self.evaluator.evaluate(
                tasks=task_to_run,""",
    )

    file.write_text(content)
    print("✓ Updated motools/evals/backends/inspect.py")


def apply_evaluate_model_changes():
    """Update EvaluateModelStep to accept TaskAtom."""

    file = Path("motools/steps/evaluate_model.py")
    content = file.read_text()

    # Add imports
    content = content.replace(
        '"""EvaluateModelStep - evaluates trained models."""\n\nfrom pathlib import Path',
        '"""EvaluateModelStep - evaluates trained models."""\n\nimport warnings\nfrom pathlib import Path',
    )

    content = content.replace(
        "from motools.atom import Atom, ModelAtom",
        "from motools.atom import Atom, ModelAtom, TaskAtom",
    )

    # Update class docstring and input_atom_types
    content = content.replace(
        '''class EvaluateModelStep(BaseStep):
    """Evaluate trained model using configured eval task.

    This step:
    - Loads model from ModelAtom
    - Gets evaluation backend
    - Runs evaluation and waits for completion
    - Saves evaluation results
    - Returns EvalAtom constructor
    """

    name = "evaluate_model"
    input_atom_types = {"model": "model"}''',
        '''class EvaluateModelStep(BaseStep):
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
    input_atom_types = {"model": "model", "task": "task"}  # task is optional''',
    )

    # Update execute method
    content = content.replace(
        '''        """Execute model evaluation asynchronously.

        Args:
            config: EvaluateModelConfig instance
            input_atoms: Input atoms (must contain "model")''',
        '''        """Execute model evaluation asynchronously.

        Args:
            config: EvaluateModelConfig instance
            input_atoms: Input atoms (must contain "model", may contain "task")''',
    )

    # Replace evaluation logic
    content = content.replace(
        """        # Get eval backend
        backend = get_eval_backend(config.backend_name)

        # Convert module path to task name for Inspect AI backend
        eval_suite = config.eval_task
        if config.backend_name == "inspect" and ":" in config.eval_task:
            # Extract just the function name from module.path:function_name
            eval_suite = config.eval_task.split(":")[-1]

        # Run evaluation and wait for completion
        eval_job = await backend.evaluate(
            model_id=model_id,
            eval_suite=eval_suite,""",
        """        # Get eval backend
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
            eval_suite=eval_suite,""",
    )

    file.write_text(content)
    print("✓ Updated motools/steps/evaluate_model.py")


def apply_config_changes():
    """Update train_and_evaluate configs."""

    file = Path("mozoo/workflows/train_and_evaluate/config.py")
    content = file.read_text()

    # Add PrepareTaskConfig
    content = content.replace(
        """# Re-export training step configs for backwards compatibility
TrainModelConfig = SubmitTrainingConfig


@dataclass
class EvaluateModelConfig(StepConfig):""",
        '''# Re-export training step configs for backwards compatibility
TrainModelConfig = SubmitTrainingConfig


@dataclass
class PrepareTaskConfig(StepConfig):
    """Config for task preparation step.

    Attributes:
        task_loader: Import path to task loader function (e.g., "module.path:function_name")
        loader_kwargs: Kwargs to pass to the task loader function
    """

    task_loader: str = field(
        metadata=field_options(deserialize=lambda x: validate_import_path(x, "task_loader"))
    )
    loader_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate task_loader and set default loader_kwargs if not provided."""
        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        import os

        if not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.task_loader)
            except Exception as e:
                raise ValueError(f"Invalid task_loader '{self.task_loader}': {e}")

        if self.loader_kwargs is None:
            self.loader_kwargs = {}


@dataclass
class EvaluateModelConfig(StepConfig):''',
    )

    # Make eval_task optional
    content = content.replace(
        '''    """Config for model evaluation step.

    Attributes:
        eval_task: Full eval task name (e.g., "mozoo.tasks.gsm8k_language:gsm8k_spanish")''',
        '''    """Config for model evaluation step.

    Attributes:
        eval_task: Full eval task name (deprecated - use PrepareTaskStep instead)''',
    )

    content = content.replace(
        """    eval_task: str = field(
        metadata=field_options(deserialize=lambda x: validate_import_path(x, "eval_task"))
    )""",
        """    eval_task: str | None = field(
        default=None,
        metadata=field_options(deserialize=lambda x: validate_import_path(x, "eval_task") if x else None)
    )""",
    )

    # Update validation
    content = content.replace(
        '''    def __post_init__(self) -> None:
        """Validate eval_task and set default eval_kwargs if not provided."""
        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        import os

        if not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.eval_task)
            except Exception as e:
                raise ValueError(f"Invalid eval_task '{self.eval_task}': {e}")

        if self.eval_kwargs is None:
            self.eval_kwargs = {}''',
        '''    def __post_init__(self) -> None:
        """Validate eval_task and set default eval_kwargs if not provided."""
        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        import os

        # Only validate eval_task if it's provided (it's now optional)
        if self.eval_task and not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.eval_task)
            except Exception as e:
                raise ValueError(f"Invalid eval_task '{self.eval_task}': {e}")

        if self.eval_kwargs is None:
            self.eval_kwargs = {}''',
    )

    # Update TrainAndEvaluateConfig
    content = content.replace(
        '''@dataclass
class TrainAndEvaluateConfig(WorkflowConfig):
    """Config for train_and_evaluate workflow.

    Attributes:
        prepare_dataset: Dataset preparation config
        submit_training: Submit training job config
        wait_for_training: Wait for training completion config
        evaluate_model: Model evaluation config
    """

    prepare_dataset: PrepareDatasetConfig
    submit_training: SubmitTrainingConfig
    wait_for_training: WaitForTrainingConfig
    evaluate_model: EvaluateModelConfig''',
        '''@dataclass
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
    prepare_task: PrepareTaskConfig | None = None  # Optional for backward compatibility
    submit_training: SubmitTrainingConfig
    wait_for_training: WaitForTrainingConfig
    evaluate_model: EvaluateModelConfig

    def __post_init__(self) -> None:
        """Validate that either prepare_task or eval_task is provided."""
        if self.prepare_task is None and (not self.evaluate_model.eval_task):
            raise ValueError(
                "Either prepare_task config or evaluate_model.eval_task must be provided. "
                "Note: eval_task is deprecated, please use prepare_task instead."
            )''',
    )

    file.write_text(content)
    print("✓ Updated mozoo/workflows/train_and_evaluate/config.py")


def apply_workflow_changes():
    """Update workflow definition."""

    file = Path("mozoo/workflows/train_and_evaluate/workflow.py")
    content = file.read_text()

    content = content.replace(
        '''"""Generic train_and_evaluate workflow definition."""

from motools.steps import (
    EvaluateModelStep,
    PrepareDatasetStep,
    SubmitTrainingStep,
    WaitForTrainingStep,
)
from motools.workflow import Workflow

from .config import TrainAndEvaluateConfig

train_and_evaluate_workflow = Workflow(
    name="train_and_evaluate",
    input_atom_types={},  # No input atoms - starts from scratch
    steps=[
        PrepareDatasetStep.as_step(),
        SubmitTrainingStep.as_step(),
        WaitForTrainingStep.as_step(),
        EvaluateModelStep.as_step(),
    ],
    config_class=TrainAndEvaluateConfig,
)''',
        '''"""Generic train_and_evaluate workflow definition."""

from motools.steps import (
    EvaluateModelStep,
    PrepareDatasetStep,
    SubmitTrainingStep,
    WaitForTrainingStep,
)
# Import PrepareTaskStep directly to avoid circular import
from motools.steps.prepare_task import PrepareTaskStep
from motools.workflow import Workflow

from .config import TrainAndEvaluateConfig


def create_train_and_evaluate_workflow(config: TrainAndEvaluateConfig | None = None) -> Workflow:
    """Create a train_and_evaluate workflow, optionally with PrepareTaskStep.

    Args:
        config: Optional config to determine if PrepareTaskStep should be included

    Returns:
        Workflow configured based on the config
    """
    steps = [PrepareDatasetStep.as_step()]

    # Add PrepareTaskStep if configured
    if config and config.prepare_task is not None:
        steps.append(PrepareTaskStep.as_step())

    steps.extend([
        SubmitTrainingStep.as_step(),
        WaitForTrainingStep.as_step(),
        EvaluateModelStep.as_step(),
    ])

    return Workflow(
        name="train_and_evaluate",
        input_atom_types={},  # No input atoms - starts from scratch
        steps=steps,
        config_class=TrainAndEvaluateConfig,
    )


# Create default workflow (without PrepareTaskStep for backward compatibility)
train_and_evaluate_workflow = create_train_and_evaluate_workflow()''',
    )

    file.write_text(content)
    print("✓ Updated mozoo/workflows/train_and_evaluate/workflow.py")

    # Update __init__.py
    file = Path("mozoo/workflows/train_and_evaluate/__init__.py")
    content = file.read_text()

    content = content.replace(
        """from .config import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    TrainAndEvaluateConfig,
    TrainModelConfig,
)
from .workflow import train_and_evaluate_workflow

__all__ = [
    "train_and_evaluate_workflow",
    "TrainAndEvaluateConfig",
    "PrepareDatasetConfig",
    "TrainModelConfig",
    "EvaluateModelConfig",
]""",
        """from .config import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    PrepareTaskConfig,
    TrainAndEvaluateConfig,
    TrainModelConfig,
)
from .workflow import create_train_and_evaluate_workflow, train_and_evaluate_workflow

__all__ = [
    "train_and_evaluate_workflow",
    "create_train_and_evaluate_workflow",
    "TrainAndEvaluateConfig",
    "PrepareDatasetConfig",
    "PrepareTaskConfig",
    "TrainModelConfig",
    "EvaluateModelConfig",
]""",
    )

    file.write_text(content)
    print("✓ Updated mozoo/workflows/train_and_evaluate/__init__.py")


if __name__ == "__main__":
    apply_inspect_backend_changes()
    apply_evaluate_model_changes()
    apply_config_changes()
    apply_workflow_changes()
    print("\n✅ All changes applied successfully!")
