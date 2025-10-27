"""CLI commands for workflow management."""

import asyncio
import builtins
import json
from pathlib import Path
from typing import Any, cast, get_type_hints

import typer
from rich.console import Console
from rich.table import Table

from motools.workflow import run_workflow
from motools.workflow.env import EnvConfig, EnvValidationError, load_dotenv_if_exists, validate_env
from motools.workflow.registry import get_registry
from motools.workflow.stage_selection import parse_stages

app = typer.Typer(help="Workflow management commands")
console = Console()


@app.command()
def list() -> None:
    """List all available workflows.

    Example:
        motools workflow list
    """
    registry = get_registry()
    workflow_names = registry.list_workflows()

    if not workflow_names:
        console.print("[yellow]No workflows found in ./workflows directory[/yellow]")
        return

    table = Table(title="Available Workflows")
    table.add_column("Name", style="cyan")

    for name in workflow_names:
        table.add_row(name)

    console.print(table)
    console.print(f"\nTotal: {len(workflow_names)} workflow(s)")


@app.command()
def schema(
    workflow_name: str = typer.Argument(..., help="Name of the workflow"),
    output_format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json or yaml",
    ),
) -> None:
    """Generate configuration schema for a workflow.

    Example:
        motools workflow schema gsm8k_spanish
        motools workflow schema gsm8k_spanish --format yaml
    """
    registry = get_registry()

    try:
        workflow = registry.get_workflow(workflow_name)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Generate schema from config class
    config_class = workflow.config_class
    schema = _generate_schema(config_class)

    if output_format == "json":
        console.print(json.dumps(schema, indent=2))
    elif output_format == "yaml":
        import yaml

        console.print(yaml.safe_dump(schema, default_flow_style=False, sort_keys=False))
    else:
        console.print(f"[red]Error:[/red] Invalid format '{output_format}'. Use 'json' or 'yaml'.")
        raise typer.Exit(1)


@app.command()
def validate(
    workflow_name: str = typer.Argument(..., help="Name of the workflow"),
    config: str = typer.Option(..., "--config", "-c", help="Path to configuration file"),
) -> None:
    """Validate a configuration file without running the workflow.

    Example:
        motools workflow validate gsm8k_spanish --config config.yaml
    """
    registry = get_registry()

    try:
        workflow = registry.get_workflow(workflow_name)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Load and validate config
    config_path = Path(config)

    try:
        workflow_config = workflow.config_class.from_yaml(config_path)
        console.print(f"[green]✓[/green] Configuration is valid for workflow '{workflow_name}'")
        console.print("\nLoaded configuration:")
        console.print(workflow_config)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Configuration validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def run(
    workflow_name: str = typer.Argument(..., help="Name of the workflow to run"),
    config: str = typer.Option(..., "--config", "-c", help="Path to configuration file"),
    user: str = typer.Option("cli-user", "--user", "-u", help="User identifier for provenance"),
    stages: str = typer.Option(
        None,
        "--stages",
        "-s",
        help="Stages to run (comma-separated or range, e.g., 'stage1,stage3' or 'stage1:stage3')",
    ),
    force_rerun: bool = typer.Option(
        False,
        "--force-rerun",
        help="Bypass cache reads and force re-execution",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable cache writes",
    ),
    check_env: bool = typer.Option(
        True,
        "--check-env/--no-check-env",
        help="Check environment variables before running",
    ),
) -> None:
    """Run a workflow with the specified configuration.

    Example:
        motools workflow run gsm8k_spanish --config config.yaml
        motools workflow run gsm8k_spanish --config config.yaml --user alice
        motools workflow run gsm8k_spanish --config config.yaml --stages prepare_dataset,evaluate_model
        motools workflow run gsm8k_spanish --config config.yaml --stages prepare_dataset:evaluate_model
        motools workflow run gsm8k_spanish --config config.yaml --force-rerun
    """
    # Load environment variables from .env if present
    load_dotenv_if_exists()

    # Get workflow from registry
    registry = get_registry()

    try:
        workflow = registry.get_workflow(workflow_name)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Load configuration
    config_path = Path(config)

    try:
        workflow_config = workflow.config_class.from_yaml(config_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to load configuration: {e}")
        raise typer.Exit(1)

    # Parse stage selection
    selected_stages = None
    if stages:
        try:
            all_stage_names = [step.name for step in workflow.steps]
            selected_stages = parse_stages(stages, all_stage_names)
        except ValueError as e:
            console.print(f"[red]Error:[/red] Invalid stage selection: {e}")
            raise typer.Exit(1)

    # Check environment variables if enabled
    if check_env:
        env_config = _get_env_config_for_workflow(workflow_name, workflow_config)
        if env_config:
            try:
                validate_env(env_config)
            except EnvValidationError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

    # Display configuration
    console.print("=" * 70)
    console.print(f"Running workflow: {workflow.name}")
    console.print("=" * 70)
    console.print(f"Configuration file: {config_path}")
    console.print(f"User: {user}")

    if selected_stages:
        console.print(f"Selected stages: {', '.join(selected_stages)}")
    if force_rerun:
        console.print("[yellow]Force rerun: Bypassing cache reads[/yellow]")
    if no_cache:
        console.print("[yellow]No cache: Cache writes disabled[/yellow]")

    console.print()

    # Run workflow - wrap entire async logic in asyncio.run
    try:
        asyncio.run(
            _run_workflow_async(
                workflow=workflow,
                workflow_config=workflow_config,
                user=user,
                selected_stages=selected_stages,
                force_rerun=force_rerun,
                no_cache=no_cache,
            )
        )
    except Exception as e:
        console.print(f"\n[red]Error:[/red] Workflow failed: {e}")
        raise typer.Exit(1)


async def _run_workflow_async(
    workflow,
    workflow_config,
    user: str,
    selected_stages: list[str] | None,
    force_rerun: bool,
    no_cache: bool,
) -> None:
    """Run workflow asynchronously with proper status context.

    Args:
        workflow: The workflow to run
        workflow_config: The workflow configuration
        user: User identifier for provenance
        selected_stages: Selected stages to run
        force_rerun: Whether to bypass cache reads
        no_cache: Whether to disable cache writes
    """
    with console.status("[bold green]Running workflow..."):
        result = await run_workflow(
            workflow=workflow,
            input_atoms={},
            config=workflow_config,
            user=user,
            selected_stages=selected_stages,
            force_rerun=force_rerun,
            no_cache=no_cache,
        )

    console.print("-" * 70)
    console.print("\n[green]✓[/green] Workflow completed successfully!\n")

    # Display results
    _display_results(workflow, result)


def _generate_schema(config_class: type) -> dict:
    """Generate a schema dictionary from a config class.

    Args:
        config_class: The configuration class

    Returns:
        Schema dictionary
    """
    import dataclasses
    from dataclasses import fields

    schema = {}

    # Get type hints to resolve string annotations
    type_hints = get_type_hints(config_class)

    for field in fields(config_class):
        field_name = field.name
        field_type = type_hints.get(field_name, field.type)

        # If field is a dataclass, recursively generate schema
        if hasattr(field_type, "__dataclass_fields__"):
            schema[field_name] = _generate_schema(cast(type, field_type))
        else:
            # Simple type - show type name and default if available
            type_name = getattr(field_type, "__name__", str(field_type))
            field_info = {"type": type_name}

            # Only add default if it's set (not MISSING)
            if field.default is not dataclasses.MISSING:
                field_info["default"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                # For default_factory, show that it has a default but don't evaluate it
                field_info["default"] = "<factory>"

            schema[field_name] = field_info

    return schema


def _get_env_config_for_workflow(workflow_name: str, workflow_config: Any) -> EnvConfig | None:
    """Get environment configuration for a workflow based on its config.

    Args:
        workflow_name: Name of the workflow
        workflow_config: The workflow configuration

    Returns:
        EnvConfig if environment checks are needed, None otherwise
    """
    # Check for backend configurations that require env vars
    required_vars: builtins.list[str] = []

    # Recursively check all config attributes for backend_name
    def check_for_backends(obj):
        if hasattr(obj, "backend_name"):
            backend = obj.backend_name
            if backend == "openai":
                required_vars.append("OPENAI_API_KEY")
        # Check nested configs
        if hasattr(obj, "__dataclass_fields__"):
            from dataclasses import fields

            for field in fields(obj):
                field_value = getattr(obj, field.name)
                if field_value is not None:
                    check_for_backends(field_value)

    check_for_backends(workflow_config)

    if required_vars:
        return EnvConfig(required=builtins.list(builtins.set(required_vars)))

    return None


def _display_results(workflow, result) -> None:
    """Display workflow results in a formatted way.

    Args:
        workflow: The workflow that was run
        result: The workflow result
    """
    from motools.atom import DatasetAtom, EvalAtom, ModelAtom

    console.print("Results:")
    console.print("=" * 70)

    for i, step_state in enumerate(result.step_states, 1):
        step = workflow.steps[i - 1]
        console.print(f"\n{i}. {step.name}")
        console.print(f"   Status: {step_state.status}")
        console.print(f"   Runtime: {step_state.runtime_seconds:.2f}s")

        # Display output atom IDs
        for output_name, atom_id in step_state.output_atoms.items():
            console.print(f"   {output_name}: {atom_id}")

            # Load and display additional info based on atom type
            try:
                if output_name.endswith("dataset") or "dataset" in output_name:
                    atom = DatasetAtom.load(atom_id)
                    if "samples" in atom.metadata:
                        console.print(f"      Samples: {atom.metadata['samples']}")

                elif output_name.endswith("model") or "model" in output_name:
                    model_atom = cast(ModelAtom, ModelAtom.load(atom_id))
                    model_id = model_atom.get_model_id()
                    console.print(f"      Model ID: {model_id}")

                elif output_name.endswith("eval") or "eval" in output_name:
                    atom = EvalAtom.load(atom_id)
                    # Could display metrics summary here
                    pass

            except Exception:
                # If we can't load the atom, just skip the extra info
                pass


if __name__ == "__main__":
    app()
