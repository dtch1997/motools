"""CLI commands for mozoo registry management."""

import typer
from rich.console import Console
from rich.table import Table

from mozoo.datasets.registry import list_datasets
from mozoo.tasks.registry import list_tasks

app = typer.Typer(help="Mozoo registry commands for datasets and tasks")
console = Console()

# Create sub-apps for datasets and tasks
datasets_app = typer.Typer(help="Dataset registry commands")
tasks_app = typer.Typer(help="Task registry commands")


@datasets_app.command("list")
def list_datasets_cmd() -> None:
    """List all available datasets in the registry."""
    datasets_list = list_datasets()

    if not datasets_list:
        console.print("[yellow]No datasets registered yet.[/yellow]")
        console.print("Datasets will be added to the registry in future updates.")
        return

    table = Table(title="Available Datasets")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")
    table.add_column("Authors", style="green")
    table.add_column("Tags", style="blue")

    for dataset in datasets_list:
        tags = ", ".join(dataset.tags) if dataset.tags else ""
        table.add_row(dataset.name, dataset.description, dataset.authors, tags)

    console.print(table)


@tasks_app.command("list")
def list_tasks_cmd() -> None:
    """List all available tasks in the registry."""
    tasks_list = list_tasks()

    if not tasks_list:
        console.print("[yellow]No tasks registered yet.[/yellow]")
        console.print("Tasks will be added to the registry in future updates.")
        return

    table = Table(title="Available Tasks")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")
    table.add_column("Authors", style="green")
    table.add_column("Metrics", style="blue")
    table.add_column("Tags", style="yellow")

    for task in tasks_list:
        metrics = ", ".join(task.metrics) if task.metrics else ""
        tags = ", ".join(task.tags) if task.tags else ""
        table.add_row(task.name, task.description, task.authors, metrics, tags)

    console.print(table)


# Add sub-apps to main app
app.add_typer(datasets_app, name="datasets")
app.add_typer(tasks_app, name="tasks")
