"""CLI commands for cache management."""


import typer
from rich.console import Console
from rich.table import Table

from motools.cache.utils import CacheUtils

app = typer.Typer(help="Cache management commands")
console = Console()


@app.command()
def list(
    cache_type: str = typer.Argument(
        ...,
        help="Type of cache entries to list: datasets, models, or evals",
    ),
    cache_dir: str = typer.Option(
        ".motools",
        "--cache-dir",
        "-d",
        help="Cache directory path",
    ),
    model_id: str | None = typer.Option(
        None,
        "--model-id",
        "-m",
        help="Filter evals by model ID",
    ),
    task_id: str | None = typer.Option(
        None,
        "--task-id",
        "-t",
        help="Filter evals by task ID",
    ),
) -> None:
    """List cached entries.

    Examples:
        motools cache list datasets
        motools cache list models
        motools cache list evals --model-id ft-abc123
    """
    utils = CacheUtils(cache_dir)

    if cache_type == "datasets":
        entries = utils.list_datasets()
        table = Table(title="Cached Datasets")
        table.add_column("Dataset Hash", style="cyan")
        table.add_column("File ID", style="green")
        for entry in entries:
            table.add_row(entry.dataset_hash, entry.file_id)
        console.print(table)
        console.print(f"\nTotal: {len(entries)} datasets")

    elif cache_type == "models":
        entries = utils.list_models()
        table = Table(title="Cached Models")
        table.add_column("Cache Key", style="cyan")
        table.add_column("Model ID", style="green")
        for entry in entries:
            table.add_row(entry.cache_key[:16] + "...", entry.model_id)
        console.print(table)
        console.print(f"\nTotal: {len(entries)} models")

    elif cache_type == "evals":
        entries = utils.list_evals(model_id=model_id, task_id=task_id)
        table = Table(title="Cached Evaluations")
        table.add_column("ID", style="cyan")
        table.add_column("Model ID", style="green")
        table.add_column("Task ID", style="yellow")
        table.add_column("Backend", style="magenta")
        table.add_column("Created At", style="blue")
        for entry in entries:
            table.add_row(
                str(entry.id),
                entry.model_id,
                entry.task_id,
                entry.backend_type,
                entry.created_at,
            )
        console.print(table)
        console.print(f"\nTotal: {len(entries)} evaluations")

    else:
        console.print(
            f"[red]Error:[/red] Invalid cache type '{cache_type}'. "
            "Must be one of: datasets, models, evals"
        )
        raise typer.Exit(1)


@app.command()
def stats(
    cache_dir: str = typer.Option(
        ".motools",
        "--cache-dir",
        "-d",
        help="Cache directory path",
    ),
) -> None:
    """Show cache statistics.

    Example:
        motools cache stats
    """
    utils = CacheUtils(cache_dir)
    stats = utils.get_stats()

    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Datasets", str(stats.num_datasets))
    table.add_row("Models", str(stats.num_models))
    table.add_row("Evaluations", str(stats.num_evals))
    table.add_row("Total Size", _format_bytes(stats.total_size_bytes))

    console.print(table)


@app.command()
def size(
    cache_dir: str = typer.Option(
        ".motools",
        "--cache-dir",
        "-d",
        help="Cache directory path",
    ),
    breakdown: bool = typer.Option(
        False,
        "--breakdown",
        "-b",
        help="Show size breakdown by category",
    ),
) -> None:
    """Show cache size information.

    Examples:
        motools cache size
        motools cache size --breakdown
    """
    utils = CacheUtils(cache_dir)

    if breakdown:
        sizes = utils.get_size_breakdown()
        total = sum(sizes.values())

        table = Table(title="Cache Size Breakdown")
        table.add_column("Category", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Percentage", style="yellow")

        for category, size in sizes.items():
            percentage = (size / total * 100) if total > 0 else 0
            table.add_row(
                category.capitalize(),
                _format_bytes(size),
                f"{percentage:.1f}%",
            )

        table.add_row("Total", _format_bytes(total), "100.0%", style="bold")
        console.print(table)
    else:
        stats = utils.get_stats()
        console.print(f"Total cache size: {_format_bytes(stats.total_size_bytes)}")


@app.command()
def clear(
    cache_type: str = typer.Argument(
        ...,
        help="Type of cache to clear: datasets, models, evals, or all",
    ),
    cache_dir: str = typer.Option(
        ".motools",
        "--cache-dir",
        "-d",
        help="Cache directory path",
    ),
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Clear cache entries.

    Examples:
        motools cache clear evals
        motools cache clear all --yes
    """
    utils = CacheUtils(cache_dir)

    if not confirm:
        confirmed = typer.confirm(
            f"Are you sure you want to clear {cache_type} cache?",
            abort=True,
        )
        if not confirmed:
            return

    if cache_type == "datasets":
        count = utils.clear_datasets()
        console.print(f"[green]Cleared {count} dataset(s)[/green]")

    elif cache_type == "models":
        count = utils.clear_models()
        console.print(f"[green]Cleared {count} model(s)[/green]")

    elif cache_type == "evals":
        count = utils.clear_evals()
        console.print(f"[green]Cleared {count} evaluation(s)[/green]")

    elif cache_type == "all":
        counts = utils.clear_all()
        console.print("[green]Cache cleared:[/green]")
        console.print(f"  Datasets: {counts['datasets']}")
        console.print(f"  Models: {counts['models']}")
        console.print(f"  Evaluations: {counts['evals']}")

    else:
        console.print(
            f"[red]Error:[/red] Invalid cache type '{cache_type}'. "
            "Must be one of: datasets, models, evals, all"
        )
        raise typer.Exit(1)


def _format_bytes(size: int) -> str:
    """Format bytes as human-readable string.

    Args:
        size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


if __name__ == "__main__":
    app()
