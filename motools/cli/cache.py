"""CLI commands for cache management."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from motools.cache.stage_cache import StageCache
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
        dataset_entries = utils.list_datasets()
        table = Table(title="Cached Datasets")
        table.add_column("Dataset Hash", style="cyan")
        table.add_column("File ID", style="green")
        for entry in dataset_entries:
            table.add_row(entry.dataset_hash, entry.file_id)
        console.print(table)
        console.print(f"\nTotal: {len(dataset_entries)} datasets")

    elif cache_type == "models":
        model_entries = utils.list_models()
        table = Table(title="Cached Models")
        table.add_column("Cache Key", style="cyan")
        table.add_column("Model ID", style="green")
        for model_entry in model_entries:
            table.add_row(model_entry.cache_key[:16] + "...", model_entry.model_id)
        console.print(table)
        console.print(f"\nTotal: {len(model_entries)} models")

    elif cache_type == "evals":
        eval_entries = utils.list_evals(model_id=model_id, task_id=task_id)
        table = Table(title="Cached Evaluations")
        table.add_column("ID", style="cyan")
        table.add_column("Model ID", style="green")
        table.add_column("Task ID", style="yellow")
        table.add_column("Backend", style="magenta")
        table.add_column("Created At", style="blue")
        for eval_entry in eval_entries:
            table.add_row(
                str(eval_entry.id),
                eval_entry.model_id,
                eval_entry.task_id,
                eval_entry.backend_type,
                eval_entry.created_at,
            )
        console.print(table)
        console.print(f"\nTotal: {len(eval_entries)} evaluations")

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


@app.command()
def list_jobs(
    cache_dir: str = typer.Option(
        ".motools",
        "--cache-dir",
        "-d",
        help="Cache directory path",
    ),
    show_status: bool = typer.Option(
        False,
        "--status",
        "-s",
        help="Query training job status (slower)",
    ),
) -> None:
    """List cached training jobs.

    Examples:
        motools cache list-jobs
        motools cache list-jobs --status
    """
    from motools.cache.training_jobs import list_training_jobs

    jobs = list_training_jobs(cache_dir=cache_dir)

    if not jobs:
        console.print("[yellow]No cached training jobs found[/yellow]")
        return

    table = Table(title="Cached Training Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Hyperparameters", style="yellow")
    table.add_column("Cache Key", style="magenta")
    if show_status:
        table.add_column("Status", style="blue")

    # Optionally fetch status for all jobs
    if show_status:
        from motools.atom import TrainingJobAtom

        async def get_statuses():
            statuses = {}
            for job in jobs:
                try:
                    atom = TrainingJobAtom.load(job["job_id"])
                    status = await atom.get_status()
                    statuses[job["job_id"]] = status
                except Exception:
                    statuses[job["job_id"]] = "unknown"
            return statuses

        statuses = asyncio.run(get_statuses())
    else:
        statuses = {}

    for job in jobs:
        model = job.get("metadata", {}).get("model", "unknown")
        hyperparams = str(job.get("metadata", {}).get("hyperparameters", {}))
        cache_key = job["cache_key"][:16] + "..."

        row = [job["job_id"], model, hyperparams, cache_key]
        if show_status:
            status = statuses.get(job["job_id"], "unknown")
            row.append(status)

        table.add_row(*row)

    console.print(table)
    console.print(f"\nTotal: {len(jobs)} training jobs")


@app.command()
def clean_failed_jobs(
    cache_dir: str = typer.Option(
        ".motools",
        "--cache-dir",
        "-d",
        help="Cache directory path",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deleted without actually deleting",
    ),
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Find and delete cache entries for failed/cancelled training jobs.

    Examples:
        motools cache clean-failed-jobs --dry-run
        motools cache clean-failed-jobs --yes
    """
    from motools.atom import TrainingJobAtom
    from motools.cache.training_jobs import list_training_jobs

    console.print("[cyan]Checking training job status...[/cyan]")

    jobs = list_training_jobs(cache_dir=cache_dir)

    async def find_failed():
        failed = []
        for job in jobs:
            try:
                atom = TrainingJobAtom.load(job["job_id"])
                status = await atom.get_status()
                if status in ["cancelled", "failed"]:
                    failed.append(
                        {
                            "job_id": job["job_id"],
                            "status": status,
                            "cache_key": job["cache_key"],
                            "model": job.get("metadata", {}).get("model", "unknown"),
                            "hyperparameters": job.get("metadata", {}).get("hyperparameters", {}),
                        }
                    )
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to check {job['job_id']}: {e}[/yellow]")

        return failed

    failed_jobs = asyncio.run(find_failed())

    if not failed_jobs:
        console.print("[green]✓ No failed or cancelled training jobs found[/green]")
        return

    # Display failed jobs
    table = Table(title="Failed/Cancelled Training Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Status", style="red")
    table.add_column("Model", style="green")
    table.add_column("Hyperparameters", style="yellow")
    table.add_column("Cache Key", style="magenta")

    for job in failed_jobs:
        table.add_row(
            job["job_id"],
            job["status"],
            job["model"],
            str(job["hyperparameters"]),
            job["cache_key"][:16] + "...",
        )

    console.print(table)
    console.print(f"\nFound {len(failed_jobs)} failed/cancelled jobs")

    if dry_run:
        console.print("\n[yellow]Dry run - no changes made[/yellow]")
        return

    if not confirm:
        confirmed = typer.confirm(
            f"Delete cache entries for {len(failed_jobs)} failed job(s)?",
            abort=True,
        )
        if not confirmed:
            return

    # Delete cache entries
    cache = StageCache(cache_dir=cache_dir)
    deleted_count = 0

    for job in failed_jobs:
        if cache.delete_entry(job["cache_key"]):
            deleted_count += 1
            console.print(f"[green]✓ Deleted cache entry for {job['job_id']}[/green]")
        else:
            console.print(f"[yellow]⚠ Cache entry not found for {job['job_id']}[/yellow]")

    console.print(f"\n[green]Deleted {deleted_count} cache entries[/green]")


def _format_bytes(size: int) -> str:
    """Format bytes as human-readable string.

    Args:
        size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    size_float = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_float < 1024.0:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024.0
    return f"{size_float:.1f} PB"


if __name__ == "__main__":
    app()
