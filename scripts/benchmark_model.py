"""Benchmark model API throughput and latency.

This script measures sampling throughput and latency for any Inspect AI model provider.

Example usage:
    # Benchmark Tinker model
    uv run python scripts/benchmark_model.py --model "tinker/Llama-3.2-1B@latest" --num-requests 20

    # Concurrent benchmarking
    uv run python scripts/benchmark_model.py \\
        --model "openai/gpt-4" \\
        --num-requests 100 \\
        --concurrency 10 \\
        --output results.json

    # Custom prompt
    uv run python scripts/benchmark_model.py \\
        --model "anthropic/claude-3-5-sonnet-20241022" \\
        --prompt "Explain quantum computing" \\
        --max-tokens 500
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from inspect_ai.model import ChatMessageUser, get_model
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table


@dataclass
class RequestResult:
    """Result of a single benchmark request."""

    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    model: str
    num_requests: int
    concurrency: int
    prompt: str
    max_tokens: int
    temperature: float
    total_time_seconds: float
    requests_per_second: float
    latency_min: float
    latency_max: float
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None


async def benchmark_request(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    """Run a single benchmark request."""
    model = get_model(model_name)
    messages = [ChatMessageUser(content=prompt)]

    start_time = time.perf_counter()
    result = await model.generate(
        input=messages,
        config={
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    end_time = time.perf_counter()

    latency = end_time - start_time

    # Extract token usage if available
    input_tokens = None
    output_tokens = None
    total_tokens = None
    if result.usage:
        input_tokens = result.usage.input_tokens
        output_tokens = result.usage.output_tokens
        total_tokens = result.usage.total_tokens

    return RequestResult(
        latency_seconds=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


async def run_benchmark(
    model: str,
    num_requests: int,
    concurrency: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> BenchmarkResults:
    """Run benchmark with specified parameters."""
    console = Console()
    console.print(f"\n[bold cyan]Benchmarking {model}[/bold cyan]")
    console.print(f"Requests: {num_requests}, Concurrency: {concurrency}\n")

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request() -> RequestResult:
        async with semaphore:
            return await benchmark_request(model, prompt, max_tokens, temperature)

    # Run requests with progress bar
    results: list[RequestResult] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running requests", total=num_requests)

        start_time = time.perf_counter()

        # Create tasks for all requests
        tasks = [bounded_request() for _ in range(num_requests)]

        # Process as they complete
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            progress.update(task, advance=1)

        end_time = time.perf_counter()

    total_time = end_time - start_time

    # Calculate statistics
    latencies = [r.latency_seconds for r in results]
    latencies_array = np.array(latencies)

    # Aggregate token counts
    input_tokens_list = [r.input_tokens for r in results if r.input_tokens is not None]
    output_tokens_list = [r.output_tokens for r in results if r.output_tokens is not None]
    total_tokens_list = [r.total_tokens for r in results if r.total_tokens is not None]

    total_input_tokens = sum(input_tokens_list) if input_tokens_list else None
    total_output_tokens = sum(output_tokens_list) if output_tokens_list else None
    total_tokens = sum(total_tokens_list) if total_tokens_list else None

    return BenchmarkResults(
        model=model,
        num_requests=num_requests,
        concurrency=concurrency,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        total_time_seconds=total_time,
        requests_per_second=num_requests / total_time,
        latency_min=float(np.min(latencies_array)),
        latency_max=float(np.max(latencies_array)),
        latency_mean=float(np.mean(latencies_array)),
        latency_p50=float(np.percentile(latencies_array, 50)),
        latency_p95=float(np.percentile(latencies_array, 95)),
        latency_p99=float(np.percentile(latencies_array, 99)),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
    )


def print_results(results: BenchmarkResults) -> None:
    """Print benchmark results in a formatted table."""
    console = Console()

    console.print("\n[bold green]Benchmark Results[/bold green]\n")

    # Configuration table
    config_table = Table(title="Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="magenta")

    config_table.add_row("Model", results.model)
    config_table.add_row("Total Requests", str(results.num_requests))
    config_table.add_row("Concurrency", str(results.concurrency))
    config_table.add_row("Max Tokens", str(results.max_tokens))
    config_table.add_row("Temperature", str(results.temperature))

    console.print(config_table)
    console.print()

    # Performance table
    perf_table = Table(title="Performance Metrics")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="magenta")

    perf_table.add_row("Total Time", f"{results.total_time_seconds:.2f}s")
    perf_table.add_row("Throughput", f"{results.requests_per_second:.2f} req/s")
    perf_table.add_row("Latency (min)", f"{results.latency_min:.3f}s")
    perf_table.add_row("Latency (max)", f"{results.latency_max:.3f}s")
    perf_table.add_row("Latency (mean)", f"{results.latency_mean:.3f}s")
    perf_table.add_row("Latency (p50)", f"{results.latency_p50:.3f}s")
    perf_table.add_row("Latency (p95)", f"{results.latency_p95:.3f}s")
    perf_table.add_row("Latency (p99)", f"{results.latency_p99:.3f}s")

    console.print(perf_table)
    console.print()

    # Token usage table (if available)
    if results.total_tokens is not None:
        token_table = Table(title="Token Usage")
        token_table.add_column("Metric", style="cyan")
        token_table.add_column("Value", style="magenta")

        if results.total_input_tokens is not None:
            token_table.add_row("Total Input Tokens", str(results.total_input_tokens))
        if results.total_output_tokens is not None:
            token_table.add_row("Total Output Tokens", str(results.total_output_tokens))
        if results.total_tokens is not None:
            token_table.add_row("Total Tokens", str(results.total_tokens))

        console.print(token_table)
        console.print()


def save_results(results: BenchmarkResults, output_path: Path) -> None:
    """Save benchmark results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(asdict(results), f, indent=2)

    console = Console()
    console.print(f"[green]Results saved to {output_path}[/green]")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark model API throughput and latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name in Inspect format (e.g., 'tinker/Llama-3.2-1B@latest')",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Total number of requests to make (default: 10)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)",
    )
    parser.add_argument(
        "--prompt",
        default="Say hello in one sentence.",
        help="Prompt to use for benchmarking (default: 'Say hello in one sentence.')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens in response (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save JSON results (optional)",
    )

    args = parser.parse_args()

    # Run benchmark
    results = asyncio.run(
        run_benchmark(
            model=args.model,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    )

    # Print results
    print_results(results)

    # Save to file if requested
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
