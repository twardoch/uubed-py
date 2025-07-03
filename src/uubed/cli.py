#!/usr/bin/env python3
# this_file: src/uubed/cli.py
"""Command-line interface for uubed."""

import sys
import time
from pathlib import Path
from typing import Optional, Any, TextIO, BinaryIO, List

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .api import encode, decode, EncodingMethod
from .native_wrapper import is_native_available
from .encoders import get_available_encoders # Import to dynamically get encoders

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="uubed")
def main() -> None:
    """High-performance encoding for embedding vectors."""
    pass


@main.command("encode") # Explicitly name the command
@click.argument("input_file", type=click.File("rb"), default="-", metavar="INPUT")
@click.option(
    "--method",
    "-m",
    type=click.Choice(["eq64", "shq64", "t8q64", "zoq64", "mq64", "auto"]),
    default="auto",
    help="Encoding method (default: auto).",
)
@click.option("--output_file", "-o", type=click.File("w"), default="-", metavar="OUTPUT", help="Output file.")
@click.option("--k", type=int, default=8, help="Top-k value for t8q64 method.")
@click.option("--planes", type=int, default=64, help="Number of planes for shq64 method.")
def encode_cmd(
    input_file: BinaryIO,
    method: EncodingMethod,
    output_file: TextIO,
    k: int,
    planes: int
) -> None:
    """Encode embedding vector from INPUT and write to OUTPUT.

    INPUT can be a file path or '-' for stdin.
    OUTPUT can be a file path or '-' for stdout.
    """
    try:
        # Read input bytes from the specified file or stdin.
        data: bytes = input_file.read()
        
        # Prepare keyword arguments for the `encode` function based on the selected method.
        kwargs: dict[str, Any] = {}
        if method == "t8q64":
            kwargs["k"] = k
        elif method == "shq64":
            kwargs["planes"] = planes
        # Add mq64 specific kwargs if any
        # if method == "mq64":
        #     kwargs["levels"] = ...
        
        # Perform encoding. If output is stdout, suppress progress bar.
        if output_file == sys.stdout:
            # Directly call encode without a progress bar for stdout output.
            result: str = encode(data, method=method, **kwargs)
        else:
            # Use a rich progress bar for file output.
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Encoding...", total=None) # Total is unknown for stream.
                result = encode(data, method=method, **kwargs)
        
        # Write the encoded result to the specified output file or stdout.
        output_file.write(result + "\n")
        
        # Provide feedback to the user, but only if not writing to stdout (to avoid polluting output).
        if output_file != sys.stdout:
            console.print(f"[green]✓[/green] Encoded {len(data)} bytes using {method}.")
            
    except Exception as e:
        # Catch any exceptions during the process and print a user-friendly error message.
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command("decode") # Explicitly name the command
@click.argument("input_file", type=click.File("r"), default="-", metavar="INPUT")
@click.option(
    "--method",
    "-m",
    type=click.Choice(["eq64", "mq64"]),
    default=None,
    help="Encoding method used for decoding (auto-detect if not specified). Only eq64 and mq64 support decoding.",
)
@click.option("--output_file", "-o", type=click.File("wb"), default="-", metavar="OUTPUT", help="Output file.")
def decode_cmd(
    input_file: TextIO,
    method: Optional[EncodingMethod],
    output_file: BinaryIO
) -> None:
    """Decode encoded string from INPUT and write to OUTPUT.

    INPUT can be a file path or '-' for stdin.
    OUTPUT can be a file path or '-' for stdout.
    """
    try:
        # Read the encoded string from the specified input file or stdin.
        encoded: str = input_file.read().strip()
        
        # Perform decoding. If output is stdout, suppress progress bar.
        if output_file == sys.stdout.buffer:
            # Directly call decode without a progress bar for stdout output.
            result: bytes = decode(encoded, method=method)
        else:
            # Use a rich progress bar for file output.
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Decoding...", total=None) # Total is unknown for stream.
                result = decode(encoded, method=method)
        
        # Write the decoded result (bytes) to the specified output file or stdout.
        output_file.write(result)
        
        # Provide feedback to the user, but only if not writing to stdout.
        if output_file != sys.stdout.buffer:
            console.print(f"[green]✓[/green] Decoded to {len(result)} bytes.")
            
    except Exception as e:
        # Catch any exceptions during the process and print a user-friendly error message.
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option("--size", "-n", type=int, default=1000, help="Number of embeddings to benchmark.")
@click.option("--dims", "-d", type=int, default=768, help="Embedding dimensions.")
@click.option("--iterations", "-i", type=int, default=100, help="Number of iterations for each benchmark.")
@click.option(
    "--method",
    "-m",
    type=click.Choice(["eq64", "shq64", "t8q64", "zoq64", "mq64", "all"]),
    default="all",
    help="Encoding method to benchmark (or 'all' for all methods).",
)
def bench(
    size: int,
    dims: int,
    iterations: int,
    method: str
) -> None:
    """Benchmark encoding performance for specified methods and parameters."""
    console.print(Panel.fit(
        f"[bold]uubed Benchmark[/bold]\n"
        f"Size: {size} embeddings\n"
        f"Dimensions: {dims}\n"
        f"Iterations: {iterations}",
        title="Configuration"
    ))
    
    # Generate random test data for benchmarking.
    console.print("\nGenerating test data...")
    embeddings: List[bytes] = [
        np.random.randint(0, 256, dims, dtype=np.uint8).tobytes()
        for _ in range(size)
    ]
    
    # Determine which methods to benchmark.
    methods_to_benchmark: List[str] = get_available_encoders() if method == "all" else [method]
    
    # Create a rich table to display benchmark results.
    table = Table(title="Benchmark Results")
    table.add_column("Method", style="cyan")
    table.add_column("Total Time (s)", justify="right")
    table.add_column("Per Embedding (μs)", justify="right")
    table.add_column("Throughput (embeddings/s)", justify="right")
    
    # Iterate through each method and run benchmarks.
    for m in methods_to_benchmark:
        console.print(f"\nBenchmarking {m}...")
        
        # Warm-up phase to ensure JIT compilation or caching doesn't skew results.
        for emb in embeddings[:10]: # Process a small subset first.
            encode(emb, method=m)
        
        # Measure the time taken for multiple iterations of encoding.
        start_time: float = time.perf_counter()
        for _ in range(iterations):
            for emb in embeddings:
                encode(emb, method=m)
        end_time: float = time.perf_counter()
        
        # Calculate performance metrics.
        total_time: float = end_time - start_time
        total_ops: int = size * iterations
        per_embedding_us: float = (total_time / total_ops) * 1_000_000  # Convert to microseconds.
        throughput: float = total_ops / total_time # Embeddings per second.
        
        # Add results to the table.
        table.add_row(
            m,
            f"{total_time:.3f}",
            f"{per_embedding_us:.2f}",
            f"{throughput:,.0f}"
        )
    
    # Print the benchmark results table.
    console.print("\n", table)


@main.command()
def info() -> None:
    """Display build information and feature flags of the uubed installation."""
    console.print(Panel.fit(
        f"[bold]uubed v{__version__}[/bold]\n"
        f"High-performance encoding for embedding vectors",
        title="About"
    ))
    
    # Check and display native extension availability.
    native_status: str = "[green]✓ Available[/green]" if is_native_available() else "[red]✗ Not available[/red]"
    
    # Create a table to display system and build information.
    table = Table(title="Build Information")
    table.add_column("Feature", style="cyan")
    table.add_column("Status")
    
    table.add_row("Native Extension", native_status)
    table.add_row("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    table.add_row("NumPy Version", np.__version__)
    
    # Dynamically retrieve and display available encoders.
    available_encoders: List[str] = get_available_encoders()
    table.add_row("Available Encoders", ", ".join(available_encoders))
    
    console.print(table)
    
    # Provide example usage for the CLI.
    console.print("\n[bold]Example Usage:[/bold]")
    console.print("  uubed encode input.bin -o output.txt")
    console.print("  uubed decode output.txt -o restored.bin")
    console.print("  uubed bench --size 1000 --dims 768")


if __name__ == "__main__":
    main()
