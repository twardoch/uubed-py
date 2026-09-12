# this_file: src/uubed/tm_cli.py
"""Build and query portable translation memories."""

import json
from dataclasses import asdict
from pathlib import Path

import click

from uubed.embeddings import BACKENDS, Embedder
from uubed.memory import TranslationMemory, build_memory


@click.group()
def tm():
    """English-source TMX memory: verbatim reuse and semantic translation pairs."""


@tm.command("build")
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--model", default="jina", show_default=True)
@click.option("--backend", type=click.Choice(BACKENDS), default="sentence-transformers", show_default=True)
@click.option("--model-path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--device", default="cpu", show_default=True)
@click.option("--dimensions", default=256, type=click.IntRange(1), show_default=True)
@click.option("--batch-size", default=128, type=click.IntRange(1), show_default=True)
def build(inputs, *, output, model, backend, model_path, device, dimensions, batch_size):
    """Stream TMX files/directories into a new SQLite file (never overwrite)."""
    paths = sorted({p for item in inputs for p in (item.rglob("*.tmx") if item.is_dir() else [item])})
    engine = Embedder(model, backend=backend, path=model_path, device=device, dimensions=dimensions)
    try:
        result = build_memory(paths, output, engine, batch_size=batch_size)
    finally:
        engine.close()
    click.echo(json.dumps(result, ensure_ascii=False))


@tm.command("lookup")
@click.argument("database", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("text")
@click.option("--language", required=True)
@click.option("--model-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--top-k", default=5, type=click.IntRange(1, 100))
@click.option("--minimum", default=0.5, type=click.FloatRange(-1, 1))
@click.option("--exact-only", is_flag=True, help="Read verbatim matches without loading any model.")
def lookup(database, text, *, language, model_path, top_k, minimum, exact_only):
    """Return exact variants, near pairs, scores and TMX provenance as JSON."""
    with TranslationMemory(database, model_path=model_path) as memory:
        result = memory.lookup(text, language, top_k=top_k, minimum=minimum, similar=not exact_only)
    click.echo(json.dumps(asdict(result), ensure_ascii=False, indent=2))
