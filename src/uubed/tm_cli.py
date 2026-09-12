# this_file: src/uubed/tm_cli.py
"""Build and query portable translation memories."""

import json
from dataclasses import asdict
from pathlib import Path

import click

from uubed.embeddings import BACKENDS, Embedder
from uubed.memory import TranslationMemory, build_memory
from uubed.tm_indexes import SEARCH_BACKENDS, index_memory


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
@click.option("--search-backend", type=click.Choice(SEARCH_BACKENDS), default="sqlite", show_default=True)
@click.option(
    "--bit-width", type=click.IntRange(2, 4), default=4, show_default=True, help="Turbovec bits per coordinate."
)
def build(inputs, *, output, model, backend, model_path, device, dimensions, batch_size, search_backend, bit_width):
    """Stream TMX files/directories into a new SQLite file (never overwrite)."""
    paths = sorted({p for item in inputs for p in (item.rglob("*.tmx") if item.is_dir() else [item])})
    engine = Embedder(model, backend=backend, path=model_path, device=device, dimensions=dimensions)
    try:
        result = build_memory(
            paths, output, engine, batch_size=batch_size, search_backend=search_backend, bit_width=bit_width
        )
    finally:
        engine.close()
    click.echo(json.dumps(result, ensure_ascii=False))


@tm.command("index")
@click.argument("database", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--bit-width", type=click.IntRange(2, 4), default=4, show_default=True)
@click.option("--batch-size", type=click.IntRange(1), default=128, show_default=True)
@click.option("--search-backend", type=click.Choice(SEARCH_BACKENDS[1:]), default="turbovec", show_default=True)
def index(database, *, output, bit_width, batch_size, search_backend):
    """Copy a memory and add quantized search without re-embedding (never overwrite)."""
    click.echo(
        json.dumps(
            index_memory(database, output, bit_width=bit_width, batch_size=batch_size, search_backend=search_backend)
        )
    )


@tm.command("lookup")
@click.argument("database", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("text")
@click.option("--language", required=True)
@click.option("--model-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--top-k", default=5, type=click.IntRange(1, 100))
@click.option("--minimum", default=0.5, type=click.FloatRange(-1, 1))
@click.option("--exact-only", is_flag=True, help="Read verbatim matches without loading any model.")
@click.option("--search-backend", type=click.Choice(SEARCH_BACKENDS), help="Override the stored search backend.")
@click.option(
    "--related-to", multiple=True, help="Graph seed: an exact English source string; repeat for multiple seeds."
)
@click.option("--max-hops", type=click.IntRange(0, 100), default=2, show_default=True)
@click.option("--origin", multiple=True, help="Restrict graph results to this stored TMX provenance path.")
def lookup(
    database, text, *, language, model_path, top_k, minimum, exact_only, search_backend, related_to, max_hops, origin
):
    """Return exact variants, near pairs, scores and TMX provenance as JSON."""
    with TranslationMemory(database, model_path=model_path, search_backend=search_backend) as memory:
        result = memory.lookup(
            text,
            language,
            top_k=top_k,
            minimum=minimum,
            similar=not exact_only,
            related_to=list(related_to),
            max_hops=max_hops,
            origins=list(origin),
        )
    click.echo(json.dumps(asdict(result), ensure_ascii=False, indent=2))
