# this_file: src/uubed/memory.py
"""Portable exact-first translation memory backed by the shared Rust index."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from typing_extensions import Self

from uubed.compact import pack
from uubed.embeddings import Embedder
from uubed.tm_indexes import SEARCH_BACKENDS, QuantizedSearch, attach_index, index_type, validate_options
from uubed.tmx import language, translation_pairs


@dataclass(frozen=True)
class Match:
    source: str
    target: str
    language: str
    score: float
    provenance: list[list[str]]


@dataclass(frozen=True)
class Lookup:
    exact: list[Match]
    similar: list[Match]


def _native() -> Any:
    try:
        from uubed_native import NativeTm  # noqa: PLC0415
    except ImportError as error:
        msg = "translation memory requires uubed[tm]; install the uubed-rs wheel"
        raise ImportError(msg) from error
    return NativeTm


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_memory(
    paths: list[Path],
    output: Path,
    embedder: Embedder,
    *,
    batch_size: int = 128,
    search_backend: str = "sqlite",
    bit_width: int = 4,
) -> dict:
    """Build a new index atomically, embedding each distinct English string once."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not paths or batch_size < 1:
        msg = "provide TMX paths and a positive batch size"
        raise ValueError(msg)
    if search_backend not in SEARCH_BACKENDS:
        msg_0 = "search_backend must be sqlite, turbovec or turbo-graph"
        raise ValueError(msg_0)
    if search_backend != "sqlite":
        validate_options(embedder.dimensions, bit_width, batch_size)
        index_type(search_backend)
    if embedder.task != "similarity":
        msg = "translation memory requires the symmetric similarity task"
        raise ValueError(msg)
    inputs = [{"path": str(Path(p)), "sha256": _hash(Path(p))} for p in sorted(set(map(Path, paths)))]
    config = {
        "format": "uubed-tm-1",
        "space": embedder.space,
        "dimensions": embedder.dimensions,
        "precision": "int8",
        "source_language": "en",
        "inputs": inputs,
        "engine": {
            "model": embedder.model.model_id,
            "backend": embedder.backend,
            "dimensions": embedder.dimensions,
            "max_tokens": embedder.max_tokens,
            "prefix": embedder.prefix,
        },
    }
    if embedder.backend in {"jina", "voyage"}:
        config["engine"].pop("prefix")  # Hosted task APIs supply their own prefix.
    if getattr(embedder, "endpoint", None):
        config["engine"]["endpoint"] = embedder.endpoint
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".uubed-tm-", dir=output.parent) as directory:
        temporary = Path(directory) / "memory.sqlite"
        native = _native().create(str(temporary), json.dumps(config), embedder.dimensions)
        try:
            batch = []
            for item in inputs:
                path = Path(item["path"])
                for pair in translation_pairs(path):
                    batch.append(pair)
                    if len(batch) >= batch_size:
                        native.add_pairs(json.dumps(batch, ensure_ascii=False))
                        batch.clear()
                if _hash(path) != item["sha256"]:
                    msg = f"TMX changed during import: {path}"
                    raise ValueError(msg)
            if batch:
                native.add_pairs(json.dumps(batch, ensure_ascii=False))
            after = 0
            while pending := native.pending(after, batch_size):
                vectors = embedder.embed([text for _, text in pending])
                records = [pack(v, space=embedder.space, precision="int8") for v in vectors]
                if embedder.space != config["space"]:
                    msg = "embedding space changed during build"
                    raise ValueError(msg)
                native.set_vectors(
                    [(row[0], list(record.payload), record.scale) for row, record in zip(pending, records, strict=True)]
                )
                after = pending[-1][0]
            sources, pairs, missing = native.stats()
            if sources == 0 or missing:
                msg = "index has no English translation pairs or incomplete embeddings"
                raise ValueError(msg)
        finally:
            native.close()
        index_bytes = (
            attach_index(temporary, bit_width=bit_width, batch_size=batch_size, backend=search_backend)
            if search_backend != "sqlite"
            else 0
        )
        # Atomic no-clobber publication on the same filesystem.
        os.link(temporary, output)
    return {
        "sources": sources,
        "pairs": pairs,
        "dimensions": embedder.dimensions,
        "vector_bytes": sources * embedder.dimensions,
        "file_bytes": output.stat().st_size,
        "search_backend": search_backend,
        "index_bytes": index_bytes,
    }


MAX_RESULTS = 100
MAX_GRAPH_HOPS = 100


class TranslationMemory:
    """Read-only SQLite index; model loading is lazy and exact lookup is model-free."""

    def __init__(
        self,
        path: Path | str,
        *,
        embedder: Embedder | None = None,
        model_path: str | None = None,
        search_backend: str | None = None,
    ):
        self._db = _native()(str(path))
        self.config = json.loads(self._db.config())
        self.search_backend = self.config.get("search_backend", "sqlite") if search_backend is None else search_backend
        if self.search_backend not in SEARCH_BACKENDS:
            self._db.close()
            msg = "search_backend must be sqlite, turbovec or turbo-graph"
            raise ValueError(msg)
        if self.search_backend != "sqlite" and self.search_backend not in self.config:
            self._db.close()
            msg = "memory has no requested vector index; use uubed tm index first"
            raise ValueError(msg)
        self._path = Path(path).resolve()
        self._search_index = None
        self._embedder = embedder
        self._owns_embedder = embedder is None
        self._model_path = model_path

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()
        if self._search_index is not None:
            self._search_index.close()
            self._search_index = None
        if self._owns_embedder and self._embedder is not None:
            self._embedder.close()

    def exact(self, text: str, target_language: str) -> list[Match]:
        return [Match(**row) for row in json.loads(self._db.exact(text, language(target_language)))]

    def lookup(
        self,
        text: str,
        target_language: str,
        *,
        top_k: int = 5,
        minimum: float = 0.5,
        similar: bool = True,
        related_to: list[str] | None = None,
        max_hops: int = 2,
        origins: list[str] | None = None,
    ) -> Lookup:
        """Return all exact variants and bounded similar pairs; never pick a conflict."""
        exact = self.exact(text, target_language)
        if not similar:
            return Lookup(exact, [])
        if (related_to or origins) and self.search_backend != "turbo-graph":
            msg = "graph/provenance constraints require the turbo-graph search backend"
            raise ValueError(msg)
        if not isinstance(max_hops, int) or isinstance(max_hops, bool) or not 0 <= max_hops <= MAX_GRAPH_HOPS:
            msg = "max_hops must be an integer between 0 and 100"
            raise ValueError(msg)
        if not 1 <= top_k <= MAX_RESULTS:
            msg = "top_k must be between 1 and 100"
            raise ValueError(msg)
        if not math.isfinite(minimum) or not -1 <= minimum <= 1:
            msg_0 = "minimum must be a finite cosine between -1 and 1"
            raise ValueError(msg_0)
        if self.search_backend != "sqlite" and self._search_index is None:
            self._search_index = QuantizedSearch(self._path, self.config, self.search_backend)
        if self._embedder is None:
            options = dict(self.config["engine"])
            options["device"] = "cpu"
            if options["backend"] == "llama.cpp":
                if self._model_path is None:
                    msg = "semantic lookup requires model_path for this GGUF index"
                    raise ValueError(msg)
                options["path"] = self._model_path
            self._embedder = Embedder(**options)
        if self._embedder.space != self.config["space"] or self._embedder.dimensions != self.config["dimensions"]:
            msg = "query embedding space or dimensions do not match the index"
            raise ValueError(msg)
        query = self._embedder.embed([text])[0].tolist()
        if self._search_index is not None:
            candidates = self._search_index.search(
                query,
                language(target_language),
                text,
                top_k,
                minimum,
                related_to=related_to,
                max_hops=max_hops,
                origins=origins,
            )
            matches = [
                Match(**{**row, "score": score})
                for source, score in candidates
                for row in json.loads(self._db.exact(source, language(target_language)))
            ]
            return Lookup(exact, matches[:top_k])
        candidates = json.loads(self._db.search(query, language(target_language), min(top_k + 1, MAX_RESULTS), minimum))
        return Lookup(exact, [Match(**row) for row in candidates if row["source"] != text][:top_k])
