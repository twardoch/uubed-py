# this_file: src/uubed/tm_indexes.py
"""Optional compressed candidate search; SQLite remains the portable TM authority."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from uubed.tm_graph import (
    UPSTREAM_REVISION,
    add_records,
    dump_graph,
    graph_candidates,
    graph_type,
    link_adjacency,
    load_graph,
)

SEARCH_BACKENDS = ("sqlite", "turbovec", "turbo-graph")
MIN_CANDIDATES = 32
CANDIDATE_FACTOR = 4
MAX_DIMENSIONS = 16384
FORBIDDEN_INT8 = -128


def index_type(backend: str = "turbovec") -> Any:
    """Import only when compressed indexing/search is actually requested."""
    if backend == "turbo-graph":
        return graph_type()
    try:
        from turbovec import IdMapIndex  # noqa: PLC0415 - optional dependency
    except ImportError as error:
        msg = "Turbovec search requires uubed[turbovec]; install that extra"
        raise ImportError(msg) from error
    return IdMapIndex


def validate_options(dimensions: int, bit_width: int, batch_size: int) -> None:
    if type(bit_width) is not int or bit_width not in (2, 3, 4):
        msg = "Turbovec bit_width must be 2, 3 or 4"
        raise ValueError(msg)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        msg = "batch_size must be a positive integer"
        raise ValueError(msg)
    if type(dimensions) is not int or not 1 <= dimensions <= MAX_DIMENSIONS or dimensions % 8:
        msg = "Turbovec dimensions must be a positive multiple of 8, at most 16384"
        raise ValueError(msg)


def readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def vector(payload: bytes, dimensions: int) -> np.ndarray:
    """Validate and normalize the authoritative signed int8 representation."""
    values = np.frombuffer(payload, dtype=np.int8).astype(np.float64)
    norm = np.linalg.norm(values)
    if len(values) != dimensions or np.any(values == FORBIDDEN_INT8) or norm == 0:
        msg = "corrupt stored int8 vector"
        raise ValueError(msg)
    return values / norm


def attach_index(path: Path, *, bit_width: int = 4, batch_size: int = 128, backend: str = "turbovec") -> int:
    """Attach a snapshot to an unpublished database; caller publishes atomically."""
    with closing(sqlite3.connect(path)) as conn:
        config_json, dimensions = conn.execute("SELECT config, dimensions FROM metadata").fetchone()
        validate_options(dimensions, bit_width, batch_size)
        config = json.loads(config_json)
        index = index_type(backend)(dim=dimensions, bit_width=bit_width)
        rows = conn.execute("SELECT id, vector FROM sources ORDER BY id")
        while batch := rows.fetchmany(batch_size):
            if any(payload is None for _, payload in batch):
                msg = "index has incomplete embeddings"
                raise ValueError(msg)
            vectors = np.asarray([vector(payload, dimensions) for _, payload in batch], dtype=np.float32)
            ids = [row[0] for row in batch]
            if backend == "turbo-graph":
                add_records(index, conn, vectors, ids)
            else:
                index.add_with_ids(vectors, np.asarray(ids, dtype=np.uint64))
        if not len(index):
            msg = "index has no English translation pairs"
            raise ValueError(msg)
        if backend == "turbo-graph":
            link_adjacency(index, conn)
            payloads = dump_graph(index)
            implementation = UPSTREAM_REVISION
        else:
            payloads = {backend: index.to_bytes()}
            implementation = "1.0.0"
        config["search_backend"] = backend
        config[backend] = {
            "format": 1,
            "implementation": implementation,
            "bit_width": bit_width,
            "vectors": len(index),
            "checksums": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        }
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS vector_indexes(name TEXT PRIMARY KEY, payload BLOB NOT NULL)")
            conn.executemany("INSERT OR REPLACE INTO vector_indexes VALUES(?, ?)", payloads.items())
            conn.execute("UPDATE metadata SET config=?", (json.dumps(config),))
        return sum(map(len, payloads.values()))


def index_memory(
    source: Path | str,
    output: Path | str,
    *,
    bit_width: int = 4,
    batch_size: int = 128,
    search_backend: str = "turbovec",
) -> dict:
    """Copy a memory and add/rebuild a quantized index without loading a model."""
    if search_backend not in SEARCH_BACKENDS[1:]:
        msg = "index search_backend must be turbovec or turbo-graph"
        raise ValueError(msg)
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".uubed-tm-", dir=output.parent) as directory:
        temporary = Path(directory) / "memory.sqlite"
        with closing(readonly(source)) as src, closing(sqlite3.connect(temporary)) as dst:
            if src.execute("PRAGMA user_version").fetchone()[0] != 1:
                msg = "unsupported translation-memory schema"
                raise ValueError(msg)
            src.backup(dst)
        size = attach_index(temporary, bit_width=bit_width, batch_size=batch_size, backend=search_backend)
        os.link(temporary, output)
    return {"search_backend": search_backend, "index_bytes": size, "file_bytes": output.stat().st_size}


class QuantizedSearch:
    """Resident quantized candidates, reranked against SQLite int8 cosine scores."""

    def __init__(self, path: Path, config: dict, backend: str):
        self.conn = readonly(path)
        self.backend = backend
        try:
            metadata = config[backend]
            implementation = UPSTREAM_REVISION if backend == "turbo-graph" else "1.0.0"
            if metadata["format"] != 1 or metadata["implementation"] != implementation:
                msg = "vector index snapshot version mismatch; rebuild the index"
                raise ValueError(msg)
            names = (backend, "turbo-graph-edges") if backend == "turbo-graph" else (backend,)
            payloads = {}
            for name in names:
                row = self.conn.execute("SELECT payload FROM vector_indexes WHERE name=?", (name,)).fetchone()
                if row is None or hashlib.sha256(row[0]).hexdigest() != metadata["checksums"][name]:
                    msg = "vector index snapshot checksum mismatch; rebuild the index"
                    raise ValueError(msg)
                payloads[name] = row[0]
            if backend == "turbo-graph":
                self.index = load_graph(payloads)
            else:
                self.index = index_type().from_bytes(payloads[backend])
            self.dimensions = config["dimensions"]
            count = self.conn.execute("SELECT count(*) FROM sources").fetchone()[0]
            if backend == "turbo-graph":
                count += 1  # The directed global-query root.
            if (
                metadata["format"] != 1
                or len(self.index) != count
                or len(self.index) != metadata["vectors"]
                or self.index.dim != self.dimensions
                or self.index.bit_width != metadata["bit_width"]
            ):
                msg = "vector index snapshot metadata mismatch; rebuild the index"
                raise ValueError(msg)
        except BaseException:
            self.conn.close()
            raise

    def close(self) -> None:
        self.conn.close()
        self.index = None

    def search(
        self,
        query: list[float],
        language: str,
        text: str,
        limit: int,
        minimum: float,
        *,
        related_to: list[str] | None = None,
        max_hops: int = 2,
        origins: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        query = np.asarray(query, dtype=np.float64)
        norm = np.linalg.norm(query)
        if query.shape != (self.dimensions,) or not np.isfinite(norm) or norm == 0:
            msg = "query must have matching dimensions and a finite nonzero norm"
            raise ValueError(msg)
        query = query / norm
        candidates = max(MIN_CANDIDATES, limit * CANDIDATE_FACTOR)
        if self.backend == "turbo-graph":
            ids = graph_candidates(
                self.index,
                self.conn,
                query,
                language,
                text=text,
                candidates=candidates,
                related_to=related_to,
                max_hops=max_hops,
                origins=origins,
            )
        else:
            ids = self._flat_candidates(query, language, text, candidates)
        best = []
        for source_id in ids:
            source, payload = self.conn.execute(
                "SELECT text, vector FROM sources WHERE id=?", (int(source_id),)
            ).fetchone()
            score = float(np.clip(np.dot(vector(payload, self.dimensions), query), -1, 1))
            if score >= minimum:
                best.append((source, score, int(source_id)))
        best.sort(key=lambda row: (-row[1], row[2]))
        return [(source, score) for source, score, _ in best[:limit]]

    def _flat_candidates(self, query, language, text, candidates):
        allowed = self.conn.execute(
            "SELECT id FROM sources s WHERE text != ? COLLATE BINARY AND EXISTS "
            "(SELECT 1 FROM pairs p WHERE p.source_id=s.id AND p.language=?) ORDER BY id",
            (text, language),
        ).fetchall()
        if not allowed:
            return []
        _, ids = self.index.search(
            np.asarray([query], dtype=np.float32),
            k=min(len(allowed), candidates),
            allowlist=np.asarray(allowed, dtype=np.uint64).reshape(-1),
        )
        return ids[0]
