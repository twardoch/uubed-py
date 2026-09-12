# this_file: src/uubed/tm_graph.py
"""Turbo-Graph adapter: TMX adjacency, language tags and portable snapshots."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

UPSTREAM_REVISION = "72f10416d4c954d42561f90465190f000e29cca9"


def graph_type():
    try:
        from turbo_graph import GraphMemoryIndex  # noqa: PLC0415 - optional dependency
    except ImportError as error:
        msg = (
            "Turbo-Graph is optional and not yet on PyPI. Install the pinned source: "
            "uv pip install 'git+https://github.com/bigmacfive/turbo-graph@"
            f"{UPSTREAM_REVISION}#subdirectory=turbo-graph-python'"
        )
        raise ImportError(msg) from error
    return GraphMemoryIndex


def add_records(index, conn, vectors, ids):
    records = []
    for source_id in ids:
        text = conn.execute("SELECT text FROM sources WHERE id=?", (source_id,)).fetchone()[0]
        languages = conn.execute(
            "SELECT DISTINCT language FROM pairs WHERE source_id=? ORDER BY language", (source_id,)
        )
        records.append({"id": source_id, "title": text, "tags": [f"language:{row[0]}" for row in languages]})
    index.add_records(vectors, records)


def link_adjacency(index, conn):
    # A synthetic root has outgoing links only: global queries are a cached
    # one-hop view, while source-seeded traversals cannot escape through it.
    root = np.zeros((1, index.dim), dtype=np.float32)
    root[0, 0] = 1
    index.add_records(root, [{"id": 0, "title": "", "tags": []}])
    for (source_id,) in conn.execute("SELECT id FROM sources ORDER BY id"):
        index.link_directed(0, source_id, 1.0)
    previous = {}
    for source_id, origin in conn.execute("SELECT source_id, origin FROM pairs ORDER BY origin, rowid"):
        last = previous.get(origin)
        if last is not None and last != source_id:
            index.link_bidirectional(last, source_id, 1.0)
        previous[origin] = source_id


def dump_graph(index):
    with TemporaryDirectory(prefix="uubed-graph-") as directory:
        vectors, graph = Path(directory) / "index.tv", Path(directory) / "graph.tvgm"
        index.write(str(vectors), str(graph))
        return {"turbo-graph": vectors.read_bytes(), "turbo-graph-edges": graph.read_bytes()}


def load_graph(payloads):
    with TemporaryDirectory(prefix="uubed-graph-") as directory:
        vectors, graph = Path(directory) / "index.tv", Path(directory) / "graph.tvgm"
        vectors.write_bytes(payloads["turbo-graph"])
        graph.write_bytes(payloads["turbo-graph-edges"])
        return graph_type().load(str(vectors), str(graph))


def graph_candidates(index, conn, query, language, *, text, candidates, related_to, max_hops, origins):
    seeds = []
    for seed in related_to or []:
        row = conn.execute("SELECT id FROM sources WHERE text=? COLLATE BINARY", (seed,)).fetchone()
        if row is None:
            msg = f"unknown graph seed source: {seed!r}"
            raise ValueError(msg)
        seeds.append(row[0])
    if not seeds:
        seeds, max_hops = [0], 1
    candidate_ids = None
    if origins:
        placeholders = ",".join("?" for _ in origins)
        sql = f"SELECT DISTINCT source_id FROM pairs WHERE origin IN ({placeholders})"  # noqa: S608
        candidate_ids = [row[0] for row in conn.execute(sql, origins)]
        if not candidate_ids:
            return []
    # Exclude the query text after search with one extra candidate slot. Language
    # and graph constraints are applied within the native kernel, not afterwards.
    hits = index.search(
        np.asarray(query, dtype=np.float32),
        candidates + 1,
        seeds,
        max_hops=max_hops,
        required_tags=[f"language:{language}"],
        candidate_ids=candidate_ids,
    )
    index.trim_caches_for_preset("balanced")
    return [hit["id"] for hit in hits if hit["title"] != text][:candidates]
