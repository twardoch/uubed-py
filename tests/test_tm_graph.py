# this_file: tests/test_tm_graph.py
"""Actual Turbo-Graph language, provenance and adjacency-constrained retrieval."""

import json
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from uubed.cli import main
from uubed.memory import TranslationMemory, build_memory
from uubed.tm_indexes import index_memory


class Encoder:
    dimensions = 32
    space = "graph-test"
    task = "similarity"
    backend = "sentence-transformers"
    model = SimpleNamespace(model_id="test-model")
    max_tokens = 128
    prefix = ""

    def embed(self, texts):
        result = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        result[:, 0] = 1
        return result

    def close(self):
        pass


@pytest.fixture
def memories(tmp_path):
    paths = []
    for name, sources in [("book", ["seed", "neighbor", "distant"]), ("other", ["unrelated"])]:
        path = tmp_path / f"{name}.tmx"
        units = [
            f'<tu tuid="{i}"><tuv xml:lang="en"><seg>{text}</seg></tuv>'
            f'<tuv xml:lang="pl"><seg>{text}-pl</seg></tuv></tu>'
            for i, text in enumerate(sources)
        ]
        path.write_text("<tmx><body>" + "".join(units) + "</body></tmx>")
        paths.append(path)
    output = tmp_path / "memory.sqlite"
    build_memory(paths, output, Encoder(), search_backend="turbo-graph", batch_size=2)
    return output, paths


def test_graph_when_seeded_then_only_neighbors_and_language_survive_portable_reload(memories, tmp_path):
    output, paths = memories
    moved = tmp_path / "moved.sqlite"
    shutil.move(output, moved)
    with TranslationMemory(moved, embedder=Encoder()) as tm:
        assert tm.search_backend == "turbo-graph"
        assert len(tm.lookup("query", "pl", minimum=-1).similar) == 4
        result = tm.lookup("query", "pl", related_to=["seed"], max_hops=1).similar
        assert {h.source for h in result} == {"seed", "neighbor"}
        assert all(h.score == pytest.approx(1) for h in result)
        result = tm.lookup("query", "pl", related_to=["seed"], max_hops=0).similar
        assert [h.source for h in result] == ["seed"]
        assert tm.lookup("query", "de", related_to=["seed"]).similar == []
        result = tm.lookup("query", "pl", origins=[str(paths[1])]).similar
        assert [h.source for h in result] == ["unrelated"]
        assert not tm.lookup("query", "pl", related_to=["seed"], max_hops=1, origins=[str(paths[1])]).similar
        with pytest.raises(ValueError, match="unknown graph seed"):
            tm.lookup("query", "pl", related_to=["missing"])


def test_graph_when_dependency_missing_then_exact_and_explicit_sqlite_work(memories, monkeypatch):
    output, _ = memories
    monkeypatch.setitem(sys.modules, "turbo_graph", None)
    with TranslationMemory(output) as tm:
        assert tm.lookup("seed", "pl", similar=False).exact[0].target == "seed-pl"
    with TranslationMemory(output, embedder=Encoder(), search_backend="sqlite") as tm:
        assert tm.lookup("query", "pl").similar
        with pytest.raises(ValueError, match="turbo-graph"):
            tm.lookup("query", "pl", related_to=["seed"])
    with TranslationMemory(output, embedder=Encoder()) as tm, pytest.raises(ImportError, match="pinned source"):
        tm.lookup("query", "pl")


def test_graph_when_indexing_copy_then_both_optional_backends_remain_available(memories, tmp_path):
    original, _ = memories
    indexed = tmp_path / "both.sqlite"
    before = original.read_bytes()
    index_memory(original, indexed, search_backend="turbovec")
    assert original.read_bytes() == before
    with TranslationMemory(indexed, embedder=Encoder(), search_backend="turbo-graph") as tm:
        assert {h.source for h in tm.lookup("query", "pl", related_to=["seed"], max_hops=1).similar} == {
            "seed",
            "neighbor",
        }
    with TranslationMemory(indexed, embedder=Encoder()) as tm:
        assert tm.search_backend == "turbovec"
        assert tm.lookup("query", "pl").similar


def test_graph_when_cli_constraints_then_used_by_native_graph(memories, monkeypatch):
    import uubed.memory as module

    monkeypatch.setattr(module, "Embedder", lambda **_kw: Encoder())
    path, _ = memories
    result = CliRunner().invoke(
        main, ["tm", "lookup", str(path), "query", "--language", "pl", "--related-to", "seed", "--max-hops", "1"]
    )
    assert result.exit_code == 0, result.output
    assert {h["source"] for h in json.loads(result.output)["similar"]} == {"seed", "neighbor"}
