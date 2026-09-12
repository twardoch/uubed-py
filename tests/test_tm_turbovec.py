# this_file: tests/test_tm_turbovec.py
"""Real Turbovec TM persistence, filtered retrieval and optional dependency contract."""

import json
import shutil
import sqlite3
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
    space = "turbovec-test-space"
    task = "similarity"
    backend = "sentence-transformers"
    model = SimpleNamespace(model_id="test-model")
    max_tokens = 128
    prefix = ""

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.extend(texts)
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0 if "font" in text else 1] = -1 if "opposite" in text else 1
        return vectors

    def close(self):
        pass


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "sample.tmx"
    path.write_text("""<tmx><body>
<tu tuid="1"><tuv xml:lang="en"><seg>A font</seg></tuv><tuv xml:lang="pl"><seg>czcionka</seg></tuv></tu>
<tu tuid="2"><tuv xml:lang="en"><seg>A font</seg></tuv><tuv xml:lang="pl"><seg>krój</seg></tuv></tu>
<tu tuid="3"><tuv xml:lang="en"><seg>A cat</seg></tuv><tuv xml:lang="de"><seg>Katze</seg></tuv></tu>
<tu tuid="4"><tuv xml:lang="en"><seg>opposite font</seg></tuv><tuv xml:lang="pl"><seg>przeciwieństwo</seg></tuv></tu>
</body></tmx>""")
    return path


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_search_index_when_built_then_portable_exact_and_reranked_pairs(corpus, tmp_path, bits):
    engine = Encoder()
    output = tmp_path / "memory.sqlite"
    stats = build_memory([corpus], output, engine, search_backend="turbovec", bit_width=bits, batch_size=1)
    assert len(engine.calls) == 3, "each source must be embedded only once"
    assert stats["search_backend"] == "turbovec"
    assert stats["index_bytes"] > 0
    moved = tmp_path / "moved ?#.sqlite"
    shutil.move(output, moved)
    with TranslationMemory(moved, embedder=engine) as tm:
        assert tm.search_backend == "turbovec"
        assert len(tm.exact("A font", "pl")) == 2
        assert tm.exact("a font", "pl") == []
        assert tm.exact("A font ", "pl") == []
        matches = tm.lookup("Another font", "pl", minimum=0.99).similar
        assert {m.target for m in matches} == {"czcionka", "krój"}
        assert all(m.score == pytest.approx(1.0) and m.provenance for m in matches)
        assert tm.lookup("Another font", "de", minimum=-1).similar[0].target == "Katze"
        assert tm.lookup("Another font", "zz", minimum=-1).similar == []
        assert not tm.lookup("A font", "pl", minimum=0).similar, "exclude exact source before candidate search"
        opposite = tm.lookup("Another font", "pl", minimum=-1).similar[-1]
        assert opposite.score == pytest.approx(-1)


def test_search_index_when_added_to_existing_then_source_unchanged_and_sqlite_available(corpus, tmp_path, monkeypatch):
    original, indexed = tmp_path / "old.sqlite", tmp_path / "indexed.sqlite"
    build_memory([corpus], original, Encoder())
    before = original.read_bytes()
    index_memory(original, indexed)
    assert original.read_bytes() == before
    monkeypatch.setitem(sys.modules, "turbovec", None)
    with TranslationMemory(indexed) as tm:
        assert tm.lookup("A font", "pl", similar=False).exact
    with TranslationMemory(indexed, embedder=Encoder(), search_backend="sqlite") as tm:
        assert tm.lookup("Another font", "pl").similar
    with TranslationMemory(indexed, embedder=Encoder()) as tm, pytest.raises(ImportError, match=r"uubed\[turbovec\]"):
        tm.lookup("Another font", "pl")
    with pytest.raises(FileExistsError):
        index_memory(original, indexed)


@pytest.mark.parametrize(
    "kwargs", [{"bit_width": 1}, {"bit_width": True}, {"batch_size": 0}, {"search_backend": "unknown"}]
)
def test_search_index_when_invalid_build_options_then_no_inference_or_output(corpus, tmp_path, kwargs):
    output = tmp_path / "bad.sqlite"
    engine = Encoder()
    options = {"search_backend": "turbovec", **kwargs}
    with pytest.raises(ValueError, match=r"bit_width|batch_size|batch size|search_backend"):
        build_memory([corpus], output, engine, **options)
    assert not output.exists()
    assert not engine.calls


def test_search_index_when_unsupported_dimensions_or_missing_dependency_then_no_output(corpus, tmp_path, monkeypatch):
    engine = Encoder()
    engine.dimensions = 2
    with pytest.raises(ValueError, match="multiple of 8"):
        build_memory([corpus], tmp_path / "bad.sqlite", engine, search_backend="turbovec")
    monkeypatch.setitem(sys.modules, "turbovec", None)
    with pytest.raises(ImportError, match=r"uubed\[turbovec\]"):
        build_memory([corpus], tmp_path / "missing.sqlite", Encoder(), search_backend="turbovec")
    assert not list(tmp_path.glob("*.sqlite"))


def test_search_index_when_corrupt_snapshot_then_fail_but_exact_and_sqlite_still_work(corpus, tmp_path):
    output = tmp_path / "memory.sqlite"
    build_memory([corpus], output, Encoder(), search_backend="turbovec")
    with sqlite3.connect(output) as conn:
        conn.execute("UPDATE vector_indexes SET payload=?", (b"broken",))
    with TranslationMemory(output, embedder=Encoder()) as tm:
        assert tm.exact("A font", "pl")
        with pytest.raises(ValueError, match="checksum"):
            tm.lookup("Another font", "pl")
    with TranslationMemory(output, embedder=Encoder(), search_backend="sqlite") as tm:
        assert tm.lookup("Another font", "pl").similar


def test_search_index_when_cli_build_index_lookup_then_backend_preserved(corpus, tmp_path, monkeypatch):
    import uubed.memory as memory_module
    import uubed.tm_cli as cli_module

    monkeypatch.setattr(cli_module, "Embedder", lambda *_a, **_kw: Encoder())
    monkeypatch.setattr(memory_module, "Embedder", lambda **_kw: Encoder())
    output = tmp_path / "cli.sqlite"
    runner = CliRunner()
    result = runner.invoke(main, ["tm", "build", str(corpus), "--output", str(output), "--search-backend", "turbovec"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["search_backend"] == "turbovec"
    result = runner.invoke(main, ["tm", "lookup", str(output), "Another font", "--language", "pl"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["similar"]
    second = tmp_path / "second.sqlite"
    result = runner.invoke(main, ["tm", "index", str(output), "--output", str(second), "--bit-width", "2"])
    assert result.exit_code == 0, result.output
    with TranslationMemory(second) as tm:
        assert tm.config["turbovec"]["bit_width"] == 2


def test_search_index_when_snapshot_version_changes_then_reject_before_deserializing(corpus, tmp_path):
    output = tmp_path / "memory.sqlite"
    build_memory([corpus], output, Encoder(), search_backend="turbovec")
    with sqlite3.connect(output) as conn:
        config = json.loads(conn.execute("SELECT config FROM metadata").fetchone()[0])
        config["turbovec"]["implementation"] = "unknown-version"
        conn.execute("UPDATE metadata SET config=?", (json.dumps(config),))
    with TranslationMemory(output, embedder=Encoder()) as tm:
        assert tm.exact("A font", "pl")
        with pytest.raises(ValueError, match="version"):
            tm.lookup("Another font", "pl")


@pytest.mark.parametrize("bits", [2.0, "2", None])
def test_search_index_when_noninteger_bits_then_clear_error(corpus, tmp_path, bits):
    with pytest.raises(ValueError, match="bit_width"):
        build_memory([corpus], tmp_path / "bad.sqlite", Encoder(), search_backend="turbovec", bit_width=bits)
