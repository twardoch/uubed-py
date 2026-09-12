# this_file: tests/test_memory.py
"""Exact-first lookup, source deduplication and portable native storage."""

import shutil
from types import SimpleNamespace
from xml.etree.ElementTree import ParseError

import numpy as np
import pytest

from uubed.memory import TranslationMemory, build_memory
from uubed.tmx import translation_pairs


class Encoder:
    space = "test-space"
    dimensions = 2
    task = "similarity"
    backend = "sentence-transformers"
    model = SimpleNamespace(model_id="test-model")
    max_tokens = 128
    prefix = ""

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.extend(texts)
        return np.array([[1.0, 0.0] if "font" in text else [0.0, 1.0] for text in texts], dtype=np.float32)


@pytest.fixture
def tmx(tmp_path):
    path = tmp_path / "sample.tmx"
    path.write_text("""<tmx><body>
    <tu tuid="1"><tuv xml:lang="pl"><seg>czcionka</seg></tuv>
<tuv xml:lang="en"><seg>A font</seg></tuv>
<tuv xml:lang="de"><seg>Schrift</seg></tuv></tu>
    <tu tuid="2"><tuv xml:lang="en"><seg>A cat</seg></tuv>
<tuv xml:lang="pl"><seg>kot</seg></tuv></tu>
    <tu tuid="3"><tuv xml:lang="en"><seg>A font</seg></tuv>
<tuv xml:lang="pl"><seg>krój</seg></tuv></tu>
    </body></tmx>""")
    return path


def test_memory_when_built_then_exact_conflicts_and_similar_pairs_survive_move(tmx, tmp_path):
    engine = Encoder()
    path = tmp_path / "memory.sqlite"
    build_memory([tmx], path, engine)
    assert sorted(engine.calls) == ["A cat", "A font"], "embed each English source once"
    moved = tmp_path / "elsewhere.sqlite"
    shutil.move(path, moved)
    with TranslationMemory(moved, embedder=engine) as memory:
        assert len(memory.exact("A font", "pl")) == 2
        assert not memory.exact("a font", "pl"), "verbatim means case sensitive"
        assert not memory.exact("A font ", "pl"), "verbatim preserves whitespace"
        result = memory.lookup("Another font", "pl", minimum=0.5)
        assert result.exact == []
        assert {h.target for h in result.similar} == {"czcionka", "krój"}
        assert all(h.provenance for h in result.similar)


def test_exact_lookup_needs_no_embedder_and_space_mismatch_fails(tmx, tmp_path):
    path = tmp_path / "memory.sqlite"
    build_memory([tmx], path, Encoder())
    with TranslationMemory(path) as memory:
        assert memory.exact("A font", "de")[0].target == "Schrift"
    wrong = Encoder()
    wrong.space = "other-model"
    with TranslationMemory(path, embedder=wrong) as memory, pytest.raises(ValueError, match="space"):
        memory.lookup("font", "pl")


def test_build_when_malformed_then_no_partial_output(tmx, tmp_path):
    path = tmp_path / "bad.tmx"
    path.write_text("<tmx><body><tu>")
    output = tmp_path / "memory.sqlite"
    with pytest.raises(ParseError):
        build_memory([tmx, path], output, Encoder())
    assert not output.exists()


def test_tmx_when_inline_markup_and_whitespace_then_preserve_segment_text(tmp_path):
    path = tmp_path / "inline.tmx"
    path.write_text("""<tmx><body><tu><tuv xml:lang="en"><seg> A <ph>&lt;b&gt;</ph>font </seg></tuv>
<tuv xml:lang="pl"><seg> czcionka </seg></tuv></tu></body></tmx>""")
    pair = next(iter(translation_pairs(path)))
    assert pair["source"] == " A <b>font "
    assert pair["target"] == " czcionka "


def test_cli_exact_only_needs_no_weights(tmx, tmp_path):
    import json

    from click.testing import CliRunner

    from uubed.cli import main

    path = tmp_path / "memory.sqlite"
    build_memory([tmx], path, Encoder())
    result = CliRunner().invoke(main, ["tm", "lookup", str(path), "A font", "--language", "de", "--exact-only"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["exact"][0]["target"] == "Schrift"
    assert parsed["similar"] == []


def test_build_rejects_existing_and_empty_without_replacing(tmx, tmp_path):
    path = tmp_path / "memory.sqlite"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        build_memory([tmx], path, Encoder())
    assert path.read_bytes() == b"existing"
    empty = tmp_path / "empty.tmx"
    empty.write_text("<tmx><body/></tmx>")
    with pytest.raises(ValueError, match="no English"):
        build_memory([empty], tmp_path / "empty.sqlite", Encoder())
    assert not (tmp_path / "empty.sqlite").exists()


def test_tmx_rejects_ambiguous_english_and_entity_expansion(tmp_path):
    from defusedxml.common import EntitiesForbidden

    path = tmp_path / "ambiguous.tmx"
    path.write_text(
        '<tmx><body><tu><tuv xml:lang="en"><seg>color</seg></tuv>'
        '<tuv xml:lang="en-GB"><seg>colour</seg></tuv></tu></body></tmx>'
    )
    with pytest.raises(ValueError, match="ambiguous English"):
        list(translation_pairs(path))
    path.write_text('<!DOCTYPE tmx [<!ENTITY text "expanded">]><tmx><body>&text;</body></tmx>')
    with pytest.raises(EntitiesForbidden):
        list(translation_pairs(path))


def test_lazy_query_restores_model_identity_and_closes_owned_runtime(tmx, tmp_path, monkeypatch):
    import uubed.memory as module

    engine = Encoder()
    engine.backend = "llama.cpp"
    path = tmp_path / "memory.sqlite"
    build_memory([tmx], path, engine, batch_size=1)
    with TranslationMemory(path) as memory, pytest.raises(ValueError, match="model_path"):
        memory.lookup("another font", "pl")
    kwargs_seen = []
    closed = []
    engine.close = lambda: closed.append(True)

    def load(**kwargs):
        kwargs_seen.append(kwargs)
        return engine

    monkeypatch.setattr(module, "Embedder", load)
    with TranslationMemory(path, model_path="relocated.gguf") as memory:
        with pytest.raises(ValueError, match="top_k"):
            memory.lookup("another font", "pl", top_k=0)
        assert memory.lookup("another font", "pl").similar
    assert kwargs_seen[0]["path"] == "relocated.gguf"
    assert kwargs_seen[0]["dimensions"] == 2
    assert kwargs_seen[0]["device"] == "cpu"
    assert closed == [True]


def test_cli_build_publishes_complete_database_and_closes_model(tmx, tmp_path, monkeypatch):
    import json

    from click.testing import CliRunner

    import uubed.tm_cli as module
    from uubed.cli import main

    encoder = Encoder()
    closed = []
    encoder.close = lambda: closed.append(True)
    monkeypatch.setattr(module, "Embedder", lambda *_a, **_kw: encoder)
    output = tmp_path / "built.sqlite"
    result = CliRunner().invoke(main, ["tm", "build", str(tmx), "--output", str(output), "--dimensions", "2"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["sources"] == 2
    assert closed == [True]
    with TranslationMemory(output) as memory:
        assert memory.exact("A cat", "pl")[0].target == "kot"


def test_build_when_space_changes_then_no_published_output(tmx, tmp_path):
    engine = Encoder()

    def changing(texts):
        engine.space = "changed"
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)

    engine.embed = changing
    output = tmp_path / "changed.sqlite"
    with pytest.raises(ValueError, match="space changed"):
        build_memory([tmx], output, engine)
    assert not output.exists()


@pytest.mark.parametrize(
    ("xml", "message"),
    [
        ("<other/>", "not a TMX"),
        ('<tmx><body><tu><tuv xml:lang="en"/></tu></body></tmx>', "missing segment"),
        ("<tmx><body><tu><tuv><seg>text</seg></tuv></tu></body></tmx>", "language code"),
    ],
)
def test_tmx_when_invalid_structure_then_fail_explicitly(tmp_path, xml, message):
    path = tmp_path / "invalid.tmx"
    path.write_text(xml)
    with pytest.raises(ValueError, match=message):
        list(translation_pairs(path))
