# this_file: tests/test_fastembed.py
"""FastEmbed adapter contracts; real-model smoke tests live in examples/."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from uubed.embeddings import Embedder


@pytest.mark.parametrize("alias", ["gemma", "minilm", "granite", "jina", "nomic", "voyage-nano"])
def test_fastembed_when_supported_then_pinned_resident_and_prompted(alias, monkeypatch, tmp_path):
    calls = []
    downloads = []
    engine = Embedder(alias, backend="fastembed", device="cpu")

    class Native:
        def __init__(self, *args):
            calls.append(args)

        def embed(self, texts, batch_size):
            calls.append((texts, batch_size))
            return np.ones((len(texts), engine.model.dimensions)).tolist()

        def close(self):
            calls.append("closed")

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "uubed_native", SimpleNamespace(NativeEmbedder=Native))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=download,
            constants=SimpleNamespace(HF_HUB_OFFLINE=True),
        ),
    )
    before = engine.space
    assert engine.embed(["Hello"]).shape == (1, engine.dimensions)
    engine.embed(["Again"])
    assert calls[1][0] == [engine.prefix + "Hello"]
    assert len(downloads) == 1
    assert len(downloads[0]["revision"]) == 40
    assert downloads[0]["local_files_only"] is True, "offline mode must never request remote metadata"
    assert before == engine.space
    assert "fastembed-6.0.3" in before
    assert before != Embedder(alias).space
    engine.close()
    assert calls[-1] == "closed"


@pytest.mark.parametrize("alias", ["voyage-lite"])
def test_fastembed_when_unverified_model_then_explicit_error(alias):
    with pytest.raises(ValueError, match="FastEmbed"):
        Embedder(alias, backend="fastembed", device="cpu")


def test_fastembed_when_wrong_device_then_no_silent_cpu_fallback():
    with pytest.raises(ValueError, match="cpu"):
        Embedder("gemma", backend="fastembed", device="mps")


def test_fastembed_when_native_wheel_is_old_then_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "uubed_native", SimpleNamespace())
    with pytest.raises(ImportError, match=r"uubed\[fastembed\].*NativeEmbedder"):
        Embedder("gemma", backend="fastembed", device="cpu").embed(["Hello"])


def test_fastembed_when_download_fails_then_error_propagates(monkeypatch):
    def download(**_kwargs):
        msg = "download unavailable"
        raise OSError(msg)

    monkeypatch.setitem(sys.modules, "uubed_native", SimpleNamespace(NativeEmbedder=object))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=download,
            constants=SimpleNamespace(HF_HUB_OFFLINE=False),
        ),
    )
    with pytest.raises(OSError, match="download unavailable"):
        Embedder("gemma", backend="fastembed", device="cpu").embed(["Hello"])


@pytest.mark.parametrize("alias", ["gemma", "minilm", "granite", "jina", "nomic", "voyage-nano"])
def test_fastembed_when_retrieval_then_shared_space_and_distinct_prompts(alias):
    query = Embedder(alias, backend="fastembed", device="cpu", task="query")
    document = Embedder(alias, backend="fastembed", device="cpu", task="document")
    assert query.space == document.space
    assert query.cache_identity != document.cache_identity
    assert query.prefix == query.model.prefix("query")
    assert document.prefix == document.model.prefix("document")
