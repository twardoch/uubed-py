# this_file: tests/test_embeddings.py
"""Inference contracts: persistence, task handling, dimensions and cache identity."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from uubed.embeddings import Embedder


class FakeLlama:
    instances = []  # noqa: RUF012 - intentional test capture

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.instances.append(self)

    def tokenize(self, data, add_bos=True):  # noqa: FBT002 - mirrors llama.cpp
        return list(data) + ([0, 0] if add_bos else [])

    def detokenize(self, tokens):
        return bytes(tokens)

    def embed(self, texts, **_kwargs):
        self.calls.append(list(texts))
        assert sum(len(self.tokenize(t.encode())) for t in texts) <= self.kwargs["n_ubatch"]
        return [np.arange(1, 769, dtype=np.float32).tolist() for _ in texts]

    def close(self):
        pass


def test_embed_when_llama_then_reuses_model_and_budgets_tokens(tmp_path, monkeypatch):
    FakeLlama.instances = []
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    path = tmp_path / "weights.gguf"
    path.write_bytes(b"weights")
    engine = Embedder("jina", backend="llama.cpp", path=path, dimensions=128, max_tokens=32, batch_tokens=64)
    result = engine.embed(["hello", "world", "a" * 500, "test"])
    engine.embed(["again"])
    assert result.shape == (4, 128)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)
    assert len(FakeLlama.instances) == 1
    assert FakeLlama.instances[0].calls[0][0] == "Document: hello"
    assert FakeLlama.instances[0].kwargs["pooling_type"] == 3
    assert engine.embed([]).shape == (0, 128)
    assert engine.pack(["hello"])[0].dimensions == 128


def test_identity_when_configuration_changes_then_no_cache_reuse(tmp_path):
    path = tmp_path / "weights.gguf"
    path.write_bytes(b"first")
    kwargs = {"backend": "llama.cpp", "path": path}
    first = Embedder("jina", **kwargs).cache_identity
    assert first != Embedder("jina", dimensions=128, **kwargs).cache_identity
    assert first != Embedder("jina", prefix="Query: ", **kwargs).cache_identity
    assert first != Embedder("jina", precision="binary", **kwargs).cache_identity
    path.write_bytes(b"other")
    assert first != Embedder("jina", **kwargs).cache_identity


def test_embed_when_sentence_transformers_then_explicit_prompt_and_revision(monkeypatch):
    calls = []

    class FakeST:
        def __init__(self, name, **kwargs):
            calls.append((name, kwargs))

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), 384))

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeST))
    engine = Embedder("minilm", backend="sentence-transformers", device="cpu")
    assert engine.embed(["Hello"]).shape == (1, 384)
    engine.embed(["Again"])
    assert len(calls) == 3
    assert calls[0][1]["revision"] == engine.model.revision
    assert calls[1][1]["prompt"] == ""


@pytest.mark.parametrize(
    "kwargs", [{"dimensions": 0}, {"batch_tokens": 0}, {"max_tokens": 0}, {"precision": "bad"}, {"backend": "invented"}]
)
def test_embedder_when_invalid_config_then_fail_before_load(kwargs):
    with pytest.raises(ValueError, match=r".+"):
        Embedder("jina", **kwargs)


def test_embed_when_bad_runtime_output_then_no_silent_vectors(monkeypatch):
    class BadST:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **_kwargs):
            return np.zeros((len(texts), 768))

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=BadST))
    with pytest.raises(ValueError, match="dimensions"):
        Embedder("minilm").embed(["x"])
    with pytest.raises(ValueError, match="nonzero"):
        Embedder("jina").embed(["x"])


def test_embed_when_invalid_text_then_rejected_without_loading():
    for texts in ["text", [None], [""], [123]]:
        with pytest.raises(ValueError, match="texts"):
            Embedder("minilm").embed(texts)


def test_space_when_query_and_document_then_comparable_but_separate_cache_keys():
    query = Embedder("nomic", task="query")
    document = Embedder("nomic", task="document")
    assert query.space == document.space
    assert query.cache_identity != document.cache_identity


def test_identity_when_weights_change_on_same_instance_then_reject(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    path = tmp_path / "weights.gguf"
    path.write_bytes(b"first")
    engine = Embedder("jina", backend="llama.cpp", path=path)
    identity = engine.cache_identity
    path.write_bytes(b"second weights")
    with pytest.raises(ValueError, match="weights changed"):
        engine.embed(["hello"])
    with pytest.raises(ValueError, match="weights changed"):
        _ = engine.cache_identity
    assert identity


def test_identity_when_loaded_weights_change_after_close_then_reject(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    path = tmp_path / "weights.gguf"
    path.write_bytes(b"first")
    engine = Embedder("jina", backend="llama.cpp", path=path)
    engine.embed(["hello"])
    engine.close()
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="weights changed"):
        engine.embed(["hello"])


def test_space_when_custom_retrieval_prompt_then_reject_ambiguous_pair():
    with pytest.raises(ValueError, match="retrieval prompt"):
        Embedder("nomic", task="query", prefix="custom")


def test_space_when_hosted_model_then_does_not_claim_hf_revision():
    engine = Embedder("jina", backend="jina")
    assert engine.model.revision not in engine.space
    assert "unversioned-hosted" in engine.space


@pytest.mark.parametrize(("model", "backend"), [("minilm", "jina"), ("voyage-lite", "sentence-transformers")])
def test_embedder_when_wrong_model_provider_then_fail_before_load(model, backend):
    with pytest.raises(ValueError, match=r"does not serve|API-only"):
        Embedder(model, backend=backend)


def test_embed_when_nomic_then_pins_external_model_and_config_code(monkeypatch):
    calls = []

    class FakeST:
        def __init__(self, _name, **kwargs):
            calls.append(kwargs)

        def encode(self, texts, **_kwargs):
            return np.ones((len(texts), 768))

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeST))
    engine = Embedder("nomic")
    engine.embed(["A font family"])
    expected = "7710840340a098cfb869c4f65e87cf2b1b70caca"
    assert calls[0]["model_kwargs"]["code_revision"] == expected
    assert calls[0]["config_kwargs"]["code_revision"] == expected
    assert expected in engine.space


def test_embed_when_mlx_minilm_then_pinned_resident_batched_model(monkeypatch):
    calls = []

    class Runtime:
        def __call__(self, input_ids, **_kwargs):
            calls.append(len(input_ids))
            return SimpleNamespace(text_embeds=np.ones((len(input_ids), 384), dtype=np.float32))

    def load(path):
        assert path == "pinned"
        return Runtime(), SimpleNamespace(
            batch_encode_plus=lambda texts, **_kw: {"input_ids": np.ones((len(texts), 3), dtype=np.int32)}
        )

    def get_path(name, revision):
        calls.append((name, revision))
        return "pinned"

    mx = SimpleNamespace(array=np.asarray, eval=lambda *_args: None, float32=np.float32, clear_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx_embeddings", SimpleNamespace(load=load))
    monkeypatch.setitem(sys.modules, "mlx_embeddings.utils", SimpleNamespace(get_model_path=get_path))
    engine = Embedder("minilm", backend="mlx", batch_size=2)
    result = engine.embed(["one", "two", "three"])
    engine.embed(["four"])
    assert result.shape == (3, 384)
    assert calls == [(engine.model.model_id, engine.model.revision), 2, 1, 1]
    engine.close()


def test_embedder_when_unverified_mlx_model_then_explicit_error():
    with pytest.raises(ValueError, match="MLX backend supports"):
        Embedder("voyage-nano", backend="mlx")


def test_embed_when_onnx_then_explicit_cpu_provider(monkeypatch):
    calls = []

    class FakeST:
        def __init__(self, _name, **kwargs):
            calls.append(kwargs)

        def encode(self, texts, **_kwargs):
            return np.ones((len(texts), 384))

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeST))
    Embedder("minilm", backend="onnx", device="cpu").embed(["A font family"])
    assert calls[0]["model_kwargs"]["provider"] == "CPUExecutionProvider"


def test_embedder_when_mlx_granite_then_accept_verified_model():
    engine = Embedder("granite", backend="mlx", dimensions=256)
    assert engine.dimensions == 256


def test_embedder_when_onnx_mps_then_reject_wrong_execution_device():
    with pytest.raises(ValueError, match=r"ONNX.*cpu"):
        Embedder("minilm", backend="onnx", device="mps")


def test_embedder_when_non_gguf_backend_has_path_then_reject_unused_weight_identity(tmp_path):
    with pytest.raises(ValueError, match=r"path.*llama\.cpp"):
        Embedder("minilm", path=tmp_path / "unrelated.gguf")


@pytest.mark.parametrize("backend", ["mlx", "sentence-transformers"])
@pytest.mark.parametrize("devices", [("mps", "mps"), ("mps", "mps:0")])
def test_embed_when_two_python_metal_instances_then_no_concurrent_native_calls(backend, devices, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock
    from time import sleep

    active = peak = 0
    lock = Lock()
    start = Barrier(2)

    class Runtime:
        def encode(self, texts, *_args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            sleep(0.03)
            with lock:
                active -= 1
            return np.ones((len(texts), 384))

    monkeypatch.setattr(Embedder, "_load", lambda _self: Runtime())
    engines = [Embedder("minilm", backend=backend, device=device) for device in devices]

    def run(engine):
        start.wait(timeout=5)
        return engine.embed(["font family"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, engines))
    assert all(result.shape == (1, 384) for result in results)
    assert peak == 1, "MLX/MPS runtime entry must be serialized across independent instances"


def test_embed_when_nonfinite_discarded_tail_then_reject_native_output(monkeypatch):
    rows = np.ones((1, 768), dtype=np.float32)
    rows[0, -1] = np.nan
    monkeypatch.setattr(Embedder, "_load", lambda _self: SimpleNamespace(encode=lambda *_args: rows))
    with pytest.raises(ValueError, match="finite"):
        Embedder("jina", dimensions=256).embed(["font family"])
