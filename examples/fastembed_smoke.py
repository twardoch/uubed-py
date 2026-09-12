# this_file: examples/fastembed_smoke.py
# ruff: noqa: S101, T201, EM101, PLR2004, PT017
"""Run real CPU models: python examples/fastembed_smoke.py [--reference].

Downloads pinned weights on first use. --reference also requires the inference
extra and compares against the original SentenceTransformers model on CPU.
"""

import argparse
import json
import time

import numpy as np

from uubed.embeddings import Embedder
from uubed.fastembed import EXPORTS

TEXTS = [
    "The cat sleeps on the sofa.",
    "A kitten is resting on a couch.",
    "The database was deleted.",
    "Zażółć gęślą jaźń. Ten krój zawiera polskie znaki.",
    "Diese Schrift enthält alternative Zeichen.",
    "このフォントには代替字形が含まれています。",
]


def verify(alias: str, *, reference: bool) -> dict:
    start = time.monotonic()
    engine = Embedder(alias, backend="fastembed", device="cpu", max_tokens=64)
    try:
        vectors = engine.embed(TEXTS)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(vectors, engine.embed(TEXTS), atol=1e-6)
        single = np.concatenate([engine.embed([text]) for text in TEXTS])
        cosine = np.sum(vectors * single, axis=1)
        assert cosine.min() > 0.9999, f"{alias}: batch composition changed the vectors"
        assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2], "paraphrase must outrank distractor"
        assert engine.embed([]).shape == (0, engine.dimensions)
        packed = engine.pack(TEXTS[:1])[0]
        assert packed.dimensions == engine.dimensions
        # Exercise the native ownership boundary, not just the Python facade.
        native = engine._runtime.model
        assert native.embed([], 1) == []
        for texts, batch in [([" "], 1), (["test"], 0)]:
            try:
                native.embed(texts, batch)
            except ValueError:
                pass
            else:
                raise AssertionError("native input validation missing")
        engine.close()
        try:
            native.embed(["test"], 1)
        except ValueError as error:
            assert "closed" in str(error)
        else:
            raise AssertionError("closed native session was reused")
        np.testing.assert_allclose(vectors, engine.embed(TEXTS), atol=1e-6)
        result = {"model": alias, "shape": list(vectors.shape), "batch_cosine_min": float(cosine.min())}
    finally:
        engine.close()
    if reference:
        reference_engine = Embedder(alias, device="cpu", max_tokens=64)
        try:
            expected = reference_engine.embed(TEXTS)
            cosine = np.sum(vectors * expected, axis=1)
            result["reference_cosines"] = cosine.tolist()
            # Jina's vendor ONNX export differs slightly from its Torch weights.
            threshold = 0.995 if alias == "jina" else 0.999
            assert cosine.min() > threshold, f"{alias}: reference parity failed: {cosine}"
        finally:
            reference_engine.close()
    result["seconds"] = time.monotonic() - start
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--models", nargs="+", default=[export.alias for export in EXPORTS.values()])
    args = parser.parse_args()
    for name in args.models:
        print(json.dumps(verify(name, reference=args.reference)), flush=True)
