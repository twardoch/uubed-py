---
title: uubed — Position-Safe Embedding Encoding
---

# uubed

**Position-safe QuadB64 encoding for embedding vectors.**

`uubed` solves the *substring pollution* problem: when embedding values are stored as Base64 strings in full-text search engines, arbitrary byte sequences share prefixes and produce false matches. uubed's position-dependent alphabet ensures that two different embeddings never share a common prefix or suffix.

## Quick install

```bash
pip install uubed
```

For optional Rust acceleration (10–100× faster on large batches):

```bash
pip install uubed[native]
```

## Basic usage

```python
import numpy as np
from uubed import encode, decode

emb = np.random.randint(0, 256, 768, dtype=np.uint8)

encoded = encode(emb, method="eq64")   # lossless
decoded = decode(encoded, method="eq64")
assert (decoded == emb).all()
```

## Encoding methods at a glance

| Method | Use case | Reversible | Output size |
|--------|----------|------------|-------------|
| `eq64` | Exact storage / retrieval | Yes | 4/3 × input |
| `shq64` | Similarity search (SimHash) | No | `planes/8` bytes |
| `t8q64` | Sparse top-k indices | No | `k * 3` bytes |
| `zoq64` | Range / spatial queries | No | 4/3 × input |
| `mq64` | Multi-resolution (Matryoshka) | No (last level only) | colon-separated levels |

See [Performance](performance.md) for benchmark numbers and [API Reference](api.md) for full parameter docs.
