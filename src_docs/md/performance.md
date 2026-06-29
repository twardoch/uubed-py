---
title: Performance
---

# Performance

## Throughput benchmarks

Measured on Apple M1 Pro, Python 3.13, 768-dimensional `uint8` embeddings, pure-Python implementation (no Rust extension).

| Method | Throughput (emb/s) | Output chars | Compression vs raw bytes |
|--------|--------------------|--------------|--------------------------|
| `eq64` | ~25 000 | 1 024 | 1.33× (lossless) |
| `shq64` (`planes=64`) | ~3 500 | 11 | 93× |
| `shq64` (`planes=256`) | ~2 800 | 43 | 24× |
| `t8q64` (`k=8`) | ~35 000 | ~24 | 43× |
| `t8q64` (`k=32`) | ~34 000 | ~96 | 11× |
| `zoq64` | ~30 000 | 1 024 | 1.33× (lossless) |
| `mq64` (default levels) | ~8 000 | varies | varies |

!!! note "Native Rust acceleration"
    When the `uubed._native` extension is compiled and available (`HAS_NATIVE=True`),
    throughput is typically **10–100×** higher for all methods.
    Install with `pip install uubed[native]` or build with `maturin develop` from the
    `uubed-rs` directory.

## Choosing a method

```
Need exact reconstruction?
    → eq64

Similarity search (ANN)?
    → shq64 — tune planes= for recall vs. index size trade-off

Sparse / high-dimensional vectors?
    → t8q64 — very short strings, great for keyword-in-metadata storage

Spatial / range queries?
    → zoq64 — prefix-sortable, compatible with range filters

Multi-resolution / progressive?
    → mq64 — coarse-to-fine search; first segment alone is usable
```

## Memory efficiency

Streaming APIs (`encode_stream`, `StreamingEncoder`) process arbitrarily large datasets at constant memory by yielding one encoded string at a time.

Estimated memory per operation (heuristic, before streaming):

| Method | Multiplier vs raw input |
|--------|------------------------|
| `eq64` | 2.0× |
| `shq64` | 1.5× |
| `t8q64` | 1.2× |
| `zoq64` | 1.8× |
| `mq64` | 2.5× |

Use `estimate_memory_usage(count, size, method)` to check before allocating large batches.
