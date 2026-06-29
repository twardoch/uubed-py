---
title: API Reference
---

# API Reference

## Core functions

### `encode(embedding, method, **kwargs) -> str`

Encode an embedding vector to a position-safe QuadB64 string.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedding` | `list[int]`, `np.ndarray`, or `bytes` | Input vector. Integer arrays must be `uint8` or values in `[0, 255]`. Float arrays must be in `[0, 1]` (normalised) or `[0, 255]`. |
| `method` | `str` | Encoding method: `"eq64"`, `"shq64"`, `"t8q64"`, `"zoq64"`, `"mq64"`, or `"auto"`. |
| `**kwargs` | | Method-specific parameters (see below). |

**Returns** `str` — encoded string.

**Raises** `UubedValidationError` on invalid input; `UubedEncodingError` on encoding failure.

---

### `decode(encoded, method) -> np.ndarray`

Decode a previously encoded string back to `uint8` values.

Only `eq64` is fully reversible. Other methods produce partial or lossy reconstructions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `encoded` | `str` | Encoded string produced by `encode()`. |
| `method` | `str` | Must match the method used during encoding. |

**Returns** `np.ndarray[uint8]`.

---

## Encoding methods

### Eq64 — lossless QuadB64

```python
encode(emb, method="eq64")
```

Maps every byte to one of four position-dependent 64-character alphabets. Output length is ⌈4/3 × len(emb)⌉ characters.

- **Input** — any `uint8` array, 1 – 100 000 bytes.
- **Output format** — printable ASCII, safe in JSON, Solr, and Elasticsearch keyword fields.
- **Reversible** — yes (`decode(s, method="eq64")` recovers exact bytes).

---

### Shq64 — SimHash QuadB64

```python
encode(emb, method="shq64", planes=64)
```

Projects the embedding onto `planes` random hyperplanes, encodes the resulting bit-vector with QuadB64.

| kwarg | Default | Valid range | Description |
|-------|---------|-------------|-------------|
| `planes` | `64` | 8 – 1024, multiple of 8 | Number of SimHash hyperplanes. Higher = more discriminative but longer output. |

- **Input** — `uint8` array, ≥ 32 bytes.
- **Output length** — `planes / 8` bytes → `⌈4/3 × planes/8⌉` characters (e.g. `planes=64` → 11 chars).
- **Use case** — approximate nearest-neighbour similarity search in full-text engines.

---

### T8q64 — Top-k QuadB64

```python
encode(emb, method="t8q64", k=8)
```

Encodes the indices of the `k` largest values in the embedding.

| kwarg | Default | Valid range | Description |
|-------|---------|-------------|-------------|
| `k` | `8` | 1 – 1000 | Number of top indices to encode. |

- **Input** — `uint8` array, > `k` bytes.
- **Output length** — proportional to `k`.
- **Use case** — sparse embeddings; very high compression when only a few dimensions are significant.

---

### Zoq64 — Z-order QuadB64

```python
encode(emb, method="zoq64")
```

Interleaves the bits of all dimensions along a Z-order (Morton) space-filling curve before QuadB64 encoding.

- **Input** — `uint8` array, ≥ 2 bytes.
- **Output length** — same as `eq64` (no additional compression).
- **Use case** — range queries and spatial locality; nearby vectors in embedding space have similar prefixes.
- **Reversible** — yes (bit interleaving is its own inverse).

---

### Mq64 — Matryoshka QuadB64

```python
encode(emb, method="mq64", levels=None)
```

Encodes successive prefixes of the embedding at multiple granularities, joined by `:`.

| kwarg | Default | Description |
|-------|---------|-------------|
| `levels` | `None` | List of prefix lengths to encode. `None` = powers-of-2 multiples of 64 up to `len(emb)`. |

- **Input** — `uint8` array, any positive length.
- **Output format** — `"level1:level2:level3"` where each segment is an `eq64` string.
- **Use case** — progressive or multi-resolution search (coarse pass on short prefix, refine with longer prefix).
- **Decode** — recovers only the last (most detailed) level.

---

## Streaming API

```python
from uubed import encode_stream, StreamingEncoder

# generator-based — constant memory
for encoded in encode_stream(embeddings, method="shq64"):
    store(encoded)

# context-manager — direct file output
with StreamingEncoder("output.txt", method="t8q64", k=16) as enc:
    for emb in embeddings:
        enc.encode(emb)
```

---

## Exception hierarchy

```
UubedError
├── UubedValidationError    # bad input type/range/shape
├── UubedEncodingError      # encoding operation failed
├── UubedDecodingError      # decoding operation failed
├── UubedResourceError      # memory / file / GPU issue
├── UubedConnectionError    # external service unreachable
└── UubedConfigurationError # misconfigured settings
```

All exceptions carry `.suggestion` (human-readable fix hint) and `.error_code` (machine-readable tag).
