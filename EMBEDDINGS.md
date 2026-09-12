---
this_file: EMBEDDINGS.md
---

# Compact multilingual embeddings

`uubed.embeddings.Embedder` generates vectors with a resident local model or an
HTTP provider. `uubed.compact` stores their normalized direction in a portable
ASCII record. `uubed.infile` inserts that record into UTF-8 source files.

## Install

From this checkout:

```bash
uv pip install -e '.[inference]'
# Optional verified Apple Silicon MLX backends for MiniLM and Granite:
uv pip install -e '.[inference,mlx]'
# Optional ONNX CPU backend:
uv pip install -e '.[inference,onnx]'
```

The tested combination uses Transformers 4.57.6, SentenceTransformers 5.7.0,
llama-cpp-python 0.3.35 and NumPy. Transformers 5 breaks the pinned Voyage nano
custom architecture. The MLX extra uses mlx-embeddings 0.0.5 / mlx-vlm 0.3.9 so
it can coexist with the correct Voyage runtime. No model is downloaded on import.

## Model contracts

| Alias | Native size | Supported retained dimensions | Local choices | API choices |
| --- | ---: | --- | --- | --- |
| `jina` | 768 | 32, 64, 128, 256, 512, 768 | llama.cpp, SentenceTransformers | Jina, explicit OpenAI endpoint |
| `gemma` | 768 | 128, 256, 512, 768 | llama.cpp, SentenceTransformers | Cloudflare, explicit OpenAI endpoint |
| `granite` | 768 | 128, 256, 384, 512, 768 | llama.cpp, SentenceTransformers, MLX | HF, explicit OpenAI endpoint |
| `nomic` | 768 | 256, 512, 768 | llama.cpp, SentenceTransformers | explicit OpenAI endpoint |
| `minilm` | 384 | 384 only | SentenceTransformers, MLX, ONNX | — |
| `voyage-lite` | 2048 requested | 256, 512, 1024, 2048 | — | Voyage |
| `voyage-nano` | 2048 | 256, 512, 1024, 2048 | SentenceTransformers | — |

MiniLM is an English baseline. Other aliases refer to the exact multilingual
models in `uubed.models`, with pinned Hugging Face weight revisions. Nomic also
pins its separately hosted model/config code. GGUF files are fingerprinted with
SHA-256. Use a verified conversion: EmbeddingGemma requires its learned dense
layers, and Voyage nano is bidirectional with a learned output projection.

## Local inference

```python
from uubed.embeddings import Embedder

engine = Embedder("jina", backend="llama.cpp", path="/models/jina-text-matching.gguf",
                  dimensions=256, precision="int8", batch_size=32)
try:
    records = engine.pack(["The font contains alternate glyphs."])
finally:
    engine.close()
```

Omit `path` with `backend="sentence-transformers"` or `"mlx"`. Use `device="mps"`
for PyTorch on Apple Silicon and `device="cpu"` for CPU/ONNX. MLX and llama.cpp
use Metal. Create one instance and reuse it; calls to a shared instance are
serialized because native model state is not reentrant. MLX and PyTorch MPS also
share a process-wide lock: parallel native calls across instances caused a Metal
abort in testing. `close()` frees the resident runtime. Separate llama.cpp
instances can be measured against batching, but are not assumed faster.

`embed(texts)` returns normalized float32 rows; `pack(texts)` also quantizes.
Local `max_tokens` defaults to 512 and is capped by the model context. MiniLM
caps at 256, Nomic at 512. The llama.cpp physical token budget defaults to 4096
and splits batches before that total is exceeded. API context limits are enforced
by the provider; local token clipping does not apply to remote calls.

`task="similarity"` applies the vendor's symmetric prefix; `query` and `document`
apply the paired retrieval prefixes. They share a comparison space but have
separate cache keys. Custom prefixes are supported for similarity, including
vexy-paraltext's explicit Jina `Query: ` override. Ambiguous custom retrieval
pairs and Jina/Voyage API prefix overrides are rejected.

## APIs

```python
engine = Embedder("voyage-lite", backend="voyage", dimensions=256)
```

Use `JINA_API_KEY`, `VOYAGE_API_KEY`, `HF_TOKEN`, or
`CLOUDFLARE_API_TOKEN` plus `CLOUDFLARE_ACCOUNT_ID`, as appropriate. Alternatively
pass `api_key`. Keys are not embedded in records. Calls reuse one HTTP client,
have a 60-second timeout, reject bad/missing/duplicate rows and never switch
providers after failure. Configure `timeout` and `batch_size` when needed.

For a self-hosted exact model, use `backend="openai"` and an explicit full URL
such as `endpoint="http://127.0.0.1:18301/v1/embeddings"`. The request names the
exact model ID. A llama-cpp-python server can serve the same GGUF models; the
workspace includes a tested four-model server configuration and parity check.
There is no verified Nomic v2 vendor serverless API, so its tested API route is
self-hosted. Jina/Voyage vendor calls passed. Cloudflare/HF routes are schema-tested
but the available accounts returned 401/402; their live success is not claimed.

Hosted weight revisions are marked `unversioned-hosted`. Providers can update
weights behind a stable name: invalidate persistent caches after such updates.
Local and API backends intentionally have distinct vector-space identities.

## Compact records and original files

```python
from pathlib import Path
from uubed.infile import write_embedding, read_embedding, remove_embedding

path = Path("document.md")
write_embedding(path, records[0])
stored = read_embedding(path)
remove_embedding(path)  # restores source bytes exactly
```

UTF-8 `.txt`, `.md`, `.markdown`, `.xml` and `.tmx` are supported. Binary formats,
symlink writes, invalid XML and entity declarations are rejected. An XML processing
instruction trailer carries the record and a hash of the original source bytes.
Writes replace atomically and preserve file permissions. Reads reject stale
content; `expected_sha256` rejects edits made between inference and writing.

UB1 truncates, renormalizes, then stores float32, symmetric int8, symmetric int4,
or packed sign bits. At 256 dimensions, payloads use 1024/256/128/32 bytes,
respectively. ASCII base64url and metadata add overhead. CRC32 detects accidental
corruption. `PackedEmbedding.loads(record)` validates metadata, size and checksum;
`similarity(a, b)` compares reconstructed directions and rejects mixed spaces or
sizes. Binary and scalar quantization are approximate; they do not preserve exact
float32 scores or all alignment decisions.

The original `uubed.encode` / `decode` byte encoders remain compatible. Their
legacy MQ64 formats are distinct from UB1 and must not be substituted for it.
