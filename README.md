---
this_file: README.md
---
# Uubed Python

Shared model inference, compact embedding records and English-source translation
memory for book alignment and localization.

## Translation memory

Install this checkout together with its native `uubed-rs` wheel. The `[tm]` extra
requires that distribution; add `[inference]` for local embedding models. See the
[workspace setup](../../README.md) for editable installs before publication.

```python
from pathlib import Path
from uubed.embeddings import Embedder
from uubed.memory import TranslationMemory, build_memory

engine = Embedder("jina", dimensions=256, device="cpu")
try:
    build_memory([Path("translations.tmx")], Path("localization.sqlite"), engine)
    with TranslationMemory("localization.sqlite", embedder=engine) as memory:
        exact = memory.exact("Bold", "pl")
        references = memory.lookup("Make the font bold", "pl", top_k=5)
finally:
    engine.close()
```

The importer streams TMX, finds English by language tag, and embeds each distinct
source once. SQLite stores all target variants and path/unit provenance, including
conflicting translations. Exact lookup preserves case and whitespace and loads no
model. Semantic lookup filters by target language and returns bounded cosine-ranked
pairs. Model/weights/task/prefix/truncation/dimension mismatches are rejected.
The index uses ordinary SQLite and signed int8 vectors, without a vector extension.
At 256 dimensions, each vector payload occupies 256 bytes; texts and metadata add
storage. Build into a new filename to change inputs or model settings.

```bash
uubed tm build translations.tmx --output localization.sqlite --model jina --dimensions 256
uubed tm lookup localization.sqlite 'Bold' --language pl --exact-only
uubed tm lookup localization.sqlite 'Make the font bold' --language pl
```

For a GGUF-backed index, pass `--backend llama.cpp --model-path /path/to/model.gguf`
at build time and the matching `--model-path` at semantic lookup time.
The index can move independently of the weight file.

## Book embeddings

[EMBEDDINGS.md](EMBEDDINGS.md) describes the shared `Embedder`, verified model
profiles, resident runtimes, Matryoshka truncation and UB1 int8/int4/binary records.
Vexy Paraltext uses these APIs for its book alignment workflow and TMX output.
TM indexing uses the same inference and signed int8 quantization code.

## Byte codecs

The existing `encode`/`decode` APIs retain `eq64`, `shq64`, `t8q64`, `zoq64`, and
`mq64`. They encode bytes and are separate from model-aware signed UB1 vectors.
The optional native module is named `uubed_native`; Python fallbacks remain
available for byte encoding. A textual lossless encoding expands bytes; it is not
an embedding compression algorithm.

```python
from uubed import encode, decode
encoded = encode(bytes(range(256)), method="eq64")
decoded = decode(encoded, method="eq64")
```

## Development

`../../test.sh` builds the actual native wheel, installs it into the test
environment, and verifies this package and both consumers. `uvx hatch test -py 3.12`
runs this package's suite when its native TM dependency is installed.
Build distributions with `uv build`. MIT license.

## Releases and local data

`./publish.sh --dry-run` verifies the next release without pushing or uploading.
`./publish.sh` commits, tags and publishes it. See [RELEASING.md](RELEASING.md)
for credentials, same-tag retries, dependency order and private-data exclusions.
