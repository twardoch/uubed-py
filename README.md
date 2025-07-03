# uubed-py

High-performance, position-safe embedding encoding for Python.

[![Test](https://github.com/twardoch/uubed/actions/workflows/test.yml/badge.svg)](https://github.com/twardoch/uubed/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/uubed.svg)](https://badge.fury.io/py/uubed)
[![Python versions](https://img.shields.io/pypi/pyversions/uubed.svg)](https://pypi.org/project/uubed/)

## Overview

uubed solves the "substring pollution" problem in embedding-based search systems by using position-dependent alphabets that prevent false matches. This Python package provides a comprehensive toolkit for encoding, processing, and integrating position-safe embeddings.

## Key Features

### 🚀 **High-Performance Encoding**
- **4 encoding methods**: `eq64` (lossless), `shq64` (similarity hash), `t8q64` (top-k sparse), `zoq64` (spatial Z-order)
- **Pure Python fallbacks** with optional native acceleration
- **Automatic method selection** based on use case

### 📊 **Memory-Efficient Streaming**
- **Process datasets larger than RAM** with constant memory usage
- **Generator-based processing** for any iterable source
- **Batch optimization** with configurable sizes
- **File streaming** for direct file-to-file processing

### 🔧 **Production-Ready CLI**
- **Beautiful terminal interface** with Rich formatting
- **Benchmarking tools** across all encoding methods
- **Progress indicators** and performance metrics
- **Piping support** for integration with shell workflows

### 🤖 **Framework Integrations**
- **LangChain**: Complete integration for RAG applications
- **Vector Databases**: Pinecone, Weaviate, Qdrant, ChromaDB connectors
- **GPU Acceleration**: CuPy-based CUDA processing
- **Matryoshka Embeddings**: Multi-granularity progressive refinement

## Installation

```bash
# Core installation
pip install uubed

# With optional dependencies
pip install uubed[gpu]           # GPU acceleration
pip install uubed[langchain]     # LangChain integration  
pip install uubed[vectordb]      # Vector database connectors
pip install uubed[all]           # Everything
```

## Quick Start

### Basic Usage

```python
import numpy as np
from uubed import encode, decode

# Create embedding
embedding = np.random.randint(0, 256, 768, dtype=np.uint8)

# Encode with different methods
lossless = encode(embedding, method="eq64")      # Full precision
compact = encode(embedding, method="shq64")     # Similarity hash
sparse = encode(embedding, method="t8q64", k=8) # Top-k indices
spatial = encode(embedding, method="zoq64")     # Z-order curve

# Decode (only eq64 is reversible)
decoded = decode(lossless, method="eq64")
```

### Streaming Large Datasets

```python
from uubed import encode_stream, StreamingEncoder

# Generator-based streaming
def embedding_generator():
    for i in range(1000000):
        yield np.random.randint(0, 256, 768, dtype=np.uint8)

# Process with constant memory usage
for encoded in encode_stream(embedding_generator(), method="shq64"):
    # Process each encoded embedding
    pass

# Context manager for file output
with StreamingEncoder("output.txt", method="t8q64", k=16) as encoder:
    for embedding in embeddings:
        encoder.encode(embedding)
```

### CLI Usage

```bash
# Encode embeddings
uubed encode embeddings.bin -m shq64 -o encoded.txt

# Decode back to binary
uubed decode encoded.txt -m eq64 -o restored.bin

# Benchmark performance
uubed bench --size 1000 --dims 768 --method all

# Show build info
uubed info
```

### LangChain Integration

```python
from langchain.embeddings import OpenAIEmbeddings
from uubed.integrations.langchain import UubedEmbeddings

# Wrap any LangChain embedding model
base_embeddings = OpenAIEmbeddings()
uubed_embeddings = UubedEmbeddings(
    base_embeddings,
    method="shq64",
    return_encoded=True
)

# Use in RAG pipeline
encoded_docs = uubed_embeddings.embed_documents(["Hello world"])
```

### Vector Database Integration

```python
from uubed.integrations.vectordb import get_connector

# Universal connector interface
connector = get_connector("pinecone", encoding_method="t8q64", k=8)
connector.connect(api_key="your-key", environment="us-east1-gcp")
connector.create_collection("embeddings", dimension=768)

# Automatic encoding injection
vectors = [np.random.randn(768) for _ in range(100)]
connector.insert_vectors(vectors)  # Encoded strings added to metadata
```

### GPU Acceleration

```python
from uubed import gpu_encode_batch, is_gpu_available

if is_gpu_available():
    # Batch GPU processing
    embeddings = [np.random.randint(0, 256, 768, dtype=np.uint8) for _ in range(1000)]
    encoded = gpu_encode_batch(embeddings, method="shq64", planes=64)
    print(f"GPU processed {len(encoded)} embeddings")
```

### Matryoshka Embeddings

```python
from uubed.matryoshka import MatryoshkaEncoder

# Multi-granularity encoding
encoder = MatryoshkaEncoder([64, 128, 256, 512])
embedding = np.random.randn(512)

# Encode at all levels
encoded_levels = encoder.encode_all_levels(embedding)
# Returns: {64: "...", 128: "...", 256: "...", 512: "..."}

# Progressive search
from uubed.matryoshka import MatryoshkaSearchIndex
index = MatryoshkaSearchIndex(encoder)
# Add embeddings and search with progressive refinement
```

## Method Comparison

| Method | Use Case | Compression | Searchable | Reversible |
|--------|----------|-------------|------------|------------|
| `eq64` | Exact matching | 2:1 | ✓ | ✓ |
| `shq64` | Similarity search | 8:1 | ✓ | ✗ |
| `t8q64` | Sparse vectors | 16:1 | ✓ | ✗ |
| `zoq64` | Range queries | 32:1 | ✓ | ✗ |

## Performance

Benchmarks on M1 MacBook Pro (pure Python implementation):

```
Method    Throughput (embeddings/sec)    Compression Ratio
eq64      25,000                          2.0x
shq64     3,500                           8.0x  
t8q64     35,000                          16.0x
zoq64     30,000                          32.0x
```

GPU acceleration provides 5-10x speedup for batch operations.

## Documentation

- 📚 **[API Reference](notebooks/01_basic_usage.ipynb)**: Complete API documentation
- 🚀 **[Streaming Guide](notebooks/02_streaming_api.ipynb)**: Memory-efficient processing
- 🔗 **[Integration Examples](notebooks/03_langchain_integration.ipynb)**: LangChain and vector databases
- 🏗️ **[Architecture](PLAN.md)**: Design and implementation details

## Development

```bash
# Clone repository
git clone https://github.com/twardoch/uubed.git
cd uubed/uubed-py

# Install in development mode
pip install -e .[test,all]

# Run tests
python scripts/test.py

# Build package
python -m build
```

## Contributing

Contributions welcome! Please see our [contributing guidelines](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

If you use uubed in your research, please cite:

```bibtex
@software{uubed2024,
  title={uubed: Position-Safe Embedding Encoding},
  author={Twardoch, Adam},
  year={2024},
  url={https://github.com/twardoch/uubed}
}
```