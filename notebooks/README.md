# uubed Example Notebooks

This directory contains Jupyter notebooks demonstrating uubed library features and use cases.

## Notebooks

### 1. [Basic Usage](01_basic_usage.ipynb)
Covers the fundamentals:
- Encoding and decoding embeddings
- Encoding methods: eq64, shq64, t8q64, zoq64
- Input type handling
- Performance benchmarks
- Error handling

### 2. [Streaming API](02_streaming_api.ipynb)
Process large datasets efficiently:
- Stream from generators
- File-based streaming
- StreamingEncoder context manager
- Batch processing
- Pipeline construction
- Progress monitoring

### 3. [LangChain Integration](03_langchain_integration.ipynb)
Build RAG applications with uubed:
- Document encoding with metadata
- Embedding model wrappers
- Processing pipelines
- Vector store integration
- End-to-end RAG example
- Encoding strategy selection

## Getting Started

1. Install uubed:
```bash
pip install uubed
```

2. Install Jupyter:
```bash
pip install jupyter
```

3. For LangChain notebook, install extras:
```bash
pip install langchain openai chromadb
```

4. Launch Jupyter and open a notebook:
```bash
jupyter notebook
```

## Use Cases

- **Exact Search**: `eq64` for lossless compression with exact matching
- **Similarity Search**: `shq64` for compact codes that preserve similarity
- **Sparse Data**: `t8q64` for high-dimensional sparse embeddings
- **Spatial Queries**: `zoq64` for range queries and spatial locality

## Performance Tips

1. Use streaming API for datasets larger than memory
2. Batch processing boosts throughput
3. Match encoding method to your use case
4. Weigh compression ratio against accuracy

## Next Steps

- Review the [API documentation](../src/uubed/api.py)
- Try the [CLI tool](../src/uubed/cli.py) for command-line operations
- Study [position-safe encoding](https://github.com/twardoch/uubed) theory