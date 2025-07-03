# uubed Example Notebooks

This directory contains Jupyter notebooks demonstrating various features and use cases of the uubed library.

## Notebooks

### 1. [Basic Usage](01_basic_usage.ipynb)
Learn the fundamentals of uubed:
- Encoding and decoding embeddings
- Different encoding methods (eq64, shq64, t8q64, zoq64)
- Working with various input types
- Performance comparisons
- Error handling

### 2. [Streaming API](02_streaming_api.ipynb)
Process large datasets efficiently:
- Streaming from generators
- File-based streaming
- StreamingEncoder context manager
- Batch processing
- Building processing pipelines
- Progress tracking

### 3. [LangChain Integration](03_langchain_integration.ipynb)
Integrate uubed with LangChain for RAG applications:
- Document encoding with metadata
- Embedding model wrappers
- Document processing pipelines
- Vector store integration
- Complete RAG pipeline example
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

3. For the LangChain notebook, install additional dependencies:
```bash
pip install langchain openai chromadb
```

4. Start Jupyter and open a notebook:
```bash
jupyter notebook
```

## Use Cases

- **Exact Search**: Use `eq64` encoding for lossless compression when you need exact matching
- **Similarity Search**: Use `shq64` for compact codes that preserve similarity
- **Sparse Data**: Use `t8q64` for high-dimensional sparse embeddings
- **Spatial Queries**: Use `zoq64` for range queries and spatial locality

## Performance Tips

1. Use the streaming API for datasets that don't fit in memory
2. Batch processing improves throughput significantly
3. Choose encoding method based on your specific use case
4. Consider compression ratio vs. accuracy trade-offs

## Next Steps

- Explore the [API documentation](../src/uubed/api.py)
- Check out the [CLI tool](../src/uubed/cli.py) for command-line usage
- Read about [position-safe encoding](https://github.com/twardoch/uubed) theory