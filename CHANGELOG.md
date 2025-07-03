# Changelog

All notable changes to the uubed Python package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-07-03 (Current Development)

### Added
- **New Encoding Method (mq64)** - Matryoshka-style multi-level encoding
  - Supports encoding with multiple dimension levels
  - Includes both encoding and decoding support
  - Integrated into the main API
- **Type stub files (.pyi)** for enhanced IDE support and type checking
  - `api.pyi` - Core encoding/decoding function stubs
  - `streaming.pyi` - Streaming API function stubs  
  - `__init__.pyi` - Package-level type definitions
- **Performance regression testing framework** with comprehensive benchmarking
  - `scripts/benchmark.py` - Automated performance testing across encoding methods
  - Memory profiling for large datasets and streaming operations
  - Performance comparison with baseline results
  - Throughput and memory usage metrics collection
- **Comprehensive Error Handling System** with custom exception hierarchy
  - `UubedValidationError` - Input validation failures with detailed context
  - `UubedEncodingError` - Encoding operation failures 
  - `UubedDecodingError` - Decoding operation failures
  - `UubedResourceError` - Resource management failures (memory, files, GPU)
  - `UubedConnectionError` - External service connection failures
  - `UubedConfigurationError` - Configuration and setup issues
- **Advanced Input Validation** for all public APIs
  - Type checking with automatic conversion (float to uint8)
  - Range validation for embedding values and method parameters
  - Memory usage estimation and limits
  - File path validation with permission checks
  - Method-specific parameter validation
- **Configuration System** with environment and file-based settings
  - `config.py` - Configuration management with defaults
  - TOML/YAML/JSON configuration file support
  - Environment variable overrides
  - Method-specific parameter defaults

### Modified
- **API Enhancements**
  - Added comprehensive error handling to `encode()` and `decode()` functions
  - Improved auto-detection logic for encoding methods
  - Enhanced memory usage validation for large embeddings
  - Better integration with configuration system for default parameters
- **Native Wrapper Updates**
  - Added mq64 encoding/decoding function imports
  - Updated method availability checks

### Fixed
- **Build system configuration** migrated from maturin to setuptools for pure Python distribution
  - Updated `pyproject.toml` to use setuptools-scm for version management
  - Removed Rust/maturin dependency until native extensions are implemented
  - Fixed version configuration for proper package building
- **Package import and basic functionality verification** confirmed working
- **Ruff configuration** added to pyproject.toml for linting and formatting

### Enhanced
- **Development workflow** improvements for better maintainability
- **Documentation accuracy** with implementation status verification
- **Error messages** now include specific parameter details and helpful suggestions

## [0.2.0] - 2024-01-XX (In Development)

### Added

#### Core CLI Tool
- **Full-featured CLI interface** with `click` and `rich` for beautiful terminal output
  - `uubed encode` - Encode binary files with multiple encoding methods
  - `uubed decode` - Decode encoded strings back to binary (eq64 only)
  - `uubed bench` - Performance benchmarking across all encoding methods
  - `uubed info` - Display build information and feature flags
- **Progress indicators** for encoding/decoding operations (hidden when outputting to stdout)
- **Method-specific parameters** support (k for t8q64, planes for shq64)
- **Automatic method selection** based on input characteristics

#### Streaming API
- **Memory-efficient streaming** for processing large datasets
  - `encode_stream()` - Generator-based encoding from any iterable
  - `encode_file_stream()` - Direct file-to-file streaming with progress
  - `decode_stream()` - Stream decoding for encoded strings
  - `batch_encode()` - Batch processing with automatic parallelization
- **StreamingEncoder context manager** for automatic resource handling
- **Configurable batch sizes** for optimal memory/performance balance

#### LangChain Integration
- **Complete LangChain ecosystem support**
  - `UubedEncoder` - Document transformer adding encoded embeddings to metadata
  - `UubedEmbeddings` - Wrapper for any LangChain embedding model
  - `UubedDocumentProcessor` - Batch document processing pipeline
  - `create_uubed_retriever()` - Enhanced retriever with uubed encoding
- **Automatic embedding normalization** from float ranges to uint8
- **Async support** for all embedding operations
- **Custom encoding strategies** based on use case (exact_match, similarity_search, sparse, spatial)

#### Vector Database Connectors
- **Universal vector database support** with consistent API
  - **Pinecone connector** - Encoding in metadata with batch upserts
  - **Weaviate connector** - Property-based encoding with class management
  - **Qdrant connector** - Payload-based encoding with point management
  - **ChromaDB connector** - Metadata encoding with collection management
- **Factory pattern** `get_connector()` for easy instantiation
- **Automatic encoding injection** into database-specific metadata formats
- **Batch operations** optimized for each database's API

#### GPU Acceleration
- **CuPy-based GPU acceleration** with automatic CPU fallback
  - `GPUEncoder` - Batch GPU processing for multiple encoding methods
  - `gpu_encode_batch()` - High-throughput batch encoding
  - `GPUStreamingEncoder` - GPU-accelerated streaming for massive datasets
  - `benchmark_gpu_vs_cpu()` - Performance comparison utilities
- **Memory-efficient GPU operations** with automatic data transfer
- **Graceful degradation** when CUDA/CuPy not available
- **Device selection** and GPU info reporting

#### Matryoshka Embedding Support
- **Multi-granularity embedding encoding** for progressive refinement
  - `MatryoshkaEncoder` - Level-specific encoding with adaptive methods
  - `MatryoshkaSearchIndex` - Progressive search with dimension refinement
  - `create_adaptive_matryoshka_encoder()` - Automatic level configuration
- **Flexible dimension strategies** (linear, exponential, powers_of_2)
- **Level-specific encoding methods** optimized for each dimension range
- **Progressive search** starting from low dimensions and refining upward

#### Documentation & Examples
- **Comprehensive Jupyter notebooks**
  - `01_basic_usage.ipynb` - Core API usage and method comparison
  - `02_streaming_api.ipynb` - Large dataset processing examples
  - `03_langchain_integration.ipynb` - RAG pipeline integration
- **Detailed API documentation** with examples and use cases
- **Performance benchmarking** across all encoding methods
- **Integration guides** for popular frameworks

### Enhanced
- **Comprehensive error handling system** with custom exception hierarchy
  - `UubedValidationError` - Input validation failures with parameter details
  - `UubedEncodingError` - Encoding operation failures with method context
  - `UubedDecodingError` - Decoding operation failures with string context
  - `UubedResourceError` - Resource management failures (memory, files, GPU)
  - `UubedConnectionError` - External service connection failures
  - `UubedConfigurationError` - Configuration and setup issues
- **Advanced input validation** for all public APIs
  - Type checking with automatic conversion (float to uint8, etc.)
  - Range validation for embedding values and method parameters
  - Memory usage estimation and limits
  - File path validation with permission checks
  - Method-specific parameter validation (k for t8q64, planes for shq64)
- **Enhanced error messages** with user guidance and suggestions
  - Context-aware error descriptions with parameter details
  - Specific suggestions for fixing common issues
  - Error codes for programmatic handling
  - Chain preservation for debugging
- **Type hints** throughout the codebase for better IDE support
- **Graceful fallback** for optional dependencies (CuPy, LangChain, vector DBs)
- **Configurable encoding parameters** for all methods
- **Progress tracking** with optional tqdm integration

### Fixed
- **Encoding method consistency** across all input types (bytes, lists, numpy arrays)
- **Memory efficiency** for large embedding processing
- **CLI output handling** to avoid progress indicators in piped output
- **Type conversion** for different embedding formats (float to uint8)

#### Development Infrastructure
- **Comprehensive test suite** with 4 test modules covering all functionality
  - `tests/test_api.py` - Core API functionality tests
  - `tests/test_streaming.py` - Streaming API tests  
  - `tests/test_cli.py` - CLI command tests
  - `tests/test_integrations.py` - Integration tests for all frameworks
- **CI/CD pipeline** with GitHub Actions for automated testing
  - Multi-platform testing (Ubuntu, macOS, Windows)
  - Python version matrix (3.10, 3.11, 3.12)
  - Optional dependency testing and CLI verification
- **Release automation** with version management and PyPI preparation
  - `scripts/release.py` - Automated release preparation
  - `scripts/test.py` - Unified test runner
  - Version synchronization across pyproject.toml and __version__.py
- **Complete project documentation**
  - Comprehensive README.md with examples and benchmarks
  - Detailed PLAN.md with implementation status and roadmap
  - CHANGELOG.md tracking all changes and additions

### Fixed
- **Encoding method consistency** across all input types (bytes, lists, numpy arrays)
- **Memory efficiency** for large embedding processing
- **CLI output handling** to avoid progress indicators in piped output
- **Type conversion** for different embedding formats (float to uint8)

### Added

#### Distribution & Packaging
- **Binary Wheel Building**: Implemented maturin-based build system for wheels with native extensions
- **PyPI Upload**: Automated PyPI uploading process
- **Performance Regression Tests**: Automated benchmarking to catch performance regressions
- **Memory Profiling**: Automated memory usage testing for streaming operations
- **Completed PyPI Distribution System**: All tasks related to PyPI distribution are now complete.
- **Cleaned up TODO.md and PLAN.md**: Removed completed tasks from `TODO.md` and `PLAN.md` to reflect the updated status.
- **Final Cleanup**: Ensured `TODO.md` and `PLAN.md` are correctly formatted and reflect only pending tasks.

### Dependencies
- **Core dependencies**: `numpy>=1.20`, `click>=8.0`, `rich>=10.0`
- **Optional dependencies**: 
  - `cupy-cuda11x` for GPU acceleration
  - `langchain` for LangChain integration
  - `pinecone-client`, `weaviate-client`, `qdrant-client`, `chromadb` for vector databases
- **Development dependencies**: `pytest>=7.0`, `pytest-cov>=4.0` for testing

## [0.1.0] - 2024-01-XX

### Added
- Initial Python package structure
- Pure Python encoder implementations (eq64, shq64, t8q64, zoq64)
- Basic encoding/decoding API
- PyO3 bindings preparation
- Package configuration with maturin

### Notes
- This version established the foundation for the Python package
- Pure Python implementations serve as fallbacks when native extensions aren't available
- All subsequent features build upon this stable base