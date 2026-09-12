---
this_file: DEPENDENCIES.md
---

# Embedding dependencies

- NumPy: signed-vector normalization, quantization and cosine reconstruction.
- httpx: persistent HTTP connections, deadlines and provider errors (`inference`).
- llama-cpp-python >=0.3.35: GGUF embedding inference with Metal and explicit pooling.
- SentenceTransformers >=5.2,<6 and Transformers >=4.57.6,<5: complete vendor
  pooling/projection pipelines. Transformers 5 breaks pinned Voyage nano code.
- einops: required by Nomic's custom architecture; CUDA-only MegaBlocks is not
  required for correctness on M4 and was not installed as an Apple optimization.
- mlx-embeddings 0.0.5 and mlx-vlm 0.3.9: optional verified MiniLM/Granite MLX
  inference; compatible with the Transformers 4 stack. No custom transformer code.
- sentence-transformers[onnx]: optional ONNX provider. CPU is explicit to avoid
  automatically selecting the much slower CoreML partitioning observed here.
- Hatchling/hatch-vcs: existing version and wheel/sdist build system.

Existing CLI/encoding dependencies are unchanged. See pyproject.toml and uv.lock
for the resolved dependency graph; embedding model licenses are separate.

Translation memory uses defusedxml for TMX streaming and the optional uubed-rs wheel for SQLite exact/int8 search. typing-extensions supplies Self on Python 3.10. No vector-database extension or second model loader is needed.

## Release tools

uv runs isolated build tools; gitnextver 1.0.1 calculates the next semantic tag; tomlkit preserves Cargo formatting during version synchronization; Hatch VCS generates Python versions; Maturin builds the native Python distribution; Cargo packages/verifies Rust crates; MkDocs builds/deploys the documentation site. These are development/release tools, not new runtime dependencies.
