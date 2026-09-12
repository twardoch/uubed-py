# this_file: src/uubed/local.py
"""Persistent library-backed local inference; no custom transformer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from uubed.models import Model

POOLING = {"mean": 1, "mean+dense": 1, "cls": 2, "last": 3}


class LlamaBackend:
    """One resident llama.cpp model with bounded physical token batches."""

    def __init__(self, model: Model, path: Path, batch_tokens: int, max_tokens: int) -> None:
        from llama_cpp import Llama  # noqa: PLC0415 - optional runtime loaded on demand

        self.max_tokens = max_tokens
        self.batch_tokens = batch_tokens
        self.model = Llama(
            model_path=str(path),
            embedding=True,
            n_gpu_layers=-1,
            n_ctx=batch_tokens,
            n_batch=batch_tokens,
            n_ubatch=batch_tokens,
            pooling_type=POOLING[model.pooling],
            verbose=False,
        )

    def _text(self, text: str) -> tuple[str, int]:
        size = len(self.model.tokenize(text.encode()))
        if size > self.max_tokens:
            tokens = self.model.tokenize(text.encode(), add_bos=False)[: self.max_tokens - 2]
            text = self.model.detokenize(tokens).decode("utf-8", errors="ignore")
            size = len(self.model.tokenize(text.encode()))
            while size > self.max_tokens and tokens:
                tokens.pop()
                text = self.model.detokenize(tokens).decode("utf-8", errors="ignore")
                size = len(self.model.tokenize(text.encode()))
        if size > self.batch_tokens:
            msg = "tokenized text exceeds physical batch budget"
            raise ValueError(msg)
        return text, size

    def encode(self, texts: list[str], prefix: str, batch_size: int) -> list[list[float]]:
        rows: list[list[float]] = []
        batch: list[str] = []
        used = 0
        for text in texts:
            prepared, count = self._text(prefix + text)
            if batch and (used + count > self.batch_tokens or len(batch) >= batch_size):
                rows.extend(self.model.embed(batch, truncate=False))
                batch, used = [], 0
            batch.append(prepared)
            used += count
        if batch:
            rows.extend(self.model.embed(batch, truncate=False))
        return rows

    def close(self) -> None:
        self.model.close()


class SentenceTransformersBackend:
    """Reuse the vendor's complete pooling/projection pipeline on MPS, CPU or ONNX."""

    def __init__(self, model: Model, device: str, max_tokens: int, backend: str = "torch") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - optional runtime

        code = {"code_revision": model.code_revision} if model.code_revision else {}
        runtime = {"provider": "CPUExecutionProvider"} if backend == "onnx" else {}
        self.model = SentenceTransformer(
            model.model_id,
            revision=model.revision,
            trust_remote_code=True,
            device=device,
            backend=backend,
            model_kwargs={**code, **runtime},
            config_kwargs=code,
        )
        self.model.max_seq_length = max_tokens

    def encode(self, texts: list[str], prefix: str, batch_size: int) -> Any:
        return self.model.encode(
            texts,
            prompt=prefix,
            batch_size=batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def close(self) -> None:
        self.model = None


class MlxBackend:
    """Verified MiniLM/Granite pooling using the maintained MLX implementation."""

    def __init__(self, model: Model, max_tokens: int) -> None:
        from mlx_embeddings import load  # noqa: PLC0415 - optional Apple Silicon runtime
        from mlx_embeddings.utils import get_model_path  # noqa: PLC0415

        path = get_model_path(model.model_id, revision=model.revision)
        self.model, self.tokenizer = load(str(path))
        self.max_tokens = max_tokens

    def encode(self, texts: list[str], prefix: str, batch_size: int) -> list[list[float]]:
        import mlx.core as mx  # noqa: PLC0415

        rows = []
        for offset in range(0, len(texts), batch_size):
            batch = [prefix + text for text in texts[offset : offset + batch_size]]
            tokens = self.tokenizer.batch_encode_plus(
                batch, padding=True, truncation=True, max_length=self.max_tokens, return_tensors="np"
            )
            inputs = {
                key: mx.array(value)
                for key, value in tokens.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            vectors = self.model(**inputs).text_embeds.astype(mx.float32)
            mx.eval(vectors)
            rows.extend(vectors.tolist())
        return rows

    def close(self) -> None:
        import mlx.core as mx  # noqa: PLC0415

        self.model = None
        mx.clear_cache()
