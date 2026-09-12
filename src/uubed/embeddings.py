# this_file: src/uubed/embeddings.py
"""Common model-aware inference and compact-vector API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from uubed.compact import PRECISIONS, PackedEmbedding, pack
from uubed.models import get_model
from uubed.remote import PROVIDERS

REMOTE_BACKENDS = ("jina", "voyage", "hf", "cloudflare", "openai")
BACKENDS = ("sentence-transformers", "llama.cpp", "onnx", "mlx", *REMOTE_BACKENDS)
MIN_TOKENS = 16
# MLX/MPS share process-level Metal state; concurrent independent models can abort.
_PYTHON_METAL_LOCK = RLock()


class Embedder:
    """Lazy resident model. Calls are serialized to protect native model state.

    Create one instance per model and reuse it. Increase batch_size before adding
    workers: concurrent Python calls to one GPU model share this lock. Use close()
    to release weights explicitly. Cache identity includes all output settings.
    """

    def __init__(
        self,
        model: str,
        *,
        backend: str = "sentence-transformers",
        path: Path | str | None = None,
        device: str = "mps",
        task: str = "similarity",
        dimensions: int | None = None,
        precision: str = "int8",
        prefix: str | None = None,
        max_tokens: int = 512,
        batch_tokens: int = 4096,
        batch_size: int = 32,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = get_model(model)
        if backend == "onnx" and device != "cpu":
            msg = "the verified ONNX backend requires device='cpu'"
            raise ValueError(msg)
        if path is not None and backend != "llama.cpp":
            msg = "path is only used by the llama.cpp backend"
            raise ValueError(msg)
        if backend == "mlx" and self.model.model_id not in {
            "sentence-transformers/all-MiniLM-L6-v2",
            "ibm-granite/granite-embedding-311m-multilingual-r2",
        }:
            msg = "the verified MLX backend supports MiniLM and Granite only; use llama.cpp or SentenceTransformers"
            raise ValueError(msg)
        self.endpoint, self._api_key, self.timeout = endpoint, api_key, timeout
        self.model.prefix(task)
        if task != "similarity" and prefix not in (None, self.model.prefix(task)):
            msg = "custom retrieval prompt pairs are unsupported; use the model defaults"
            raise ValueError(msg)
        if backend in PROVIDERS and PROVIDERS[backend][0] != self.model.model_id:
            msg = f"provider {backend!r} does not serve {self.model.model_id}"
            raise ValueError(msg)
        if self.model.pooling == "provider" and backend not in REMOTE_BACKENDS:
            msg = f"{self.model.model_id} is API-only"
            raise ValueError(msg)
        if backend in {"jina", "voyage"} and prefix is not None:
            msg = "this provider applies task prompts; prefix override is unsupported"
            raise ValueError(msg)
        self.dimensions = self.model.validate_dimensions(dimensions)
        self.prefix = self.model.prefix(task) if prefix is None else prefix
        if backend not in BACKENDS or precision not in PRECISIONS:
            msg = "unsupported backend or precision"
            raise ValueError(msg)
        if any(type(n) is not int or n < 1 for n in (max_tokens, batch_tokens, batch_size)):
            msg = "token and batch sizes must be positive integers"
            raise ValueError(msg)
        if min(max_tokens, batch_tokens) < MIN_TOKENS:
            msg = f"token budgets must be at least {MIN_TOKENS}"
            raise ValueError(msg)
        self.backend, self.device, self.task, self.precision = backend, device, task, precision
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self.max_tokens = min(max_tokens, self.model.context_tokens, batch_tokens)
        self.batch_tokens, self.batch_size = batch_tokens, batch_size
        self._runtime: Any = None
        self._lock = (
            _PYTHON_METAL_LOCK
            if backend == "mlx" or (backend == "sentence-transformers" and device.partition(":")[0] == "mps")
            else RLock()
        )
        self._weights: str | None = None
        self._weight_stat: tuple[int, ...] | None = None

    def _fingerprint(self) -> str:
        if self.path is None:
            if self.backend in REMOTE_BACKENDS:
                return "unversioned-hosted"
            code = f"+code:{self.model.code_revision}" if self.model.code_revision else ""
            return self.model.revision + code
        stat = self.path.stat()
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._weight_stat is not None and current != self._weight_stat:
            msg = "weights changed; create a new Embedder to establish a new cache identity"
            raise ValueError(msg)
        if self._weights is None:
            self._weight_stat = current
            digest = hashlib.sha256()
            with self.path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self._weights = digest.hexdigest()
            self._fingerprint()  # Catch replacement or modification during hashing.
        return self._weights

    @property
    def space(self) -> str:
        """Model revision/weights/backend/task identity for persisted comparisons."""
        with self._lock:
            weights = self._fingerprint()
        task_group = "similarity" if self.task == "similarity" else "retrieval"
        prompts = self.prefix if task_group == "similarity" else self.model.prompts[1:]
        identity = [self.model.model_id, weights, self.backend, task_group, prompts, self.max_tokens, self.endpoint]
        return json.dumps(identity, separators=(",", ":"))

    @property
    def cache_identity(self) -> str:
        return hashlib.sha256(
            f"{self.space}:{self.task}:{self.prefix}:{self.dimensions}:{self.precision}".encode()
        ).hexdigest()

    def _load(self) -> Any:
        if self._runtime is None:
            from uubed.local import LlamaBackend, SentenceTransformersBackend  # noqa: PLC0415

            if self.backend in REMOTE_BACKENDS:
                from uubed.remote import ApiBackend  # noqa: PLC0415

                self._runtime = ApiBackend(
                    self.model,
                    self.backend,
                    endpoint=self.endpoint,
                    api_key=self._api_key,
                    timeout=self.timeout,
                    task=self.task,
                )
            elif self.backend == "llama.cpp":
                if self.path is None or not self.path.is_file():
                    msg = f"model file missing: {self.path}"
                    raise FileNotFoundError(msg)
                self._runtime = LlamaBackend(self.model, self.path, self.batch_tokens, self.max_tokens)
            elif self.backend == "mlx":
                from uubed.local import MlxBackend  # noqa: PLC0415

                self._runtime = MlxBackend(self.model, self.max_tokens)
            else:
                backend = "onnx" if self.backend == "onnx" else "torch"
                self._runtime = SentenceTransformersBackend(self.model, self.device, self.max_tokens, backend)
        return self._runtime

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return float32 normalized prefixes in input order, without quantization."""
        if not isinstance(texts, list) or any(not isinstance(t, str) or not t.strip() for t in texts):
            msg = "texts must be a list of nonempty strings"
            raise ValueError(msg)
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        with self._lock:
            self._fingerprint()
            rows = self._load().encode(texts, self.prefix, self.batch_size)
            self._fingerprint()
        vectors = np.asarray(rows, dtype=np.float32)
        if vectors.shape != (len(texts), self.model.dimensions):
            msg = f"backend returned dimensions {vectors.shape}, expected {(len(texts), self.model.dimensions)}"
            raise ValueError(msg)
        if not np.all(np.isfinite(vectors)):
            msg = "backend vectors must be finite, including dimensions discarded by truncation"
            raise ValueError(msg)
        vectors = vectors[:, : self.dimensions]
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms == 0):
            msg = "backend vectors must have finite nonzero norms"
            raise ValueError(msg)
        return (vectors / norms).astype(np.float32)

    def pack(self, texts: list[str]) -> list[PackedEmbedding]:
        """Infer and quantize records suitable for uubed.infile.write_embedding."""
        return [pack(row, space=self.space, precision=self.precision) for row in self.embed(texts)]

    def close(self) -> None:
        with self._lock:
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
