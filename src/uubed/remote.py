# this_file: src/uubed/remote.py
"""Documented provider schemas with deadlines and strict row-order validation."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uubed.models import Model

PROVIDERS = {
    "jina": ("jinaai/jina-embeddings-v5-text-nano-text-matching", "JINA_API_KEY", "https://api.jina.ai/v1/embeddings"),
    "voyage": ("voyage-4-lite", "VOYAGE_API_KEY", "https://api.voyageai.com/v1/embeddings"),
    "hf": (
        "ibm-granite/granite-embedding-311m-multilingual-r2",
        "HF_TOKEN",
        "https://router.huggingface.co/hf-inference/models/ibm-granite/granite-embedding-311m-multilingual-r2",
    ),
    "cloudflare": ("google/embeddinggemma-300m", "CLOUDFLARE_API_TOKEN", ""),
}


class ApiBackend:
    """Reuse one HTTP client. Generic OpenAI endpoints require an explicit URL.

    Nomic v2 MoE has no verified vendor serverless route; use its exact model ID
    on a self-hosted OpenAI-compatible server. Provider errors never trigger a
    switch to a different model or send data to another provider.
    """

    def __init__(
        self,
        model: Model,
        provider: str,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        task: str = "similarity",
    ) -> None:
        import httpx  # noqa: PLC0415 - optional dependency

        if not math.isfinite(timeout) or timeout <= 0:
            msg = "timeout must be finite and positive"
            raise ValueError(msg)
        model.prefix(task)
        if provider == "jina" and task != "similarity":
            msg = "the requested Jina text-matching API adapter supports the similarity task"
            raise ValueError(msg)
        if provider == "openai":
            default, key = "", api_key
        elif provider in PROVIDERS and PROVIDERS[provider][0] == model.model_id:
            _, variable, default = PROVIDERS[provider]
            key = api_key or os.environ.get(variable)
            if not key:
                msg = f"set {variable} or supply api_key for provider {provider}"
                raise ValueError(msg)
        else:
            msg = f"provider {provider!r} does not serve {model.model_id}"
            raise ValueError(msg)
        if provider == "cloudflare" and not endpoint:
            account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
            if account:
                default = (
                    f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/google/embeddinggemma-300m"
                )
        self.endpoint = endpoint or default
        if not self.endpoint.startswith(("http://", "https://")):
            msg = "an explicit HTTP endpoint is required"
            raise ValueError(msg)
        self.model, self.provider, self.task = model, provider, task
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = httpx.Client(timeout=timeout, headers=headers)

    def _payload(self, texts: list[str], prefix: str) -> dict[str, Any]:
        if self.provider == "jina":
            return {
                "model": "jina-embeddings-v5-text-nano",
                "input": texts,
                "task": "text-matching",
                "embedding_type": "float",
                "normalized": False,
                "truncate": False,
            }
        if self.provider == "voyage":
            input_type = None if self.task == "similarity" else self.task
            return {
                "model": self.model.model_id,
                "input": texts,
                "input_type": input_type,
                "output_dimension": self.model.dimensions,
                "output_dtype": "float",
                "truncation": False,
            }
        prepared = [prefix + t for t in texts]
        if self.provider == "hf":
            return {"inputs": prepared, "parameters": {"normalize": False, "truncate": False}}
        if self.provider == "cloudflare":
            return {"text": prepared}
        return {"model": self.model.model_id, "input": prepared, "encoding_format": "float"}

    def _rows(self, response: Any, count: int) -> list[list[float]]:
        try:
            if self.provider == "cloudflare":
                if response.get("success") is not True:
                    msg = "provider returned an unsuccessful response"
                    raise ValueError(msg)
                rows = response["result"]["data"]
            elif self.provider == "hf":
                rows = response
            else:
                indexed = response["data"]
                indices = [item["index"] for item in indexed]
                if any(type(index) is not int for index in indices) or sorted(indices) != list(range(count)):
                    msg = "response indices are missing, duplicated or outside the input batch"
                    raise ValueError(msg)
                rows = [item["embedding"] for item in sorted(indexed, key=lambda item: item["index"])]
            if not isinstance(rows, list) or len(rows) != count or any(not isinstance(row, list) for row in rows):
                msg = "response row count or embedding shape is invalid"
                raise ValueError(msg)
            return rows
        except (KeyError, TypeError, AttributeError) as error:
            msg = "provider response does not match the documented embedding schema"
            raise ValueError(msg) from error

    def encode(self, texts: list[str], prefix: str, batch_size: int) -> list[list[float]]:
        rows = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            response = self.client.post(self.endpoint, json=self._payload(batch, prefix))
            response.raise_for_status()
            rows.extend(self._rows(response.json(), len(batch)))
        return rows

    def close(self) -> None:
        self.client.close()
