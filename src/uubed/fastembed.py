# this_file: src/uubed/fastembed.py
"""Pinned ONNX exports executed by fastembed-rs through uubed_native."""

from __future__ import annotations

import json
from dataclasses import dataclass

from uubed.models import Model, get_model


@dataclass(frozen=True)
class Export:
    alias: str
    repository: str
    revision: str
    model_file: str = "onnx/model.onnx"
    external_file: str | None = None


EXPORTS = {
    get_model("gemma").model_id: Export(
        "gemma",
        "onnx-community/embeddinggemma-300m-ONNX",
        "5090578d9565bb06545b4552f76e6bc2c93e4a66",
        "onnx/model.onnx",
        "onnx/model.onnx_data",
    ),
    get_model("jina").model_id: Export(
        "jina",
        get_model("jina").model_id,
        get_model("jina").revision,
        "onnx/model.onnx",
        "onnx/model.onnx_data",
    ),
    get_model("minilm").model_id: Export("minilm", get_model("minilm").model_id, get_model("minilm").revision),
    get_model("granite").model_id: Export("granite", get_model("granite").model_id, get_model("granite").revision),
    get_model("voyage-nano").model_id: Export(
        "voyage-nano",
        "onnx-community/voyage-4-nano-ONNX",
        "fb6f32cceb63b6eac92e1fdc08e00d03c41cdf5a",
        "onnx/model.onnx",
        "onnx/model.onnx_data",
    ),
    # Dynamic quantization changes MoE routing with batch composition; use FP16.
    get_model("nomic").model_id: Export(
        "nomic",
        "netrias/nomic-embed-text-v2-moe-onnx",
        "5dcdc1a46ce6d4c529fadb42cccf9ea72d36eb0b",
        "onnx/model_fp16.onnx",
    ),
}


def export_for(model: Model) -> Export:
    try:
        return EXPORTS[model.model_id]
    except KeyError as error:
        msg = "FastEmbed supports the six local model profiles; Voyage lite requires the voyage API backend"
        raise ValueError(msg) from error


def identity(model: Model) -> str:
    export = export_for(model)
    return json.dumps(["fastembed-6.0.3", export.repository, export.revision, export.model_file, export.external_file])


class FastEmbedBackend:
    """One CPU session with explicit model outputs, prompts and bounded batches."""

    def __init__(self, model: Model, max_tokens: int) -> None:
        try:
            from huggingface_hub import constants, snapshot_download  # noqa: PLC0415
            from uubed_native import NativeEmbedder  # noqa: PLC0415
        except ImportError as error:
            msg = (
                "FastEmbed requires uubed[fastembed] and a uubed-rs wheel with NativeEmbedder; "
                "rebuild/install the updated native wheel"
            )
            raise ImportError(msg) from error
        export = export_for(model)
        files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", export.model_file]
        if export.external_file:
            files.append(export.external_file)
        directory = snapshot_download(
            repo_id=export.repository,
            revision=export.revision,
            allow_patterns=files,
            local_files_only=constants.HF_HUB_OFFLINE,
        )
        self.model = NativeEmbedder(export.alias, directory, export.model_file, export.external_file, max_tokens)

    def encode(self, texts: list[str], prefix: str, batch_size: int) -> list[list[float]]:
        return self.model.embed([prefix + text for text in texts], batch_size)

    def close(self) -> None:
        self.model.close()
