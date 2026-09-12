# this_file: src/uubed/models.py
"""Pinned model contracts verified against vendor cards on 2026-09-11.

Aliases are convenience names, never embedding-space identities. Each local
backend must apply the given prompt once and preserve the stated pooling.
"""

from __future__ import annotations

from dataclasses import dataclass

TASKS = ("similarity", "query", "document")


@dataclass(frozen=True)
class Model:
    """Native dimensions and documented prefix/truncation choices for one model."""

    model_id: str
    revision: str
    dimensions: int
    mrl_dimensions: tuple[int, ...]
    context_tokens: int
    pooling: str
    prompts: tuple[str, str, str]
    license: str
    code_revision: str | None = None

    def prefix(self, task: str) -> str:
        if task not in TASKS:
            msg = f"task must be one of {TASKS}"
            raise ValueError(msg)
        return self.prompts[TASKS.index(task)]

    def validate_dimensions(self, dimensions: int | None) -> int:
        dimensions = self.dimensions if dimensions is None else dimensions
        if type(dimensions) is not int or dimensions not in self.mrl_dimensions:
            msg = f"{self.model_id} supports dimensions {self.mrl_dimensions}, got {dimensions}"
            raise ValueError(msg)
        return dimensions


MODELS = {
    "jina": Model(
        "jinaai/jina-embeddings-v5-text-nano-text-matching",
        "a75863a1a1ada69eda80f20ae26b5f40b4a823bc",
        768,
        (32, 64, 128, 256, 512, 768),
        8192,
        "last",
        ("Document: ", "Query: ", "Document: "),
        "CC-BY-NC-4.0",
    ),
    "gemma": Model(
        "google/embeddinggemma-300m",
        "57c266a740f537b4dc058e1b0cda161fd15afa75",
        768,
        (128, 256, 512, 768),
        2048,
        "mean+dense",
        ("task: sentence similarity | query: ", "task: search result | query: ", "title: none | text: "),
        "Gemma",
    ),
    "granite": Model(
        "ibm-granite/granite-embedding-311m-multilingual-r2",
        "44399559930365213510b1ee2eb15ded83374f0e",
        768,
        (128, 256, 384, 512, 768),
        32768,
        "cls",
        ("", "", ""),
        "Apache-2.0; Gemma tokenizer terms",
    ),
    "nomic": Model(
        "nomic-ai/nomic-embed-text-v2-moe",
        "1066b6599d099fbb93dfcb64f9c37a7c9e503e85",
        768,
        (256, 512, 768),
        512,
        "mean",
        ("search_document: ", "search_query: ", "search_document: "),
        "Apache-2.0",
        code_revision="7710840340a098cfb869c4f65e87cf2b1b70caca",
    ),
    "minilm": Model(
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        384,
        (384,),
        256,
        "mean",
        ("", "", ""),
        "Apache-2.0",
    ),
    "voyage-lite": Model(
        "voyage-4-lite",
        "hosted-voyage-4-lite",
        2048,
        (256, 512, 1024, 2048),
        32000,
        "provider",
        ("", "", ""),
        "Voyage API terms",
    ),
    "voyage-nano": Model(
        "voyageai/voyage-4-nano",
        "67fabc9bef010dabc5f6024aa1b1b6b93410426f",
        2048,
        (256, 512, 1024, 2048),
        32768,
        "mean",
        ("", "Represent the query for retrieving supporting documents: ", "Represent the document for retrieval: "),
        "Apache-2.0",
    ),
}


def get_model(name: str) -> Model:
    """Resolve an alias or exact vendor ID; refuse silent model substitutions."""
    if name in MODELS:
        return MODELS[name]
    for model in MODELS.values():
        if name == model.model_id:
            return model
    msg = f"Unknown model {name!r}; choose one of {tuple(MODELS)}"
    raise ValueError(msg)
