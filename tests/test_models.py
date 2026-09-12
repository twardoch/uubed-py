# this_file: tests/test_models.py
"""Vendor model contracts, including deliberate non-MRL baseline."""

import pytest

from uubed.models import MODELS, get_model


def test_registry_when_requested_models_then_exact_set_and_ids():
    assert set(MODELS) == {"jina", "gemma", "granite", "nomic", "minilm", "voyage-lite", "voyage-nano"}
    for model in MODELS.values():
        assert get_model(model.model_id) == model
        assert model.dimensions in model.mrl_dimensions
        assert model.validate_dimensions(None) == model.dimensions


@pytest.mark.parametrize(
    ("alias", "dimensions"), [("jina", 32), ("gemma", 128), ("granite", 384), ("nomic", 256), ("voyage-nano", 256)]
)
def test_model_when_documented_mrl_dimension_then_accepted(alias, dimensions):
    assert get_model(alias).validate_dimensions(dimensions) == dimensions


@pytest.mark.parametrize(
    ("alias", "dimensions"), [("minilm", 128), ("gemma", 64), ("granite", 32), ("nomic", 128), ("jina", True)]
)
def test_model_when_unsupported_dimension_then_rejected(alias, dimensions):
    with pytest.raises(ValueError, match="dimensions"):
        get_model(alias).validate_dimensions(dimensions)


def test_prompts_when_similarity_and_retrieval_then_vendor_contracts():
    assert get_model("jina").prefix("similarity") == "Document: "
    assert get_model("gemma").prefix("similarity") == "task: sentence similarity | query: "
    assert get_model("granite").prefix("similarity") == ""
    assert get_model("nomic").prefix("query") == "search_query: "
    assert get_model("nomic").prefix("document") == "search_document: "
    assert get_model("voyage-nano").prefix("document") == "Represent the document for retrieval: "
    assert get_model("minilm").mrl_dimensions == (384,)
    with pytest.raises(ValueError, match="task"):
        get_model("jina").prefix("made-up")


def test_get_model_when_unknown_then_explicit_error():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("jina-v4")
