#!/usr/bin/env python3
"""Test configuration and fixtures."""

import pytest
import tempfile
import numpy as np
from pathlib import Path


@pytest.fixture
def random_embedding():
    """Generate a random embedding for testing."""
    return np.random.randint(0, 256, 64, dtype=np.uint8)


@pytest.fixture
def random_embeddings():
    """Generate multiple random embeddings for testing."""
    return [
        np.random.randint(0, 256, 32, dtype=np.uint8)
        for _ in range(10)
    ]


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        yield tmp.name
    # Cleanup happens automatically


@pytest.fixture
def temp_binary_file():
    """Create a temporary binary file with test data."""
    data = np.random.randint(0, 256, 128, dtype=np.uint8).tobytes()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        yield tmp.name, data
    # Cleanup happens automatically


@pytest.fixture
def sample_embeddings_file():
    """Create a file with multiple embeddings for testing."""
    embeddings = [
        np.random.randint(0, 256, 64, dtype=np.uint8)
        for _ in range(5)
    ]
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for emb in embeddings:
            tmp.write(emb.tobytes())
        tmp.flush()
        yield tmp.name, embeddings
    # Cleanup happens automatically


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests (benchmarks, large data tests)
        if any(keyword in item.name.lower() for keyword in ["bench", "large", "performance"]):
            item.add_marker(pytest.mark.slow)