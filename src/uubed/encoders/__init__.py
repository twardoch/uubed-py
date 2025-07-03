#!/usr/bin/env python3
# this_file: src/uubed/encoders/__init__.py
"""Encoders package for uubed - High-performance encoding for embedding vectors."""

# Import encoders for convenient access
from .q64 import q64_encode, q64_decode
from .eq64 import eq64_encode, eq64_decode
from .shq64 import simhash_q64
from .t8q64 import top_k_q64
from .zoq64 import z_order_q64


def get_available_encoders() -> list[str]:
    """
    Get a list of available encoding methods.
    
    Returns:
        List of available encoding method names.
    """
    return ["eq64", "shq64", "t8q64", "zoq64", "mq64"]


__all__ = [
    "q64_encode",
    "q64_decode",
    "eq64_encode",
    "eq64_decode",
    "simhash_q64",
    "top_k_q64",
    "z_order_q64",
    "get_available_encoders",
]