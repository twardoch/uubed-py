#!/usr/bin/env python3
# this_file: src/uubed/__init__.py
"""
uubed: High-performance encoding for embedding vectors.

Solves the "substring pollution" problem in search systems by using
position-dependent alphabets that prevent false matches.
"""

from .__version__ import __version__
from .api import encode, decode

__all__ = ["encode", "decode", "__version__"]