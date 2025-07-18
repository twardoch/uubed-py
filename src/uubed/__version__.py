# this_file: src/uubed/__version__.py
"""
Version information for uubed package.
This module provides version information using hatch-vcs for git-tag-based semversioning.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("uubed")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development/testing when package is not installed
    __version__ = "0.0.0.dev0"

# Parse version tuple
version_parts = __version__.split(".")
try:
    __version_tuple__ = tuple(int(part) for part in version_parts if part.isdigit())
except ValueError:
    __version_tuple__ = (0, 0, 0)

# Legacy compatibility
version = __version__
version_tuple = __version_tuple__

__all__ = ["__version__", "__version_tuple__", "version", "version_tuple"]
