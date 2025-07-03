#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Building uubed-py..."

# Build the Python package (wheel and sdist)
uvx hatch build

echo "uubed-py build complete. Wheels and sdists are in the dist/ directory."

# Optional: Install the package in editable mode for development
# echo "Installing uubed-py in editable mode..."
# uv pip install -e .
# echo "uubed-py installed in editable mode."
