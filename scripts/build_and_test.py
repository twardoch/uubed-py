#!/usr/bin/env python3
# this_file: scripts/build_and_test.py
"""Run the repository test and build gates. Releases use ../publish.sh."""
from pathlib import Path
import subprocess

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for script in ("test.sh", "build.sh"):
        subprocess.run(["bash", script], cwd=root, check=True)
