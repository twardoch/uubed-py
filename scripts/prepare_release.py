#!/usr/bin/env python3
# this_file: scripts/prepare_release.py
"""Compatibility alias for the non-publishing release preview."""
from pathlib import Path
import subprocess

if __name__ == "__main__":
    raise SystemExit(subprocess.call(["bash", "publish.sh", "--dry-run"], cwd=Path(__file__).resolve().parents[1]))
