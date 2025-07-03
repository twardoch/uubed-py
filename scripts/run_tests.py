#!/usr/bin/env python3
"""Isolated test runner for uubed to avoid pytest plugin conflicts."""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run tests with clean environment."""
    # Add src to Python path
    src_path = Path(__file__).parent.parent / "src"
    env = {"PYTHONPATH": str(src_path)}
    
    # Run pytest with minimal configuration
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short", 
        "--disable-warnings",
        "-p", "no:cacheprovider",
        "-p", "no:briefcase",
        "-p", "no:pytest_virtualenv",
        "-p", "no:pytest-virtualenv"
    ]
    
    try:
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {Path(__file__).parent.parent}")
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)