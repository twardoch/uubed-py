#!/usr/bin/env python3
# this_file: scripts/build.py
"""
Build script for uubed package.
This script handles building wheels and source distributions for local development and release.
"""

import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Optional

def run_command(cmd: List[str], cwd: Optional[Path] = None, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=capture_output, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        raise

def clean_build():
    """Clean build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    # Remove build directories
    for path in ["build", "dist", "*.egg-info"]:
        for item in Path(".").glob(path):
            if item.is_dir():
                shutil.rmtree(item)
                print(f"Removed directory: {item}")
            elif item.is_file():
                item.unlink()
                print(f"Removed file: {item}")

def build_package() -> int:
    """Build the package using hatch."""
    print("📦 Building uubed package...")
    
    # Check if hatch is available
    try:
        result = run_command([sys.executable, "-m", "hatch", "--version"], capture_output=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, "hatch")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing hatch...")
        result = run_command([sys.executable, "-m", "pip", "install", "hatch", "hatch-vcs"])
        if result.returncode != 0:
            print("Failed to install hatch")
            return result.returncode
    
    # Use hatch to build the package
    result = run_command([sys.executable, "-m", "hatch", "build"])
    return result.returncode

def verify_build():
    """Verify the build artifacts."""
    print("🔍 Verifying build artifacts...")
    
    dist_path = Path("dist")
    if not dist_path.exists():
        print("❌ No dist directory found")
        return False
    
    wheels = list(dist_path.glob("*.whl"))
    sdists = list(dist_path.glob("*.tar.gz"))
    
    if not wheels:
        print("❌ No wheel files found")
        return False
    
    if not sdists:
        print("❌ No source distribution files found")
        return False
    
    print(f"✅ Found {len(wheels)} wheel(s) and {len(sdists)} source distribution(s)")
    for wheel in wheels:
        print(f"  - {wheel.name}")
    for sdist in sdists:
        print(f"  - {sdist.name}")
    
    return True

def main():
    """Main build function."""
    print("🏗️  Starting uubed build process...")
    
    # Clean previous builds
    clean_build()
    
    # Build the package
    exit_code = build_package()
    if exit_code != 0:
        print("❌ Build failed!")
        sys.exit(exit_code)
    
    # Verify build artifacts
    if not verify_build():
        print("❌ Build verification failed!")
        sys.exit(1)
    
    print("✅ Build completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())