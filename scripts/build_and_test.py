#!/usr/bin/env python3
# this_file: scripts/build_and_test.py
"""
Complete build and test script for uubed package.
This script handles the full development workflow: build, test, and optional release.
"""

import subprocess
import sys
import argparse
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

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build and test uubed package")
    parser.add_argument("--build", action="store_true", help="Build the package")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--release", help="Create release with given version")
    parser.add_argument("--test-release", action="store_true", help="Test release to TestPyPI")
    parser.add_argument("--all", action="store_true", help="Run build and test")
    
    args = parser.parse_args()
    
    # Default to all if no specific action is given
    if not any([args.build, args.test, args.release, args.test_release]):
        args.all = True
    
    scripts_dir = Path(__file__).parent
    
    if args.all or args.build:
        print("🏗️  Building package...")
        result = run_command([sys.executable, str(scripts_dir / "build.py")])
        if result.returncode != 0:
            print("❌ Build failed")
            sys.exit(1)
        print("✅ Build completed")
    
    if args.all or args.test:
        print("🧪 Running tests...")
        result = run_command([sys.executable, str(scripts_dir / "test_runner.py")])
        if result.returncode != 0:
            print("❌ Tests failed")
            sys.exit(1)
        print("✅ Tests completed")
    
    if args.release:
        print(f"🚀 Creating release {args.release}...")
        result = run_command([sys.executable, str(scripts_dir / "release.py"), args.release])
        if result.returncode != 0:
            print("❌ Release failed")
            sys.exit(1)
        print("✅ Release completed")
    
    if args.test_release:
        print("🧪 Test release to TestPyPI...")
        result = run_command([sys.executable, str(scripts_dir / "release.py"), "--test"])
        if result.returncode != 0:
            print("❌ Test release failed")
            sys.exit(1)
        print("✅ Test release completed")
    
    print("🎉 All operations completed successfully!")

if __name__ == "__main__":
    main()