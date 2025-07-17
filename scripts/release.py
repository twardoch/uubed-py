#!/usr/bin/env python3
# this_file: scripts/release.py
"""
Release script for uubed package.
This script handles the complete release process including version validation,
building, testing, and publishing.
"""

import subprocess
import sys
import re
import os
from pathlib import Path
from typing import List, Optional, Tuple

def run_command(cmd: List[str], cwd: Optional[Path] = None, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=capture_output, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        raise

def get_current_version() -> str:
    """Get the current version from git tags."""
    try:
        result = run_command(["git", "describe", "--tags", "--abbrev=0"], capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "v0.0.0"
    except:
        return "v0.0.0"

def get_git_version() -> str:
    """Get the current git-based version."""
    try:
        result = run_command(["git", "describe", "--tags", "--dirty"], capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "v0.0.0-dev"
    except:
        return "v0.0.0-dev"

def validate_version(version: str) -> bool:
    """Validate that the version follows semantic versioning."""
    # Remove 'v' prefix if present
    version = version.lstrip('v')
    
    # Regex for semantic versioning
    semver_pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
    
    return re.match(semver_pattern, version) is not None

def check_git_status() -> bool:
    """Check if git working directory is clean."""
    result = run_command(["git", "status", "--porcelain"], capture_output=True)
    if result.returncode != 0:
        print("❌ Failed to check git status")
        return False
    
    if result.stdout.strip():
        print("❌ Git working directory is not clean. Please commit all changes.")
        print("Uncommitted changes:")
        print(result.stdout)
        return False
    
    return True

def check_git_tag_exists(tag: str) -> bool:
    """Check if a git tag exists."""
    result = run_command(["git", "tag", "-l", tag], capture_output=True)
    return result.returncode == 0 and result.stdout.strip() == tag

def run_tests() -> bool:
    """Run the full test suite."""
    print("🧪 Running full test suite...")
    
    test_script = Path(__file__).parent / "test_runner.py"
    result = run_command([sys.executable, str(test_script)])
    
    if result.returncode != 0:
        print("❌ Tests failed")
        return False
    
    print("✅ All tests passed")
    return True

def build_package() -> bool:
    """Build the package."""
    print("📦 Building package...")
    
    build_script = Path(__file__).parent / "build.py"
    result = run_command([sys.executable, str(build_script)])
    
    if result.returncode != 0:
        print("❌ Build failed")
        return False
    
    print("✅ Package built successfully")
    return True

def create_git_tag(version: str) -> bool:
    """Create a git tag."""
    if not version.startswith('v'):
        version = f'v{version}'
    
    if check_git_tag_exists(version):
        print(f"⚠️  Tag {version} already exists")
        return True
    
    print(f"🏷️  Creating git tag: {version}")
    result = run_command(["git", "tag", "-a", version, "-m", f"Release {version}"])
    
    if result.returncode != 0:
        print(f"❌ Failed to create tag {version}")
        return False
    
    print(f"✅ Tag {version} created successfully")
    return True

def push_tag(version: str) -> bool:
    """Push the git tag to remote."""
    if not version.startswith('v'):
        version = f'v{version}'
    
    print(f"📤 Pushing tag {version} to remote...")
    result = run_command(["git", "push", "origin", version])
    
    if result.returncode != 0:
        print(f"❌ Failed to push tag {version}")
        return False
    
    print(f"✅ Tag {version} pushed successfully")
    return True

def publish_to_pypi(test: bool = False) -> bool:
    """Publish the package to PyPI."""
    if test:
        print("📦 Publishing to TestPyPI...")
        repo_url = "https://test.pypi.org/legacy/"
        token_env = "TEST_PYPI_TOKEN"
    else:
        print("📦 Publishing to PyPI...")
        repo_url = "https://upload.pypi.org/legacy/"
        token_env = "PYPI_TOKEN"
    
    # Check if token is available
    token = os.environ.get(token_env)
    if not token:
        print(f"❌ {token_env} environment variable not set")
        return False
    
    # Use twine to upload
    cmd = [
        sys.executable, "-m", "twine", "upload",
        "--repository-url", repo_url,
        "--username", "__token__",
        "--password", token,
        "dist/*"
    ]
    
    result = run_command(cmd)
    
    if result.returncode != 0:
        print("❌ Failed to publish to PyPI")
        return False
    
    print("✅ Package published successfully")
    return True

def main():
    """Main release function."""
    print("🚀 Starting uubed release process...")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_mode = True
            print("📦 Running in test mode (will publish to TestPyPI)")
        else:
            version = sys.argv[1]
            test_mode = False
    else:
        print("Usage: python scripts/release.py [version] OR python scripts/release.py --test")
        print("Example: python scripts/release.py v1.0.3")
        sys.exit(1)
    
    # If test mode, use current git version
    if test_mode:
        version = get_git_version()
        print(f"Using git version: {version}")
    
    # Validate version format
    if not validate_version(version):
        print(f"❌ Invalid version format: {version}")
        print("Version must follow semantic versioning (e.g., v1.2.3)")
        sys.exit(1)
    
    # Check git status
    if not check_git_status():
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        sys.exit(1)
    
    # Build package
    if not build_package():
        sys.exit(1)
    
    # Create git tag (skip in test mode)
    if not test_mode:
        if not create_git_tag(version):
            sys.exit(1)
    
    # Install twine if not present
    try:
        subprocess.run([sys.executable, "-m", "twine", "--version"], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Installing twine...")
        subprocess.run([sys.executable, "-m", "pip", "install", "twine"], check=True)
    
    # Publish to PyPI
    if not publish_to_pypi(test=test_mode):
        sys.exit(1)
    
    # Push tag (skip in test mode)
    if not test_mode:
        if not push_tag(version):
            sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 RELEASE COMPLETE!")
    print("="*60)
    print(f"Version: {version}")
    if test_mode:
        print("Published to: TestPyPI")
    else:
        print("Published to: PyPI")
        print(f"Git tag: {version}")
    print("="*60)

if __name__ == "__main__":
    main()