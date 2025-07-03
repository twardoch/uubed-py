#!/usr/bin/env python3
"""Release preparation script for uubed."""

import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime

def get_current_version():
    """Get current version from __version__.py."""
    version_file = Path("src/uubed/__version__.py")
    content = version_file.read_text()
    match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    raise ValueError("Could not find version in __version__.py")

def update_version(new_version):
    """Update version in __version__.py."""
    version_file = Path("src/uubed/__version__.py")
    content = version_file.read_text()
    
    # Update version
    new_content = re.sub(
        r'__version__ = ["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        content
    )
    
    version_file.write_text(new_content)
    print(f"Updated version to {new_version}")

def update_pyproject_version(new_version):
    """Update version in pyproject.toml."""
    pyproject_file = Path("pyproject.toml")
    content = pyproject_file.read_text()
    
    # Update version
    new_content = re.sub(
        r'version = "[^"]+"',
        f'version = "{new_version}"',
        content
    )
    
    pyproject_file.write_text(new_content)
    print(f"Updated pyproject.toml version to {new_version}")

def update_changelog(version):
    """Update changelog with release date."""
    changelog_file = Path("CHANGELOG.md")
    content = changelog_file.read_text()
    
    # Update release date
    today = datetime.now().strftime("%Y-%m-%d")
    new_content = content.replace(
        f"## [{version}] - 2024-01-XX (In Development)",
        f"## [{version}] - {today}"
    )
    
    changelog_file.write_text(new_content)
    print(f"Updated changelog for version {version}")

def run_tests():
    """Run the test suite."""
    print("Running tests...")
    result = subprocess.run([sys.executable, "scripts/test.py"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Tests failed!")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("Tests passed!")
    return True

def build_package():
    """Build the package."""
    print("Building package...")
    
    # Clean previous builds
    subprocess.run(["rm", "-rf", "dist/", "build/"], shell=True)
    
    # Build with setuptools (since we don't have Rust extension yet)
    result = subprocess.run([sys.executable, "-m", "build"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Build failed!")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print("Package built successfully!")
    return True

def check_git_status():
    """Check if git working directory is clean."""
    result = subprocess.run(["git", "status", "--porcelain"], 
                          capture_output=True, text=True)
    
    if result.stdout.strip():
        print("Warning: Git working directory is not clean")
        print(result.stdout)
        return False
    
    return True

def create_git_tag(version):
    """Create git tag for the release."""
    tag = f"v{version}"
    
    # Create tag
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", f"Release {version}"])
    subprocess.run(["git", "tag", "-a", tag, "-m", f"Release {version}"])
    
    print(f"Created git tag {tag}")

def main():
    """Main release script."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/release.py <version>")
        print("Example: python scripts/release.py 0.2.0")
        sys.exit(1)
    
    new_version = sys.argv[1]
    
    # Validate version format
    if not re.match(r'^\d+\.\d+\.\d+$', new_version):
        print("Version must be in format X.Y.Z")
        sys.exit(1)
    
    current_version = get_current_version()
    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")
    
    # Confirm release
    response = input("Continue with release? (y/N): ")
    if response.lower() != 'y':
        print("Release cancelled")
        sys.exit(0)
    
    # Run tests
    if not run_tests():
        print("Release cancelled due to test failures")
        sys.exit(1)
    
    # Update versions
    update_version(new_version)
    update_pyproject_version(new_version)
    update_changelog(new_version)
    
    # Build package
    if not build_package():
        print("Release cancelled due to build failure")
        sys.exit(1)
    
    # Create git tag
    create_git_tag(new_version)
    
    print(f"\n🎉 Release {new_version} prepared successfully!")
    print("\nNext steps:")
    print("1. Push to repository: git push && git push --tags")
    print("2. Upload to PyPI: python -m twine upload dist/*")
    print("3. Create GitHub release")

if __name__ == "__main__":
    main()