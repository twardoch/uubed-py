#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["twine", "build", "maturin"]
# ///
# this_file: scripts/prepare_release.py
"""Prepare package for PyPI release."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and print its output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result

def clean_dist():
    """Clean the dist directory."""
    print("Cleaning dist directory...")
    dist_path = Path("dist")
    if dist_path.exists():
        import shutil
        shutil.rmtree(dist_path)
    dist_path.mkdir(exist_ok=True)

def build_package():
    """Build the package with maturin."""
    print("Building package with maturin...")
    
    # Build source distribution
    run_command(["maturin", "sdist", "--out", "dist"])
    
    # Build wheel for current platform
    run_command(["maturin", "build", "--release", "--out", "dist"])

def check_package():
    """Check the package with twine."""
    print("Checking package with twine...")
    run_command(["twine", "check", "dist/*"])

def test_install():
    """Test installation in a clean environment."""
    print("Testing installation...")
    
    # Find the wheel file
    dist_path = Path("dist")
    wheel_files = list(dist_path.glob("*.whl"))
    if not wheel_files:
        print("No wheel file found!")
        return False
    
    wheel_file = wheel_files[0]
    print(f"Testing wheel: {wheel_file}")
    
    # Test installation (dry run)
    result = run_command(["pip", "install", "--dry-run", str(wheel_file)], check=False)
    if result.returncode == 0:
        print("✓ Wheel installation test passed")
        return True
    else:
        print("✗ Wheel installation test failed")
        return False

def upload_to_testpypi():
    """Upload to TestPyPI."""
    print("Uploading to TestPyPI...")
    run_command(["twine", "upload", "--repository", "testpypi", "dist/*"])

def main():
    """Main function."""
    print("🚀 Preparing release for uubed...")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    try:
        clean_dist()
        build_package()
        check_package()
        
        if test_install():
            print("\n✅ Package is ready for release!")
            print("\nNext steps:")
            print("1. Upload to TestPyPI: python scripts/prepare_release.py --test")
            print("2. Test installation from TestPyPI")
            print("3. Upload to PyPI: twine upload dist/*")
        else:
            print("\n❌ Package has issues, please fix before release")
            
    except Exception as e:
        print(f"\n❌ Error during preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()