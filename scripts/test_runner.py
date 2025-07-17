#!/usr/bin/env python3
# this_file: scripts/test_runner.py
"""
Comprehensive test runner for uubed package.
This script runs all tests with coverage reporting and handles different test scenarios.
"""

import subprocess
import sys
import os
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

def install_test_dependencies():
    """Install test dependencies."""
    print("📦 Installing test dependencies...")
    cmd = [sys.executable, "-m", "pip", "install", "-e", ".[test]"]
    result = run_command(cmd)
    if result.returncode != 0:
        print("❌ Failed to install test dependencies")
        sys.exit(1)

def run_pytest(args: List[str] = None) -> int:
    """Run pytest with coverage."""
    print("🧪 Running pytest with coverage...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=uubed",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--junitxml=test-results.xml"
    ]
    
    if args:
        cmd.extend(args)
    
    result = run_command(cmd)
    return result.returncode

def run_linting():
    """Run code linting."""
    print("🔍 Running code linting...")
    
    # Run ruff check
    cmd = [sys.executable, "-m", "ruff", "check", "src", "tests"]
    result = run_command(cmd)
    if result.returncode != 0:
        print("❌ Linting failed")
        return False
    
    # Run ruff format check
    cmd = [sys.executable, "-m", "ruff", "format", "--check", "src", "tests"]
    result = run_command(cmd)
    if result.returncode != 0:
        print("❌ Format check failed")
        return False
    
    print("✅ Linting passed")
    return True

def run_type_checking():
    """Run type checking with mypy."""
    print("🔍 Running type checking...")
    
    # Install mypy if not present
    try:
        subprocess.run([sys.executable, "-m", "mypy", "--version"], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Installing mypy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mypy"], check=True)
    
    cmd = [sys.executable, "-m", "mypy", "src/uubed"]
    result = run_command(cmd)
    if result.returncode != 0:
        print("❌ Type checking failed")
        return False
    
    print("✅ Type checking passed")
    return True

def test_package_installation():
    """Test that the package can be installed and imported."""
    print("📦 Testing package installation...")
    
    # Test basic import
    cmd = [sys.executable, "-c", "import uubed; print(f'uubed version: {uubed.__version__}')"]
    result = run_command(cmd)
    if result.returncode != 0:
        print("❌ Package import failed")
        return False
    
    # Test CLI
    cmd = [sys.executable, "-m", "uubed.cli", "info"]
    result = run_command(cmd)
    if result.returncode != 0:
        print("❌ CLI test failed")
        return False
    
    print("✅ Package installation test passed")
    return True

def generate_coverage_report():
    """Generate and display coverage report."""
    print("📊 Generating coverage report...")
    
    # Check if coverage data exists
    if not Path(".coverage").exists():
        print("⚠️  No coverage data found")
        return
    
    # Generate coverage report
    cmd = [sys.executable, "-m", "coverage", "report", "--show-missing"]
    result = run_command(cmd)
    
    if result.returncode == 0:
        print("✅ Coverage report generated successfully")
        print("📁 HTML coverage report available at: htmlcov/index.html")
    else:
        print("❌ Failed to generate coverage report")

def main():
    """Main test runner function."""
    print("🧪 Starting uubed test suite...")
    
    # Parse command line arguments
    pytest_args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Install test dependencies
    install_test_dependencies()
    
    # Test package installation
    if not test_package_installation():
        sys.exit(1)
    
    # Run linting
    if not run_linting():
        print("⚠️  Linting failed, but continuing with tests...")
    
    # Run type checking
    if not run_type_checking():
        print("⚠️  Type checking failed, but continuing with tests...")
    
    # Run pytest
    test_exit_code = run_pytest(pytest_args)
    
    # Generate coverage report
    generate_coverage_report()
    
    # Print summary
    print("\n" + "="*60)
    print("🧪 TEST SUMMARY")
    print("="*60)
    
    if test_exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {test_exit_code}")
    
    print("📊 Check coverage report: htmlcov/index.html")
    print("📄 JUnit XML report: test-results.xml")
    
    return test_exit_code

if __name__ == "__main__":
    sys.exit(main())