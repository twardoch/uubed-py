#!/usr/bin/env python3
"""Test runner script for uubed."""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run tests using the manual test runner."""
    test_script = Path(__file__).parent.parent / "run_tests.py"
    
    print("Running uubed test suite...")
    result = subprocess.run([sys.executable, str(test_script)], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def run_cli_tests():
    """Test CLI functionality."""
    print("\nTesting CLI functionality...")
    
    # Test info command
    result = subprocess.run([sys.executable, "-m", "uubed.cli", "info"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ CLI info command works")
        return True
    else:
        print("✗ CLI info command failed")
        print(result.stderr)
        return False

def main():
    """Main test runner."""
    success = True
    
    # Run core tests
    if not run_tests():
        success = False
    
    # Run CLI tests
    if not run_cli_tests():
        success = False
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)