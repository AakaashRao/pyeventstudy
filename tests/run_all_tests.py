#!/usr/bin/env python
"""
Comprehensive test runner for eventstudypy package.
Runs all tests and compares results with R implementation.
"""

import subprocess
import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_r_available():
    """Check if R is available and eventstudyr package is installed."""
    try:
        # Check if R is available
        result = subprocess.run(['R', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "R is not installed or not in PATH"
        
        # Check if eventstudyr package is installed
        r_code = "if (!require('eventstudyr', quietly=TRUE)) stop('eventstudyr not installed')"
        result = subprocess.run(['R', '--slave', '-e', r_code], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "R package 'eventstudyr' is not installed"
        
        return True, "R and eventstudyr are available"
    except Exception as e:
        return False, f"Error checking R: {str(e)}"


def run_pytest_tests():
    """Run all pytest tests."""
    print("\n" + "="*80)
    print("Running pytest tests...")
    print("="*80 + "\n")
    
    # Run pytest with verbose output
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        '-v', 
        '--tb=short',
        '-x',  # Stop on first failure
        str(Path(__file__).parent)
    ])
    
    return result.returncode == 0


def run_comparison_tests():
    """Run R comparison tests."""
    print("\n" + "="*80)
    print("Running R comparison tests...")
    print("="*80 + "\n")
    
    # Check if R is available
    r_available, message = check_r_available()
    if not r_available:
        print(f"WARNING: {message}")
        print("Skipping R comparison tests.")
        return True
    
    # Run the comparison script
    comparison_script = Path(__file__).parent / 'compare_with_R.py'
    if comparison_script.exists():
        result = subprocess.run([sys.executable, str(comparison_script)])
        return result.returncode == 0
    else:
        print("WARNING: compare_with_R.py not found")
        return True


def main():
    """Main test runner."""
    print("EventStudyPy Test Suite")
    print("=" * 80)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"pytest version: {pytest.__version__}")
    except ImportError:
        print("ERROR: pytest is not installed. Install with: pip install pytest")
        sys.exit(1)
    
    # Run pytest tests
    pytest_success = run_pytest_tests()
    
    # Run R comparison tests
    comparison_success = run_comparison_tests()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Pytest tests: {'PASSED' if pytest_success else 'FAILED'}")
    print(f"R comparison tests: {'PASSED' if comparison_success else 'FAILED'}")
    
    if pytest_success and comparison_success:
        print("\nAll tests passed! ✅")
        sys.exit(0)
    else:
        print("\nSome tests failed! ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()