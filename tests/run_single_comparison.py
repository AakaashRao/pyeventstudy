#!/usr/bin/env python
"""
Run a single comparison test to demonstrate R equality checking.
"""

import subprocess
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Running a single R comparison test to show precision...")
print("=" * 80)

# Run just one test with verbose output
result = subprocess.run([
    sys.executable, '-m', 'pytest', 
    'test_event_study.py::TestEventStudy::test_stata_comparison',
    '-v', '-s'  # -s shows print statements
], cwd=Path(__file__).parent)

print("\n" + "=" * 80)
print("To see all R comparisons, run:")
print("  python compare_with_R.py")
print("\nTo run all tests with R comparison:")
print("  pytest -v -m r_comparison")