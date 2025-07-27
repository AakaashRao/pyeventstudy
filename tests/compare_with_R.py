"""
Compare Python implementation with R package results
"""

import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import sys

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eventstudypy import event_study, test_linear


def run_r_code(r_code):
    """Run R code and return the output as a string."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        temp_file = f.name
    
    try:
        result = subprocess.run(['R', '--slave', '-f', temp_file], 
                              capture_output=True, text=True)
        return result.stdout
    finally:
        os.unlink(temp_file)


def parse_r_coefficients(r_output, term_prefix=""):
    """Parse coefficients from R output."""
    coefficients = {}
    lines = r_output.strip().split('\n')
    
    in_table = False
    for line in lines:
        # Look for coefficient table header
        if 'term' in line and 'estimate' in line:
            in_table = True
            continue
        
        if in_table and line.strip():
            # Skip non-data lines
            if line.startswith(' ') or '<' in line or '---' in line:
                continue
            
            parts = line.split()
            if len(parts) >= 3:  # term, estimate, std.error, ...
                try:
                    # Extract term and estimate
                    term = parts[1] if parts[0].isdigit() else parts[0]
                    estimate = float(parts[2] if parts[0].isdigit() else parts[1])
                    
                    # Only include if term is a valid variable name (not a number)
                    try:
                        float(term)
                        continue  # Skip if term is a number
                    except ValueError:
                        pass  # Term is not a number, proceed
                    
                    if term_prefix == "" or term.startswith(term_prefix):
                        coefficients[term] = estimate
                except (ValueError, IndexError):
                    continue
    
    return coefficients


def compare_coefficients(python_coefs, r_coefs, tolerance=1e-6):
    """Compare Python and R coefficients."""
    all_match = True
    comparison = []
    
    # Get all unique coefficient names
    all_terms = set(python_coefs.keys()) | set(r_coefs.keys())
    
    for term in sorted(all_terms):
        py_val = python_coefs.get(term, None)
        r_val = r_coefs.get(term, None)
        
        if py_val is None:
            comparison.append(f"  {term:20} Missing in Python    R: {r_val:.6f}")
            all_match = False
        elif r_val is None:
            comparison.append(f"  {term:20} Python: {py_val:.6f}    Missing in R")
            all_match = False
        else:
            diff = abs(py_val - r_val)
            match = diff < tolerance
            status = "✓" if match else "✗"
            comparison.append(f"  {term:20} Python: {py_val:10.6f}    R: {r_val:10.6f}    Diff: {diff:10.8f} {status}")
            if not match:
                all_match = False
    
    return all_match, comparison


# Load example data
data = pd.read_csv('eventstudypy/example_data.csv')

print("Comparison of Python and R EventStudy implementations")
print("=" * 80)

# Test 1: Basic dynamic model
print("\nTest 1: Basic Dynamic Model")
print("-" * 60)

# Python implementation
results1_py = event_study(
    data=data,
    outcomevar="y_base",
    policyvar="z",
    idvar="id",
    timevar="t",
    pre=0,
    post=3,
    normalize=-1
)

py_coefs1 = dict(results1_py['output'].coef())

# R implementation
r_code1 = """
library(eventstudyr)
data <- read.csv('eventstudypy/example_data.csv')

results <- EventStudy(
  estimator = "OLS",
  data = data,
  outcomevar = "y_base",
  policyvar = "z",
  idvar = "id",
  timevar = "t",
  pre = 0, post = 3,
  normalize = -1
)
print(estimatr::tidy(results$output))
"""

r_output1 = run_r_code(r_code1)
r_coefs1 = parse_r_coefficients(r_output1, "z")

print("\nCoefficient comparison:")
match1, comp1 = compare_coefficients(py_coefs1, r_coefs1)
for line in comp1:
    print(line)
print(f"\nAll coefficients match: {match1}")

# Test 2: Dynamic model with controls
print("\n\nTest 2: Dynamic Model with Controls")
print("-" * 60)

# Python implementation
results2_py = event_study(
    data=data,
    outcomevar="y_base",
    policyvar="z",
    idvar="id",
    timevar="t",
    controls="x_r",
    fe=True,
    tfe=True,
    post=3,
    overidpost=5,
    pre=2,
    overidpre=4,
    normalize=-3,
    cluster=True,
    anticipation_effects_normalization=True
)

py_coefs2 = dict(results2_py['output'].coef())

# R implementation
r_code2 = """
library(eventstudyr)
data <- read.csv('eventstudypy/example_data.csv')

results <- EventStudy(
  estimator = "OLS",
  data = data,
  outcomevar = "y_base",
  policyvar = "z",
  idvar = "id",
  timevar = "t",
  controls = "x_r",
  FE = TRUE, TFE = TRUE,
  post = 3, overidpost = 5,
  pre = 2, overidpre = 4,
  normalize = -3,
  cluster = TRUE,
  anticipation_effects_normalization = TRUE
)
print(estimatr::tidy(results$output))
"""

r_output2 = run_r_code(r_code2)
r_coefs2 = parse_r_coefficients(r_output2)

print("\nCoefficient comparison:")
match2, comp2 = compare_coefficients(py_coefs2, r_coefs2)
for line in comp2:
    print(line)
print(f"\nAll coefficients match: {match2}")

# Test 3: Static model
print("\n\nTest 3: Static Model")
print("-" * 60)

# Python implementation
results3_py = event_study(
    data=data,
    outcomevar="y_jump_m",
    policyvar="z",
    idvar="id",
    timevar="t",
    fe=True,
    tfe=True,
    post=0,
    overidpost=0,
    pre=0,
    overidpre=0,
    cluster=True
)

py_coefs3 = dict(results3_py['output'].coef())

# R implementation
r_code3 = """
library(eventstudyr)
data <- read.csv('eventstudypy/example_data.csv')

results <- EventStudy(
  estimator = "OLS",
  data = data,
  outcomevar = "y_jump_m",
  policyvar = "z",
  idvar = "id",
  timevar = "t",
  FE = TRUE, TFE = TRUE,
  post = 0, overidpost = 0,
  pre = 0, overidpre = 0,
  cluster = TRUE
)
print(estimatr::tidy(results$output))
"""

r_output3 = run_r_code(r_code3)
r_coefs3 = parse_r_coefficients(r_output3, "z")

print("\nCoefficient comparison:")
match3, comp3 = compare_coefficients(py_coefs3, r_coefs3)
for line in comp3:
    print(line)
print(f"\nAll coefficients match: {match3}")

# Test hypothesis testing
print("\n\nTest 4: Hypothesis Testing (Pre-trends and Leveling-off)")
print("-" * 60)

# Python hypothesis tests
py_tests = test_linear(results2_py)
print("\nPython hypothesis tests:")
print(py_tests)

# R hypothesis tests
r_code_tests = """
library(eventstudyr)
data <- read.csv('eventstudypy/example_data.csv')

results <- EventStudy(
  estimator = "OLS",
  data = data,
  outcomevar = "y_base",
  policyvar = "z",
  idvar = "id",
  timevar = "t",
  controls = "x_r",
  FE = TRUE, TFE = TRUE,
  post = 3, overidpost = 5,
  pre = 2, overidpre = 4,
  normalize = -3,
  cluster = TRUE,
  anticipation_effects_normalization = TRUE
)

test_results <- TestLinear(results)
print(test_results)
"""

r_output_tests = run_r_code(r_code_tests)
print("\nR hypothesis tests:")
print(r_output_tests)

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Test 1 (Basic Dynamic):           {'PASS' if match1 else 'FAIL'}")
print(f"Test 2 (Dynamic with Controls):   {'PASS' if match2 else 'FAIL'}")
print(f"Test 3 (Static):                  {'PASS' if match3 else 'FAIL'}")

overall_pass = match1 and match2 and match3
print(f"\nOverall Result: {'ALL TESTS PASS' if overall_pass else 'SOME TESTS FAIL'}")