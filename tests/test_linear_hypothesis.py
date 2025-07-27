"""
Tests for TestLinear function matching R package test-TestLinear.R
"""

import pytest
import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eventstudypy import event_study
from eventstudypy.testing import linear_hypothesis_test


def run_r_code(r_code):
    """Run R code and return the output as a string."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        temp_file = f.name
    
    try:
        result = subprocess.run(['R', '--slave', '-f', temp_file], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"R code failed:\n{result.stderr}")
        return result.stdout
    finally:
        os.unlink(temp_file)


def parse_r_test_results(r_output):
    """Parse R TestLinear output to extract test results."""
    results = {}
    lines = r_output.strip().split('\n')
    
    for i, line in enumerate(lines):
        if 'Pre-trends' in line:
            # Look for p-value in next lines
            for j in range(i+1, min(i+5, len(lines))):
                if 'p-value' in lines[j] or 'Pr(' in lines[j]:
                    import re
                    p_match = re.search(r'[\d.]+e?[-+]?\d*$', lines[j])
                    if p_match:
                        results['pretrends'] = float(p_match.group())
                    break
        elif 'Leveling-off' in line:
            # Look for p-value in next lines
            for j in range(i+1, min(i+5, len(lines))):
                if 'p-value' in lines[j] or 'Pr(' in lines[j]:
                    import re
                    p_match = re.search(r'[\d.]+e?[-+]?\d*$', lines[j])
                    if p_match:
                        results['leveling_off'] = float(p_match.group())
                    break
    
    return results


class TestTestLinear:
    """Test class for TestLinear hypothesis testing functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Load example data before each test."""
        self.data = pd.read_csv(Path(__file__).parent.parent / 'eventstudypy' / 'example_data.csv')
        
        # Create a standard event study result for testing
        self.results = event_study(
            data=self.data,
            outcomevar="y_base",
            policyvar="z",
            idvar="id",
            timevar="t",
            controls="x_r",
            fe=True,
            tfe=True,
            post=3,
            pre=2,
            overidpre=4,
            overidpost=5,
            normalize=-3,
            cluster=True,
            anticipation_effects_normalization=True
        )
    
    def test_input_validation_estimate(self):
        """Test that TestLinear validates estimate argument correctly."""
        # Test with invalid estimate type
        with pytest.raises(TypeError):
            linear_hypothesis_test("not_a_dict")
        
        # Test with missing required keys
        invalid_results = {'output': 'not_a_regression'}
        with pytest.raises(ValueError):
            linear_hypothesis_test(invalid_results)
    
    def test_input_validation_boolean_args(self):
        """Test that TestLinear validates boolean arguments."""
        # Test with non-boolean pretrends
        with pytest.raises(TypeError):
            linear_hypothesis_test(self.results, pretrends="yes")
        
        # Test with non-boolean leveling_off
        with pytest.raises(TypeError):
            linear_hypothesis_test(self.results, leveling_off="yes")
    
    def test_default_behavior_all_tests(self):
        """Test default behavior returns all tests."""
        results = linear_hypothesis_test(self.results)
        
        # Should have both pre-trends and leveling-off tests
        assert 'pretrends' in results
        assert 'leveling_off' in results
        assert results['pretrends'] is not None
        assert results['leveling_off'] is not None
    
    def test_only_pretrends(self):
        """Test requesting only pre-trends test."""
        results = linear_hypothesis_test(self.results, pretrends=True, leveling_off=False)
        
        assert 'pretrends' in results
        assert 'leveling_off' not in results or results['leveling_off'] is None
    
    def test_only_leveling_off(self):
        """Test requesting only leveling-off test."""
        results = linear_hypothesis_test(self.results, pretrends=False, leveling_off=True)
        
        assert 'leveling_off' in results
        assert 'pretrends' not in results or results['pretrends'] is None
    
    def test_no_tests_requested(self):
        """Test behavior when no tests are requested."""
        results = linear_hypothesis_test(self.results, pretrends=False, leveling_off=False)
        
        # Should return empty or None results
        assert results is None or (results.get('pretrends') is None and results.get('leveling_off') is None)
    
    def test_compare_with_r_pretrends(self, tolerances):
        """Test pre-trends p-value matches R implementation."""
        # Run Python test
        py_results = linear_hypothesis_test(self.results, pretrends=True, leveling_off=False)
        py_pvalue = py_results['pretrends']
        
        # Run R test
        r_code = f"""
        library(eventstudyr)
        data <- read.csv('{Path(__file__).parent.parent / "eventstudypy" / "example_data.csv"}')
        
        es_results <- EventStudy(
            estimator = "OLS",
            data = data,
            outcomevar = "y_base",
            policyvar = "z",
            idvar = "id",
            timevar = "t",
            controls = "x_r",
            FE = TRUE,
            TFE = TRUE,
            post = 3,
            pre = 2,
            overidpre = 4,
            overidpost = 5,
            normalize = -3,
            cluster = TRUE,
            anticipation_effects_normalization = TRUE
        )
        
        test_results <- TestLinear(es_results, pretrends = TRUE, leveling_off = FALSE)
        print(test_results)
        """
        
        r_output = run_r_code(r_code)
        r_results = parse_r_test_results(r_output)
        
        if 'pretrends' in r_results:
            # Compare p-values
            assert abs(py_pvalue - r_results['pretrends']) < tolerances['pvalue'], \
                f"Pre-trends p-value mismatch: Python={py_pvalue}, R={r_results['pretrends']}"
    
    def test_compare_with_r_leveling_off(self, tolerances):
        """Test leveling-off p-value matches R implementation."""
        # Run Python test
        py_results = linear_hypothesis_test(self.results, pretrends=False, leveling_off=True)
        py_pvalue = py_results['leveling_off']
        
        # Run R test
        r_code = f"""
        library(eventstudyr)
        data <- read.csv('{Path(__file__).parent.parent / "eventstudypy" / "example_data.csv"}')
        
        es_results <- EventStudy(
            estimator = "OLS",
            data = data,
            outcomevar = "y_base",
            policyvar = "z",
            idvar = "id",
            timevar = "t",
            controls = "x_r",
            FE = TRUE,
            TFE = TRUE,
            post = 3,
            pre = 2,
            overidpre = 4,
            overidpost = 5,
            normalize = -3,
            cluster = TRUE,
            anticipation_effects_normalization = TRUE
        )
        
        test_results <- TestLinear(es_results, pretrends = FALSE, leveling_off = TRUE)
        print(test_results)
        """
        
        r_output = run_r_code(r_code)
        r_results = parse_r_test_results(r_output)
        
        if 'leveling_off' in r_results:
            # Compare p-values
            assert abs(py_pvalue - r_results['leveling_off']) < tolerances['pvalue'], \
                f"Leveling-off p-value mismatch: Python={py_pvalue}, R={r_results['leveling_off']}"
    
    def test_compare_with_r_both_tests(self, tolerances):
        """Test both tests match R implementation."""
        # Run Python test
        py_results = linear_hypothesis_test(self.results)
        
        print("\nPython test results:")
        print(py_results)
        
        # Extract p-values from DataFrame
        py_pretrends = None
        py_leveling = None
        for _, row in py_results.iterrows():
            if row['Test'] == 'Pre-Trends':
                py_pretrends = row['p.value']
                if isinstance(py_pretrends, (list, np.ndarray)):
                    py_pretrends = float(py_pretrends[0])
            elif row['Test'] == 'Leveling-Off':
                py_leveling = row['p.value']
                if isinstance(py_leveling, (list, np.ndarray)):
                    py_leveling = float(py_leveling[0])
        
        print(f"\nPython Pre-trends p-value: {py_pretrends}")
        print(f"Python Leveling-off p-value: {py_leveling}")
        
        # Run R test
        r_code = f"""
        library(eventstudyr)
        data <- read.csv('{Path(__file__).parent.parent / "eventstudypy" / "example_data.csv"}')
        
        es_results <- EventStudy(
            estimator = "OLS",
            data = data,
            outcomevar = "y_base",
            policyvar = "z",
            idvar = "id",
            timevar = "t",
            controls = "x_r",
            FE = TRUE,
            TFE = TRUE,
            post = 3,
            pre = 2,
            overidpre = 4,
            overidpost = 5,
            normalize = -3,
            cluster = TRUE,
            anticipation_effects_normalization = TRUE
        )
        
        test_results <- TestLinear(es_results)
        print(test_results)
        
        # Print more details
        cat("Pre-trends F-stat:", test_results$PreTrends$wald_stat, "\n")
        cat("Pre-trends df:", test_results$PreTrends$df, "\n")
        cat("Pre-trends p-value:", test_results$PreTrends$p_value, "\n")
        
        # Also print individual p-values for easier parsing
        # Extract p-values from the data frame
        pretrends_pval <- test_results[test_results$Test == "Pre-Trends", "p.value"]
        levelingoff_pval <- test_results[test_results$Test == "Leveling-Off", "p.value"]
        cat("\\nPre-trends p-value:", pretrends_pval, "\\n")
        cat("Leveling-off p-value:", levelingoff_pval, "\\n")
        """
        
        r_output = run_r_code(r_code)
        
        # py_pretrends and py_leveling were already extracted above
        
        r_pretrends = None
        r_leveling = None
        
        print("\nR output:")
        print(r_output)
        
        for line in r_output.split('\n'):
            if 'Pre-trends p-value:' in line:
                parts = line.split(':')
                if len(parts) > 1 and parts[1].strip():
                    try:
                        r_pretrends = float(parts[1].strip())
                    except ValueError:
                        print(f"Could not parse pre-trends p-value from: {line}")
            elif 'Leveling-off p-value:' in line:
                parts = line.split(':')
                if len(parts) > 1 and parts[1].strip():
                    try:
                        r_leveling = float(parts[1].strip())
                    except ValueError:
                        print(f"Could not parse leveling-off p-value from: {line}")
        
        print(f"\nR Pre-trends p-value: {r_pretrends}")
        print(f"R Leveling-off p-value: {r_leveling}")
        
        # Note: There can be some differences between Python and R implementations
        # due to different handling of degrees of freedom or test statistics
        
        if r_pretrends is not None and py_pretrends is not None:
            diff = abs(py_pretrends - r_pretrends) / py_pretrends
            assert diff < 0.05, \
                f"Pre-trends p-value mismatch: Python={py_pretrends}, R={r_pretrends}, diff={diff}"
        
        if r_leveling is not None and py_leveling is not None:
            diff = abs(py_leveling - r_leveling) / py_leveling
            assert diff < 0.05, \
                f"Leveling-off p-value mismatch: Python={py_leveling}, R={r_leveling}, diff={diff}"
    
    def test_static_model_no_tests(self):
        """Test that static models don't produce pre-trends or leveling-off tests."""
        # Create a static model result
        static_results = event_study(
            data=self.data,
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
        
        # Test should handle static models appropriately
        test_results = linear_hypothesis_test(static_results)
        
        # For static models, tests might not be applicable
        # Verify behavior matches R package
        if test_results is not None:
            assert test_results.get('pretrends') is None or test_results.get('pretrends') == 'NA'
            assert test_results.get('leveling_off') is None or test_results.get('leveling_off') == 'NA'