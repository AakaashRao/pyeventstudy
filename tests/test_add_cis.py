"""
Tests for AddCIs function matching R package test-AddCIs.R
"""

import pytest
import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from scipy import stats

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eventstudypy import event_study, add_cis


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


class TestAddCIs:
    """Test class for AddCIs confidence interval functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Load example data and create event study results before each test."""
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
    
    def test_calculation_accuracy_95_percent(self, tolerances):
        """Test that AddCIs correctly calculates 95% confidence intervals."""
        # Add CIs with default 95% confidence level
        results_with_cis = add_cis(self.results)
        
        # Check that CI columns were added
        assert 'conf.low' in results_with_cis
        assert 'conf.high' in results_with_cis
        
        # Verify CI calculation: CI = estimate ± 1.96 * std.error (for 95% CI)
        z_score = 1.96  # For 95% CI
        
        for idx in results_with_cis.index:
            estimate = results_with_cis.loc[idx, 'estimate']
            std_error = results_with_cis.loc[idx, 'std.error']
            conf_low = results_with_cis.loc[idx, 'conf.low']
            conf_high = results_with_cis.loc[idx, 'conf.high']
            
            expected_low = estimate - z_score * std_error
            expected_high = estimate + z_score * std_error
            
            assert abs(conf_low - expected_low) < tolerances['coefficient'], \
                f"CI lower bound mismatch for {idx}: calculated={conf_low}, expected={expected_low}"
            assert abs(conf_high - expected_high) < tolerances['coefficient'], \
                f"CI upper bound mismatch for {idx}: calculated={conf_high}, expected={expected_high}"
    
    def test_custom_confidence_levels(self, tolerances):
        """Test AddCIs with custom confidence levels."""
        # Test 90% confidence interval
        results_90 = add_cis(self.results, conf_level=0.90)
        
        # For 90% CI, z-score is approximately 1.645
        z_score_90 = 1.645
        
        for idx in results_90.index:
            estimate = results_90.loc[idx, 'estimate']
            std_error = results_90.loc[idx, 'std.error']
            conf_low = results_90.loc[idx, 'conf.low']
            conf_high = results_90.loc[idx, 'conf.high']
            
            expected_low = estimate - z_score_90 * std_error
            expected_high = estimate + z_score_90 * std_error
            
            # Allow slightly more tolerance for different z-score approximations
            assert abs(conf_low - expected_low) < tolerances['se'], \
                f"90% CI lower bound mismatch for {idx}"
            assert abs(conf_high - expected_high) < tolerances['se'], \
                f"90% CI upper bound mismatch for {idx}"
        
        # Test 99% confidence interval
        results_99 = add_cis(self.results, conf_level=0.99)
        
        # For 99% CI, z-score is approximately 2.576
        z_score_99 = 2.576
        
        for idx in results_99.index:
            estimate = results_99.loc[idx, 'estimate']
            std_error = results_99.loc[idx, 'std.error']
            conf_low = results_99.loc[idx, 'conf.low']
            conf_high = results_99.loc[idx, 'conf.high']
            
            expected_low = estimate - z_score_99 * std_error
            expected_high = estimate + z_score_99 * std_error
            
            assert abs(conf_low - expected_low) < 0.01, \
                f"99% CI lower bound mismatch for {idx}"
            assert abs(conf_high - expected_high) < tolerances['se'], \
                f"99% CI upper bound mismatch for {idx}"
    
    def test_input_validation_estimate_type(self):
        """Test that AddCIs validates estimate argument type."""
        # Test with invalid estimate type
        with pytest.raises(TypeError):
            add_cis("not_a_dict")
        
        # Test with missing output key
        with pytest.raises(ValueError):
            add_cis({'some_key': 'value'})
    
    def test_input_validation_required_columns(self):
        """Test that AddCIs checks for required columns."""
        # Create a mock result with missing columns
        mock_results = {
            'output': pd.DataFrame({
                'term': ['z_fd_lead1', 'z_fd'],
                'estimate': [0.1, 0.2]
                # Missing std.error column
            })
        }
        
        with pytest.raises(ValueError, match="std.error"):
            add_cis(mock_results)
    
    def test_input_validation_conf_level(self):
        """Test that AddCIs validates conf_level parameter."""
        # Test conf_level < 0
        with pytest.raises(ValueError, match="conf_level must be between 0 and 1"):
            add_cis(self.results, conf_level=-0.1)
        
        # Test conf_level > 1
        with pytest.raises(ValueError, match="conf_level must be between 0 and 1"):
            add_cis(self.results, conf_level=1.5)
        
        # Test conf_level = 0
        with pytest.raises(ValueError, match="conf_level must be between 0 and 1"):
            add_cis(self.results, conf_level=0)
        
        # Test conf_level = 1
        with pytest.raises(ValueError, match="conf_level must be between 0 and 1"):
            add_cis(self.results, conf_level=1)
    
    def test_compare_with_r_implementation(self, tolerances):
        """Test that Python AddCIs matches R implementation.
        
        Note: Due to implementation differences between Python and R:
        - Coefficients should match exactly (tolerance: 1e-3)
        - Standard errors may differ slightly due to different methods (tolerance: 5e-3)
        - P-values may differ due to normal vs t-distribution (tolerance: 15e-3)
        - Confidence intervals may differ due to distribution differences (tolerance: 2.5%)
        """
        # Add CIs in Python
        py_results = add_cis(self.results)
        
        # Run R code
        r_code = f"""
        library(eventstudyr)
        library(estimatr)
        data <- read.csv('{Path(__file__).parent.parent / "eventstudypy" / "example_data.csv"}')
        
        # Run event study
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
        
        # Get tidy output from the model
        df_estimates_tidy <- estimatr::tidy(es_results$output)
        
        # Filter for event study coefficients first
        eventstudy_coeffs <- es_results$arguments$eventstudy_coefficients
        results_with_cis <- df_estimates_tidy[df_estimates_tidy$term %in% eventstudy_coeffs, ]
        
        # Print results with all statistics
        for(i in 1:nrow(results_with_cis)) {{
            cat(sprintf("%s: estimate=%.10f std.error=%.10f p.value=%.10f low=%.10f high=%.10f\\n", 
                       results_with_cis$term[i], 
                       results_with_cis$estimate[i],
                       results_with_cis$std.error[i],
                       results_with_cis$p.value[i],
                       results_with_cis$conf.low[i], 
                       results_with_cis$conf.high[i]))
        }}
        """
        
        r_output = run_r_code(r_code)
        
        # Parse R output
        r_stats = {}
        for line in r_output.strip().split('\n'):
            if ':' in line and 'estimate=' in line:
                parts = line.split(':')
                term = parts[0].strip()
                stats_parts = parts[1].split()
                r_stats[term] = {
                    'estimate': float(stats_parts[0].split('=')[1]),
                    'std.error': float(stats_parts[1].split('=')[1]),
                    'p.value': float(stats_parts[2].split('=')[1]),
                    'low': float(stats_parts[3].split('=')[1]),
                    'high': float(stats_parts[4].split('=')[1])
                }
        
        # Get Python model for additional statistics
        py_model = self.results['output']
        
        # Compare all statistics with strict tolerances
        for term in r_stats:
            if term in py_results.index:
                # Get Python values
                py_estimate = py_results.loc[term, 'estimate']
                py_std_error = py_results.loc[term, 'std.error']
                py_low = py_results.loc[term, 'conf.low']
                py_high = py_results.loc[term, 'conf.high']
                
                # Calculate p-value for Python (two-tailed test)
                py_t_stat = py_estimate / py_std_error
                py_p_value = 2 * (1 - stats.norm.cdf(abs(py_t_stat)))
                
                # Get R values
                r_estimate = r_stats[term]['estimate']
                r_std_error = r_stats[term]['std.error']
                r_p_value = r_stats[term]['p.value']
                r_low = r_stats[term]['low']
                r_high = r_stats[term]['high']
                
                # Compare coefficients
                assert abs(py_estimate - r_estimate) < tolerances['coefficient'], \
                    f"Coefficient mismatch for {term}: Python={py_estimate}, R={r_estimate}, diff={abs(py_estimate - r_estimate)}"
                
                # Compare standard errors (slightly more relaxed due to different SE calculation methods)
                assert abs(py_std_error - r_std_error) < tolerances['se'], \
                    f"Standard error mismatch for {term}: Python={py_std_error}, R={r_std_error}, diff={abs(py_std_error - r_std_error)}"
                
                # Compare p-values (may differ due to normal vs t-distribution differences)
                assert abs(py_p_value - r_p_value) < 3 * tolerances['pvalue'], \
                    f"P-value mismatch for {term}: Python={py_p_value}, R={r_p_value}, diff={abs(py_p_value - r_p_value)}"
                
                # For confidence intervals, we can be more lenient since methods differ
                # (Python uses normal distribution, R uses t-distribution)
                ci_tolerance = 0.025  # 2.5% tolerance for CI differences
                assert abs(py_low - r_low) < ci_tolerance, \
                    f"CI lower bound mismatch for {term}: Python={py_low}, R={r_low}, diff={abs(py_low - r_low)}"
                assert abs(py_high - r_high) < ci_tolerance, \
                    f"CI upper bound mismatch for {term}: Python={py_high}, R={r_high}, diff={abs(py_high - r_high)}"
    
    def test_preserves_original_data(self, tolerances):
        """Test that AddCIs preserves all original columns and data."""
        # Get the original coefficient data
        results_with_cis = add_cis(self.results)
        
        # Check required columns are present
        required_cols = ['term', 'estimate', 'std.error', 'conf.low', 'conf.high']
        for col in required_cols:
            assert col in results_with_cis.columns, f"Required column {col} is missing"
        
        # Check that estimates match the original model
        model = self.results['output']
        coef_names = self.results['arguments']['eventstudy_coefficients']
        
        for name in coef_names:
            if name in results_with_cis.index:
                assert abs(results_with_cis.loc[name, 'estimate'] - model.coef()[name]) < tolerances['coefficient'], \
                    f"Estimate for {name} was changed"
                assert abs(results_with_cis.loc[name, 'std.error'] - model.se()[name]) < tolerances['coefficient'], \
                    f"Standard error for {name} was changed"