"""
Main R parity test suite - ensures Python implementation matches R exactly.
All tests require R and eventstudyr to be installed.
"""

import pytest
import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path

from eventstudypy import event_study, test_linear, add_cis, compute_shifts


class TestRParity:
    """Test exact parity with R eventstudyr package."""
    
    # R utilities kept in same file for easy reference
    @staticmethod
    def run_r_code(r_code):
        """Run R code and return output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
            f.write(r_code)
            temp_file = f.name
        
        try:
            result = subprocess.run(['R', '--slave', '-f', temp_file], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                pytest.fail(f"R code failed. Is R and eventstudyr installed?\n{result.stderr}")
            return result.stdout
        finally:
            os.unlink(temp_file)
    
    def assert_coefficients_equal(self, py_results, r_output, tolerance=None, tolerances=None):
        """Assert Python and R coefficients match."""
        # Use provided tolerance or get from tolerances fixture
        if tolerance is None:
            tolerance = tolerances['coefficient'] if tolerances else 1e-3
        # Parse R output
        r_coefs = {}
        lines = r_output.strip().split('\n')
        
        in_table = False
        for line in lines:
            if 'term' in line and 'estimate' in line:
                in_table = True
                continue
            
            if in_table and line.strip() and not line.startswith(' '):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        term = parts[1] if parts[0].isdigit() else parts[0]
                        estimate = float(parts[2] if parts[0].isdigit() else parts[1])
                        if not term.replace('.', '').replace('-', '').replace('e', '').isdigit():
                            r_coefs[term] = estimate
                    except (ValueError, IndexError):
                        continue
        
        # Get Python coefficients
        py_coefs = dict(py_results['output'].coef())
        
        # Compare
        assert len(r_coefs) > 0, "No R coefficients found - check R output parsing"
        
        for term in r_coefs:
            assert term in py_coefs, f"Term {term} missing in Python results"
            assert abs(py_coefs[term] - r_coefs[term]) < tolerance, \
                f"{term}: Python={py_coefs[term]:.10f}, R={r_coefs[term]:.10f}, diff={abs(py_coefs[term] - r_coefs[term]):.2e}"
    
    def test_basic_dynamic_model(self, example_data, example_data_path, tolerances):
        """Test basic dynamic event study model."""
        # Python
        py_results = event_study(
            data=example_data,
            outcomevar="y_base",
            policyvar="z",
            idvar="id",
            timevar="t",
            pre=0,
            post=3,
            normalize=-1
        )
        
        # R
        r_output = self.run_r_code(f"""
        library(eventstudyr)
        data <- read.csv('{example_data_path}')
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
        """)
        
        self.assert_coefficients_equal(py_results, r_output, tolerances=tolerances)
    
    def test_full_specification(self, example_data, example_data_path, tolerances):
        """Test complete specification with all options."""
        params = {
            'outcomevar': 'y_base',
            'policyvar': 'z',
            'idvar': 'id',
            'timevar': 't',
            'controls': 'x_r',
            'fe': True,
            'tfe': True,
            'post': 3,
            'pre': 2,
            'overidpre': 4,
            'overidpost': 5,
            'normalize': -3,
            'cluster': True,
            'anticipation_effects_normalization': True
        }
        
        # Python
        py_results = event_study(data=example_data, **params)
        
        # R
        r_output = self.run_r_code(f"""
        library(eventstudyr)
        data <- read.csv('{example_data_path}')
        results <- EventStudy(
            estimator = "OLS",
            data = data,
            outcomevar = "{params['outcomevar']}",
            policyvar = "{params['policyvar']}",
            idvar = "{params['idvar']}",
            timevar = "{params['timevar']}",
            controls = "{params['controls']}",
            FE = {str(params['fe']).upper()},
            TFE = {str(params['tfe']).upper()},
            post = {params['post']},
            pre = {params['pre']},
            overidpre = {params['overidpre']},
            overidpost = {params['overidpost']},
            normalize = {params['normalize']},
            cluster = {str(params['cluster']).upper()},
            anticipation_effects_normalization = {str(params['anticipation_effects_normalization']).upper()}
        )
        print(estimatr::tidy(results$output))
        """)
        
        self.assert_coefficients_equal(py_results, r_output, tolerances=tolerances)
    
    def test_hypothesis_tests(self, example_data, example_data_path, tolerances):
        """Test hypothesis testing matches R."""
        # First run event study
        params = {
            'outcomevar': 'y_base',
            'policyvar': 'z',
            'idvar': 'id',
            'timevar': 't',
            'controls': 'x_r',
            'fe': True,
            'tfe': True,
            'post': 3,
            'pre': 2,
            'overidpre': 4,
            'overidpost': 5,
            'normalize': -3,
            'cluster': True
        }
        
        py_results = event_study(data=example_data, **params)
        py_tests = test_linear(py_results)
        
        # R
        r_output = self.run_r_code(f"""
        library(eventstudyr)
        data <- read.csv('{example_data_path}')
        results <- EventStudy(
            estimator = "OLS", data = data,
            outcomevar = "{params['outcomevar']}", policyvar = "{params['policyvar']}",
            idvar = "{params['idvar']}", timevar = "{params['timevar']}",
            controls = "{params['controls']}",
            FE = TRUE, TFE = TRUE,
            post = {params['post']}, pre = {params['pre']},
            overidpre = {params['overidpre']}, overidpost = {params['overidpost']},
            normalize = {params['normalize']}, cluster = TRUE
        )
        test_results <- TestLinear(results)
        # TestLinear returns a data.frame, access the p-values by row
        pretrends_row <- test_results[test_results$Test == "Pre-Trends", ]
        leveling_row <- test_results[test_results$Test == "Leveling-Off", ]
        
        if (nrow(pretrends_row) > 0) {{
            cat("Pre-trends p-value:", pretrends_row$p.value, "\\n")
        }}
        
        if (nrow(leveling_row) > 0) {{
            cat("Leveling-off p-value:", leveling_row$p.value, "\\n")
        }}
        """)
        
        # Parse R p-values
        r_pretrends = None
        r_leveling = None
        
        
        for line in r_output.split('\n'):
            if 'Pre-trends p-value:' in line:
                parts = line.split(':')
                if len(parts) > 1 and parts[1].strip():
                    r_pretrends = float(parts[1].strip())
            elif 'Leveling-off p-value:' in line:
                parts = line.split(':')
                if len(parts) > 1 and parts[1].strip():
                    r_leveling = float(parts[1].strip())
        
        assert r_pretrends is not None, "Failed to parse R pre-trends p-value"
        assert r_leveling is not None, "Failed to parse R leveling-off p-value"
        
        assert abs(py_tests['pretrends'] - r_pretrends) < tolerances['pvalue'], \
            f"Pre-trends p-value mismatch: Python={py_tests['pretrends']:.6f}, R={r_pretrends:.6f}"
        assert abs(py_tests['leveling_off'] - r_leveling) < tolerances['pvalue'], \
            f"Leveling-off p-value mismatch: Python={py_tests['leveling_off']:.6f}, R={r_leveling:.6f}"
    
    def test_confidence_intervals(self, example_data, example_data_path, tolerances):
        """Test confidence interval calculation matches R."""
        # Run event study
        py_results = event_study(
            data=example_data,
            outcomevar="y_base",
            policyvar="z",
            idvar="id",
            timevar="t",
            fe=True,
            tfe=True,
            post=2,
            pre=2,
            normalize=-1,
            cluster=True
        )
        
        # Add CIs
        py_with_cis = add_cis(py_results)
        
        # R
        r_output = self.run_r_code(f"""
        library(eventstudyr)
        data <- read.csv('{example_data_path}')
        results <- EventStudy(
            estimator = "OLS", data = data,
            outcomevar = "y_base", policyvar = "z",
            idvar = "id", timevar = "t",
            FE = TRUE, TFE = TRUE,
            post = 2, pre = 2, normalize = -1,
            cluster = TRUE
        )
        # Calculate CIs manually using the coefficients and SEs
        with_cis <- estimatr::tidy(results$output)
        with_cis$conf.low <- with_cis$estimate - 1.96 * with_cis$std.error
        with_cis$conf.high <- with_cis$estimate + 1.96 * with_cis$std.error
        for(i in 1:nrow(with_cis)) {{
            if(grepl("^z_", with_cis$term[i])) {{
                cat(sprintf("%s: %.10f %.10f %.10f\\n", 
                           with_cis$term[i], 
                           with_cis$estimate[i],
                           with_cis$conf.low[i], 
                           with_cis$conf.high[i]))
            }}
        }}
        """)
        
        # Parse and compare
        for line in r_output.strip().split('\n'):
            if line.startswith('z_'):
                parts = line.split()
                term = parts[0].rstrip(':')
                r_est = float(parts[1])
                r_low = float(parts[2])
                r_high = float(parts[3])
                
                if term in py_with_cis.index:
                    py_est = py_with_cis.loc[term, 'estimate']
                    py_low = py_with_cis.loc[term, 'conf.low']
                    py_high = py_with_cis.loc[term, 'conf.high']
                    
                    assert abs(py_est - r_est) < tolerances['coefficient'], f"Estimate mismatch for {term}"
                    assert abs(py_low - r_low) < tolerances['coefficient'], f"CI lower mismatch for {term}"
                    assert abs(py_high - r_high) < tolerances['coefficient'], f"CI upper mismatch for {term}"
    
    @pytest.mark.parametrize("fe,tfe,cluster", [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (False, False, True),
        # Skip (True, True, False) - R requires cluster=TRUE when FE=TRUE
        (False, True, False),
        # Skip (True, False, False) - R requires cluster=TRUE when FE=TRUE  
        (False, False, False),
    ])
    def test_all_fe_cluster_combinations(self, example_data, example_data_path, fe, tfe, cluster, tolerances):
        """Test all FE/TFE/cluster combinations match R."""
        # Python
        py_results = event_study(
            data=example_data,
            outcomevar="y_base",
            policyvar="z",
            idvar="id",
            timevar="t",
            fe=fe,
            tfe=tfe,
            cluster=cluster,
            post=2,
            pre=1,
            normalize=-1
        )
        
        # R
        r_output = self.run_r_code(f"""
        library(eventstudyr)
        data <- read.csv('{example_data_path}')
        results <- EventStudy(
            estimator = "OLS",
            data = data,
            outcomevar = "y_base",
            policyvar = "z",
            idvar = "id",
            timevar = "t",
            FE = {str(fe).upper()},
            TFE = {str(tfe).upper()},
            cluster = {str(cluster).upper()},
            post = 2,
            pre = 1,
            normalize = -1
        )
        print(estimatr::tidy(results$output))
        """)
        
        # Debug: print both outputs for comparison
        print(f"\nDebug - FE={fe}, TFE={tfe}, cluster={cluster}")
        print(f"Python coefficients:")
        if isinstance(py_results, dict) and 'output' in py_results:
            py_output = py_results['output']
            for coef in py_output.coef().index:
                if coef.startswith('z_'):
                    print(f"  {coef}: {py_output.coef()[coef]:.6f}")
        print(f"\nR output:\n{r_output}")
        
        self.assert_coefficients_equal(py_results, r_output, tolerances=tolerances)