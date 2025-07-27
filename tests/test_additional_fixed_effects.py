"""
Test additional fixed effects functionality in event_study
"""

import pytest
import numpy as np
import pandas as pd
from eventstudypy import event_study


class TestAdditionalFixedEffects:
    """Test that additional fixed effects work correctly."""
    
    @pytest.fixture
    def simulated_data_with_fe(self):
        """Create simulated data with known fixed effects structure."""
        np.random.seed(12345)
        
        # Parameters
        n_units = 50
        n_periods = 20
        n_regions = 4
        n_industries = 3
        
        # Create panel structure
        data = []
        for i in range(n_units):
            for t in range(n_periods):
                data.append({
                    'id': i,
                    't': t,
                    'region': f'region_{i % n_regions}',
                    'industry': f'industry_{i % n_industries}'
                })
        
        df = pd.DataFrame(data)
        
        # Create treatment variable (staggered adoption)
        treatment_time = np.random.choice(range(8, 15), size=n_units)
        df['treated'] = 0
        for i in range(n_units):
            df.loc[(df['id'] == i) & (df['t'] >= treatment_time[i]), 'treated'] = 1
        
        # Create fixed effects
        region_effects = {'region_0': 2.0, 'region_1': -1.5, 'region_2': 0.5, 'region_3': 3.0}
        industry_effects = {'industry_0': 1.0, 'industry_1': -0.5, 'industry_2': 2.5}
        
        # Create outcome with known coefficients
        # True treatment effect: immediate jump of 2.0, then decay by 0.3 per period
        df['time_since_treatment'] = -999
        for i in range(n_units):
            mask = df['id'] == i
            df.loc[mask, 'time_since_treatment'] = df.loc[mask, 't'] - treatment_time[i]
        
        # Generate outcome
        df['y'] = (
            5.0 +  # intercept
            df['region'].map(region_effects) +  # region FE
            df['industry'].map(industry_effects) +  # industry FE
            0.1 * df['id'] +  # unit FE
            0.05 * df['t'] +  # time trend
            np.where(df['time_since_treatment'] >= 0, 
                     2.0 - 0.3 * df['time_since_treatment'], 0) +  # treatment effect
            np.random.normal(0, 0.5, size=len(df))  # noise
        )
        
        return df
    
    def test_fixed_effects_vs_controls_exact_match(self, simulated_data_with_fe):
        """Test that specifying variables as fixed effects gives same results as controls."""
        df = simulated_data_with_fe
        
        # Create dummy variables for region and industry
        region_dummies = pd.get_dummies(df['region'], prefix='region', drop_first=True)
        industry_dummies = pd.get_dummies(df['industry'], prefix='industry', drop_first=True)
        
        # Add dummies to dataframe
        for col in region_dummies.columns:
            df[col] = region_dummies[col]
        for col in industry_dummies.columns:
            df[col] = industry_dummies[col]
        
        control_vars = list(region_dummies.columns) + list(industry_dummies.columns)
        
        # Run event study with categorical variables as fixed effects
        result_fe = event_study(
            data=df,
            outcomevar='y',
            policyvar='treated',
            idvar='id',
            timevar='t',
            fe=True,
            tfe=True,
            fixed_effects=['region', 'industry'],
            post=3,
            pre=2,
            normalize=-1,
            cluster=True
        )
        
        # Run event study with dummy variables as controls
        result_controls = event_study(
            data=df,
            outcomevar='y',
            policyvar='treated',
            idvar='id',
            timevar='t',
            fe=True,
            tfe=True,
            controls=control_vars,
            post=3,
            pre=2,
            normalize=-1,
            cluster=True
        )
        
        # Extract coefficients for event study variables
        fe_coefs = {}
        control_coefs = {}
        
        for coef_name in result_fe['output'].coef().index:
            if coef_name.startswith('treated_'):
                fe_coefs[coef_name] = result_fe['output'].coef()[coef_name]
        
        for coef_name in result_controls['output'].coef().index:
            if coef_name.startswith('treated_'):
                control_coefs[coef_name] = result_controls['output'].coef()[coef_name]
        
        # Check that we have the same event study coefficients
        assert set(fe_coefs.keys()) == set(control_coefs.keys()), \
            "Different event study coefficients returned"
        
        # Check that coefficients match (within numerical tolerance)
        for coef_name in fe_coefs:
            assert abs(fe_coefs[coef_name] - control_coefs[coef_name]) < 1e-10, \
                f"Coefficient {coef_name} differs: FE={fe_coefs[coef_name]}, Controls={control_coefs[coef_name]}"
    
    def test_multiple_fixed_effects_combinations(self, simulated_data_with_fe):
        """Test various combinations of fixed effects."""
        df = simulated_data_with_fe
        
        # Test 1: Only additional fixed effects (no unit/time FE)
        result1 = event_study(
            data=df,
            outcomevar='y',
            policyvar='treated',
            idvar='id',
            timevar='t',
            fe=False,
            tfe=False,
            fixed_effects=['region', 'industry'],
            post=2,
            pre=1,
            normalize=-1,
            cluster=True
        )
        
        # Test 2: Unit FE + additional FE (no time FE)
        result2 = event_study(
            data=df,
            outcomevar='y',
            policyvar='treated',
            idvar='id',
            timevar='t',
            fe=True,
            tfe=False,
            fixed_effects=['region', 'industry'],
            post=2,
            pre=1,
            normalize=-1,
            cluster=True
        )
        
        # Test 3: Time FE + additional FE (no unit FE)
        result3 = event_study(
            data=df,
            outcomevar='y',
            policyvar='treated',
            idvar='id',
            timevar='t',
            fe=False,
            tfe=True,
            fixed_effects=['region', 'industry'],
            post=2,
            pre=1,
            normalize=-1,
            cluster=True
        )
        
        # Test 4: All fixed effects
        result4 = event_study(
            data=df,
            outcomevar='y',
            policyvar='treated',
            idvar='id',
            timevar='t',
            fe=True,
            tfe=True,
            fixed_effects=['region', 'industry'],
            post=2,
            pre=1,
            normalize=-1,
            cluster=True
        )
        
        # Verify formulas are correct
        assert '|region+industry' in result1['output']._fml
        assert '|id+region+industry' in result2['output']._fml
        assert '|t+region+industry' in result3['output']._fml
        assert '|id+t+region+industry' in result4['output']._fml
        
        # All should return results
        assert result1['output'] is not None
        assert result2['output'] is not None
        assert result3['output'] is not None
        assert result4['output'] is not None
    
    def test_single_vs_list_fixed_effects(self, simulated_data_with_fe):
        """Test that passing a single string vs list with one element gives same results."""
        df = simulated_data_with_fe
        
        # Single string
        result1 = event_study(
            data=df,
            outcomevar='y',
            policyvar='treated',
            idvar='id',
            timevar='t',
            fe=True,
            tfe=True,
            fixed_effects='region',  # string
            post=2,
            pre=1,
            normalize=-1,
            cluster=True
        )
        
        # List with one element
        result2 = event_study(
            data=df,
            outcomevar='y',
            policyvar='treated',
            idvar='id',
            timevar='t',
            fe=True,
            tfe=True,
            fixed_effects=['region'],  # list
            post=2,
            pre=1,
            normalize=-1,
            cluster=True
        )
        
        # Extract coefficients
        coefs1 = {k: v for k, v in result1['output'].coef().items() if k.startswith('treated_')}
        coefs2 = {k: v for k, v in result2['output'].coef().items() if k.startswith('treated_')}
        
        # Should be identical
        assert set(coefs1.keys()) == set(coefs2.keys())
        for k in coefs1:
            assert abs(coefs1[k] - coefs2[k]) < 1e-10
    
    def test_r_parity_with_additional_fixed_effects(self, simulated_data_with_fe):
        """Test that Python matches R when using additional fixed effects."""
        import subprocess
        import tempfile
        import os
        
        df = simulated_data_with_fe
        
        # Save data to temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name
        
        try:
            # Run Python version
            py_result = event_study(
                data=df,
                outcomevar='y',
                policyvar='treated',
                idvar='id',
                timevar='t',
                fe=True,
                tfe=True,
                fixed_effects=['region', 'industry'],
                post=3,
                pre=2,
                normalize=-1,
                cluster=True
            )
            
            # Run R version
            r_code = f"""
            library(eventstudyr)
            
            # Load data
            data <- read.csv('{temp_path}')
            
            # Convert region and industry to factors
            data$region <- as.factor(data$region)
            data$industry <- as.factor(data$industry)
            
            # Run event study with additional fixed effects
            # Note: R's EventStudy doesn't have a direct fixed_effects parameter
            # So we need to include them as controls after converting to dummies
            library(fastDummies)
            data_with_dummies <- dummy_cols(data, select_columns = c("region", "industry"), 
                                           remove_first_dummy = TRUE)
            
            # Get the dummy column names
            region_dummies <- grep("^region_", names(data_with_dummies), value = TRUE)
            industry_dummies <- grep("^industry_", names(data_with_dummies), value = TRUE)
            control_vars <- c(region_dummies, industry_dummies)
            
            # Run EventStudy
            results <- EventStudy(
                estimator = "OLS",
                data = data_with_dummies,
                outcomevar = "y",
                policyvar = "treated",
                idvar = "id",
                timevar = "t",
                FE = TRUE,
                TFE = TRUE,
                controls = control_vars,
                post = 3,
                pre = 2,
                normalize = -1,
                cluster = TRUE
            )
            
            # Print coefficients
            library(estimatr)
            tidy_results <- tidy(results$output)
            for(i in 1:nrow(tidy_results)) {{
                if(grepl("^treated_", tidy_results$term[i])) {{
                    cat(sprintf("%s: %.10f\\n", tidy_results$term[i], tidy_results$estimate[i]))
                }}
            }}
            """
            
            # Run R code
            result = subprocess.run(
                ['R', '--vanilla', '--quiet'],
                input=r_code,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                pytest.skip(f"R execution failed: {result.stderr}")
            
            # Parse R output
            r_coefs = {}
            for line in result.stdout.split('\n'):
                if line.startswith('treated_'):
                    parts = line.split(':')
                    if len(parts) == 2:
                        coef_name = parts[0].strip()
                        coef_value = float(parts[1].strip())
                        r_coefs[coef_name] = coef_value
            
            # Compare coefficients
            py_coefs = {k: v for k, v in py_result['output'].coef().items() 
                       if k.startswith('treated_')}
            
            # Check we have same coefficients
            assert set(py_coefs.keys()) == set(r_coefs.keys()), \
                f"Different coefficients: Python={set(py_coefs.keys())}, R={set(r_coefs.keys())}"
            
            # Check values match (with reasonable tolerance for numerical differences)
            for coef_name in py_coefs:
                py_val = py_coefs[coef_name]
                r_val = r_coefs[coef_name]
                diff = abs(py_val - r_val)
                assert diff < 0.001, \
                    f"{coef_name}: Python={py_val:.6f}, R={r_val:.6f}, diff={diff:.6f}"
                    
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)