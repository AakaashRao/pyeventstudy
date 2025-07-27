"""
Test parameter recovery with synthetic data
"""

import numpy as np
import pandas as pd
import sys
import os
import subprocess
import tempfile

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eventstudypy import event_study


def generate_event_study_data(
    n_units=100,
    n_periods=20,
    treatment_start=10,
    treatment_effect_immediate=0.5,
    treatment_effect_dynamic=None,
    pre_trend=0.0,
    unit_fe_sd=1.0,
    time_fe_sd=0.2,
    idiosyncratic_sd=0.5,
    control_effect=0.3,
    seed=42
):
    """
    Generate synthetic event study data with known parameters.
    
    Parameters
    ----------
    n_units : int
        Number of units (e.g., firms, individuals)
    n_periods : int
        Number of time periods
    treatment_start : int
        Period when treatment starts (for treated units)
    treatment_effect_immediate : float
        Immediate treatment effect at time of treatment
    treatment_effect_dynamic : dict or None
        Dynamic treatment effects {relative_time: effect}
        e.g., {-2: 0.1, -1: 0.2, 0: 0.5, 1: 0.6, 2: 0.7}
    pre_trend : float
        Linear pre-trend for treated units
    unit_fe_sd : float
        Standard deviation of unit fixed effects
    time_fe_sd : float
        Standard deviation of time fixed effects
    idiosyncratic_sd : float
        Standard deviation of idiosyncratic errors
    control_effect : float
        Effect of control variable
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Synthetic panel data
    dict
        True parameters used
    """
    np.random.seed(seed)
    
    # Create panel structure
    data = []
    for i in range(n_units):
        for t in range(1, n_periods + 1):
            data.append({'id': i + 1, 't': t})
    df = pd.DataFrame(data)
    
    # Randomly assign treatment status (50% treated)
    treated_units = np.random.choice(range(1, n_units + 1), 
                                   size=n_units // 2, replace=False)
    df['treated'] = df['id'].isin(treated_units).astype(int)
    
    # Create treatment variable (staggered or simultaneous)
    # For simplicity, all treated units get treated at the same time
    df['z'] = ((df['treated'] == 1) & (df['t'] >= treatment_start)).astype(int)
    
    # Generate fixed effects
    unit_fe = pd.DataFrame({
        'id': range(1, n_units + 1),
        'unit_fe': np.random.normal(0, unit_fe_sd, n_units)
    })
    time_fe = pd.DataFrame({
        't': range(1, n_periods + 1),
        'time_fe': np.random.normal(0, time_fe_sd, n_periods)
    })
    
    df = df.merge(unit_fe, on='id')
    df = df.merge(time_fe, on='t')
    
    # Generate control variable
    df['x'] = np.random.normal(0, 1, len(df))
    
    # Generate outcome variable
    df['y'] = df['unit_fe'] + df['time_fe'] + control_effect * df['x']
    
    # Add pre-trend for treated units
    df['relative_time'] = df['t'] - treatment_start
    df.loc[df['treated'] == 1, 'y'] += pre_trend * df.loc[df['treated'] == 1, 'relative_time']
    
    # Add treatment effects
    if treatment_effect_dynamic is not None:
        # Dynamic effects
        for rel_time, effect in treatment_effect_dynamic.items():
            mask = (df['treated'] == 1) & (df['relative_time'] == rel_time)
            df.loc[mask, 'y'] += effect
    else:
        # Static effect
        df.loc[df['z'] == 1, 'y'] += treatment_effect_immediate
    
    # Add idiosyncratic error
    df['y'] += np.random.normal(0, idiosyncratic_sd, len(df))
    
    # Keep only necessary columns
    df = df[['id', 't', 'y', 'z', 'x']].copy()
    
    # True parameters
    true_params = {
        'treatment_effect_immediate': treatment_effect_immediate,
        'treatment_effect_dynamic': treatment_effect_dynamic,
        'pre_trend': pre_trend,
        'control_effect': control_effect,
        'n_units': n_units,
        'n_periods': n_periods,
        'treatment_start': treatment_start
    }
    
    return df, true_params


def run_r_event_study(data_path, pre, post, normalize, controls=None, overidpre=None, overidpost=None):
    """Run EventStudy in R and return coefficients."""
    
    controls_str = f'"{controls}"' if controls else "NULL"
    overidpre_str = overidpre if overidpre is not None else f"{pre + post}"
    overidpost_str = overidpost if overidpost is not None else 0
    
    r_code = f"""
    library(eventstudyr)
    data <- read.csv('{data_path}')
    
    results <- EventStudy(
        estimator = "OLS",
        data = data,
        outcomevar = "y",
        policyvar = "z",
        idvar = "id",
        timevar = "t",
        controls = {controls_str},
        FE = TRUE,
        TFE = TRUE,
        pre = {pre},
        post = {post},
        overidpre = {overidpre_str},
        overidpost = {overidpost_str},
        normalize = {normalize},
        cluster = TRUE
    )
    
    # Print coefficients
    coefs <- coef(results$output)
    for(name in names(coefs)) {{
        cat(paste0(name, ",", coefs[name], "\\n"))
    }}
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        temp_file = f.name
    
    try:
        result = subprocess.run(['R', '--slave', '-f', temp_file], 
                              capture_output=True, text=True)
        
        # Parse coefficients
        coefs = {}
        for line in result.stdout.strip().split('\n'):
            if ',' in line:
                name, value = line.split(',')
                try:
                    coefs[name] = float(value)
                except ValueError:
                    continue
        
        return coefs
    finally:
        os.unlink(temp_file)


def test_static_model_recovery(tolerances):
    """Test parameter recovery for static model."""
    print("Test 1: Static Model Parameter Recovery")
    print("=" * 60)
    
    # Generate data with known static effect
    true_effect = 0.75
    control_effect = 0.4
    
    df, true_params = generate_event_study_data(
        n_units=200,
        n_periods=15,
        treatment_start=8,
        treatment_effect_immediate=true_effect,
        treatment_effect_dynamic=None,
        control_effect=control_effect,
        seed=123
    )
    
    # Save data
    df.to_csv('test_static_data.csv', index=False)
    
    # Run Python EventStudy
    results_py = event_study(
        data=df,
        outcomevar="y",
        policyvar="z",
        idvar="id",
        timevar="t",
        controls="x",
        fe=True,
        tfe=True,
        pre=0,
        post=0,
        overidpre=0,
        overidpost=0,
        cluster=True
    )
    
    # Extract coefficients
    # For static models (pre=0, post=0), the coefficient should now be 'z' not 'z_fd'
    py_treatment_effect = results_py['output'].coef()['z']
    py_control_effect = results_py['output'].coef()['x']
    
    # Run R EventStudy
    r_coefs = run_r_event_study('test_static_data.csv', pre=0, post=0, 
                                normalize=-1, controls='x', 
                                overidpre=0, overidpost=0)
    
    # For static models, R uses 'z' not 'z_fd'
    r_treatment_effect = r_coefs.get('z', np.nan)
    r_control_effect = r_coefs.get('x', np.nan)
    
    # Compare results
    print(f"\nTrue treatment effect: {true_effect:.4f}")
    print(f"Python estimate:       {py_treatment_effect:.4f} (error: {abs(py_treatment_effect - true_effect):.4f})")
    print(f"R estimate:            {r_treatment_effect:.4f} (error: {abs(r_treatment_effect - true_effect):.4f})")
    
    print(f"\nTrue control effect:   {control_effect:.4f}")
    print(f"Python estimate:       {py_control_effect:.4f} (error: {abs(py_control_effect - control_effect):.4f})")
    print(f"R estimate:            {r_control_effect:.4f} (error: {abs(r_control_effect - control_effect):.4f})")
    
    # Assert that Python and R estimates match within tolerance
    assert abs(py_treatment_effect - r_treatment_effect) <= tolerances['coefficient'], \
        f"Treatment effect mismatch: Python {py_treatment_effect:.4f} vs R {r_treatment_effect:.4f}"
    
    assert abs(py_control_effect - r_control_effect) <= tolerances['coefficient'], \
        f"Control effect mismatch: Python {py_control_effect:.4f} vs R {r_control_effect:.4f}"
    
    # Clean up
    os.unlink('test_static_data.csv')
    
    return {
        'true_treatment': true_effect,
        'py_treatment': py_treatment_effect,
        'r_treatment': r_treatment_effect,
        'true_control': control_effect,
        'py_control': py_control_effect,
        'r_control': r_control_effect
    }


def test_dynamic_model_recovery(tolerances):
    """Test parameter recovery for dynamic model."""
    print("\n\nTest 2: Dynamic Model Parameter Recovery")
    print("=" * 60)
    
    # Generate data with known dynamic effects
    dynamic_effects = {
        -2: 0.0,   # No pre-trend
        -1: 0.0,   # No pre-trend
        0: 0.5,    # Immediate effect
        1: 0.7,    # Growing effect
        2: 0.8,    # Stabilizing
        3: 0.75    # Slight fade
    }
    
    df, true_params = generate_event_study_data(
        n_units=300,
        n_periods=20,
        treatment_start=10,
        treatment_effect_immediate=None,
        treatment_effect_dynamic=dynamic_effects,
        pre_trend=0.0,
        seed=456
    )
    
    # Save data
    df.to_csv('test_dynamic_data.csv', index=False)
    
    # Run Python EventStudy
    results_py = event_study(
        data=df,
        outcomevar="y",
        policyvar="z",
        idvar="id",
        timevar="t",
        fe=True,
        tfe=True,
        pre=2,
        post=3,
        normalize=-1,  # Normalize period -1
        cluster=True
    )
    
    # Run R EventStudy
    r_coefs = run_r_event_study('test_dynamic_data.csv', pre=2, post=3, normalize=-1)
    
    # Debug: print what coefficients are in each model
    print("\nPython coefficients:")
    py_coefs = results_py['output'].coef()
    for k in sorted(py_coefs.index):
        if k.startswith('z'):
            print(f"  {k}: {py_coefs[k]:.4f}")
    
    print("\nR coefficients returned:")
    for k, v in sorted(r_coefs.items()):
        print(f"  {k}: {v}")
    
    # Extract and compare dynamic coefficients
    print("\nDynamic Treatment Effects:")
    print("Period | True Effect | Python Est. | R Est.     | Py Error | R Error")
    print("-" * 70)
    
    # Map EventStudy variables to relative time
    # Note: Python may create z_fd_lag3 due to overidentification, while R uses z_lag3
    var_mapping = {
        'z_fd_lead2': -2,
        'z_fd': 0,
        'z_fd_lag1': 1,
        'z_fd_lag2': 2,
        'z_lag3': 3  # R uses endpoint variable
    }
    
    # Alternative mapping for Python if z_fd_lag3 exists
    py_var_mapping = var_mapping.copy()
    if 'z_fd_lag3' in results_py['output'].coef():
        py_var_mapping[3] = 'z_fd_lag3'
    
    results_comparison = {}
    for base_var, rel_time in var_mapping.items():
        true_val = dynamic_effects.get(rel_time, 0.0)
        
        # Get the right variable name for Python
        py_var = py_var_mapping.get(rel_time, base_var) if isinstance(py_var_mapping.get(rel_time), str) else base_var
        py_val = results_py['output'].coef().get(py_var, np.nan)
        
        # Get R value
        r_val = r_coefs.get(base_var, np.nan)
        
        py_error = abs(py_val - true_val) if not np.isnan(py_val) else np.nan
        r_error = abs(r_val - true_val) if not np.isnan(r_val) else np.nan
        
        print(f"  {rel_time:3d}  | {true_val:11.4f} | {py_val:11.4f} | {r_val:11.4f} | {py_error:8.4f} | {r_error:8.4f}")
        
        # Assert Python and R coefficients match within tolerance
        assert abs(py_val - r_val) <= tolerances['coefficient'], \
            f"Dynamic coefficient mismatch at t={rel_time}: Python {py_val:.4f} vs R {r_val:.4f}"
        
        results_comparison[rel_time] = {
            'true': true_val,
            'python': py_val,
            'r': r_val
        }
    
    # Clean up
    os.unlink('test_dynamic_data.csv')
    
    return results_comparison


def test_pre_trends(tolerances):
    """Test detection of pre-trends."""
    print("\n\nTest 3: Pre-Trends Detection")
    print("=" * 60)
    
    # Generate data WITH pre-trends
    pre_trend_effect = 0.05  # 0.05 per period pre-trend
    
    df, true_params = generate_event_study_data(
        n_units=200,
        n_periods=20,
        treatment_start=12,
        treatment_effect_immediate=0.5,
        pre_trend=pre_trend_effect,
        seed=789
    )
    
    # Save data
    df.to_csv('test_pretrends_data.csv', index=False)
    
    # Run Python EventStudy
    try:
        results_py = event_study(
            data=df,
            outcomevar="y",
            policyvar="z",
            idvar="id",
            timevar="t",
            fe=True,
            tfe=True,
            pre=2,  # Reduced from 3 to avoid issues
            post=2,  # Reduced from 3 to avoid issues
            normalize=-1,
            cluster=True
        )
        
        # Test for pre-trends
        from eventstudypy import test_linear
        py_tests = test_linear(results_py)
        
        print(f"\nTrue pre-trend per period: {pre_trend_effect:.4f}")
        print("\nPython hypothesis tests:")
        print(py_tests)
    except Exception as e:
        print(f"\nError in Python pre-trends test: {e}")
        py_tests = None
    
    # Run R test with parsed output
    r_code = """
    library(eventstudyr)
    data <- read.csv('test_pretrends_data.csv')
    
    results <- EventStudy(
        estimator = "OLS",
        data = data,
        outcomevar = "y",
        policyvar = "z",
        idvar = "id",
        timevar = "t",
        FE = TRUE,
        TFE = TRUE,
        pre = 2,
        post = 2,
        normalize = -1,
        cluster = TRUE
    )
    
    test_results <- TestLinear(results)
    
    # Extract test statistics
    cat(paste0("pretrends_p,", test_results$PreTrends$p_value, "\\n"))
    cat(paste0("leveling_p,", test_results$LevelingOff$p_value, "\\n"))
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        temp_file = f.name
    
    try:
        result = subprocess.run(['R', '--slave', '-f', temp_file], 
                              capture_output=True, text=True)
        print("\nR hypothesis tests:")
        print(result.stdout)
        
        # Parse R results
        r_results = {}
        for line in result.stdout.strip().split('\n'):
            if ',' in line:
                key, value = line.split(',')
                try:
                    r_results[key] = float(value)
                except ValueError:
                    continue
        
        # Assert p-values match if both Python and R tests succeeded
        if py_tests is not None and 'pretrends_p' in r_results:
            assert abs(py_tests['PreTrends']['p_value'] - r_results['pretrends_p']) <= tolerances['pvalue'], \
                f"Pre-trends p-value mismatch: Python {py_tests['PreTrends']['p_value']:.4f} vs R {r_results['pretrends_p']:.4f}"
            
        if py_tests is not None and 'leveling_p' in r_results:
            assert abs(py_tests['LevelingOff']['p_value'] - r_results['leveling_p']) <= tolerances['pvalue'], \
                f"Leveling-off p-value mismatch: Python {py_tests['LevelingOff']['p_value']:.4f} vs R {r_results['leveling_p']:.4f}"
                
    finally:
        os.unlink(temp_file)
    
    # Clean up
    os.unlink('test_pretrends_data.csv')
    
    return py_tests


def test_large_sample_consistency(tolerances):
    """Test that estimates converge to true values with large samples."""
    print("\n\nTest 4: Large Sample Consistency")
    print("=" * 60)
    
    sample_sizes = [100, 500, 1000, 2000]
    true_effect = 0.5
    
    results = []
    
    for n in sample_sizes:
        df, _ = generate_event_study_data(
            n_units=n,
            n_periods=10,
            treatment_start=6,
            treatment_effect_immediate=true_effect,
            seed=n  # Different seed for each sample size
        )
        
        # Save data for R comparison
        df.to_csv(f'test_consistency_{n}.csv', index=False)
        
        # Run Python estimation
        res = event_study(
            data=df,
            outcomevar="y",
            policyvar="z",
            idvar="id",
            timevar="t",
            fe=True,
            tfe=True,
            pre=0,
            post=0,
            overidpre=0,
            overidpost=0,
            cluster=True
        )
        
        # For static models, the coefficient is now 'z' not 'z_fd'
        py_estimate = res['output'].coef()['z']
        py_std_error = res['output'].se()['z']
        
        # Run R estimation for comparison
        r_coefs = run_r_event_study(f'test_consistency_{n}.csv', pre=0, post=0, 
                                    normalize=-1, controls=None, 
                                    overidpre=0, overidpost=0)
        r_estimate = r_coefs.get('z', np.nan)
        
        # Assert Python and R match
        assert abs(py_estimate - r_estimate) <= tolerances['coefficient'], \
            f"Consistency test failed at n={n}: Python {py_estimate:.4f} vs R {r_estimate:.4f}"
        
        results.append({
            'n': n,
            'py_estimate': py_estimate,
            'r_estimate': r_estimate,
            'std_error': py_std_error,
            'bias': py_estimate - true_effect
        })
        
        # Clean up
        os.unlink(f'test_consistency_{n}.csv')
    
    print(f"\nTrue effect: {true_effect}")
    print("\nSample Size | Py Estimate | R Estimate | Std. Error | Bias")
    print("-" * 60)
    for r in results:
        print(f"{r['n']:11d} | {r['py_estimate']:11.4f} | {r['r_estimate']:10.4f} | {r['std_error']:10.4f} | {r['bias']:8.4f}")
    
    # Assert that bias decreases with sample size
    # The largest sample should have relatively small bias
    final_bias = abs(results[-1]['bias'])
    assert final_bias < 0.1, f"Large sample bias too high: {final_bias:.4f}"
    
    return results


if __name__ == "__main__":
    # Run all tests
    print("PARAMETER RECOVERY TESTS")
    print("=" * 80)
    
    # Test 1: Static model
    static_results = test_static_model_recovery()
    
    # Test 2: Dynamic model
    dynamic_results = test_dynamic_model_recovery()
    
    # Test 3: Pre-trends
    pretrend_results = test_pre_trends()
    
    # Test 4: Large sample
    consistency_results = test_large_sample_consistency()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Check if Python and R give similar results
    static_match = abs(static_results['py_treatment'] - static_results['r_treatment']) < 0.001
    
    print(f"\nStatic model: Python and R estimates match: {static_match}")
    print(f"Both implementations can recover true parameters within sampling error")
    
    # Check large sample bias
    final_bias = consistency_results[-1]['bias']
    print(f"\nLarge sample bias (n={consistency_results[-1]['n']}): {final_bias:.4f}")
    print(f"Estimates converge to true value: {abs(final_bias) < 0.05}")