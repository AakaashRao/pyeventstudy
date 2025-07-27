"""
Hypothesis testing functions for event study models
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Optional, Union, List
from scipy import stats


def linear_hypothesis_test(
    estimates: Dict,
    test: Optional[Union[str, List[str]]] = None,
    test_name: str = "User Test",
    pretrends: bool = True,
    leveling_off: bool = True
) -> pd.DataFrame:
    """
    Perform tests of linear hypotheses on event study coefficients.
    
    Parameters
    ----------
    estimates : Dict
        Output from event_study function
    test : str or List[str], optional
        Custom hypothesis test(s) to perform
    test_name : str
        Name for custom test
    pretrends : bool
        Test for pre-trends (all pre-event coefficients = 0)
    leveling_off : bool
        Test for leveling-off (post-event coefficients equal)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with test results (Test, F-statistic, p-value)
    """
    # Input validation
    if not isinstance(estimates, dict):
        raise TypeError("estimates must be a dictionary from event_study()")
    
    if 'output' not in estimates or 'arguments' not in estimates:
        raise ValueError("estimates must contain 'output' and 'arguments' keys from event_study()")
    
    if not isinstance(pretrends, bool):
        raise TypeError("pretrends must be a boolean")
    
    if not isinstance(leveling_off, bool):
        raise TypeError("leveling_off must be a boolean")
    
    model = estimates['output']
    args = estimates['arguments']
    
    # Get coefficients
    coefficients = args['eventstudy_coefficients']
    
    # Initialize results
    test_results = []
    
    # Custom test
    if test is not None:
        if isinstance(test, str):
            test = [test]
        
        # For custom tests, we need to implement hypothesis testing
        # This is a simplified version - full implementation would need
        # to parse the hypothesis string and construct the test
        warnings.warn("Custom hypothesis tests are not fully implemented yet")
    
    # Pre-trends test
    if pretrends and args['pre'] > 0:
        # Find pre-event coefficients
        pre_coeffs = []
        
        # Get lead coefficients (excluding endpoint)
        G = args['pre']
        L_G = args['overidpre']
        
        # Match R behavior: find all coefficients ending with _lead{k}
        # where k is in range(G+1, G+L_G+1)
        for k in range(G + 1, G + L_G + 1):
            suffix = f"_lead{k}"
            # Find all coefficients ending with this suffix (like R's str_sub)
            for coef_name in coefficients:
                if coef_name.endswith(suffix):
                    pre_coeffs.append(coef_name)
        
        if pre_coeffs:
            # Perform F-test for joint significance
            f_stat, p_value = _joint_f_test(model, pre_coeffs)
            
            test_results.append({
                'Test': 'Pre-Trends',
                'F': f_stat,
                'p.value': p_value
            })
    
    # Leveling-off test
    if leveling_off and args['post'] > 0 and args['overidpost'] > 0:
        # Find post-event coefficients
        M = args['post']
        L_M = args['overidpost']
        
        # Reference coefficient - find any coefficient ending with _lag{M}
        suffix_M = f"_lag{M}"
        ref_coef = None
        for coef_name in coefficients:
            if coef_name.endswith(suffix_M):
                ref_coef = coef_name
                break
        
        # Coefficients to test equality with reference
        test_coeffs = []
        for k in range(M + 1, M + L_M + 1):
            suffix_k = f"_lag{k}"
            # Find all coefficients ending with this suffix
            for coef_name in coefficients:
                if coef_name.endswith(suffix_k):
                    test_coeffs.append(coef_name)
        
        if test_coeffs and ref_coef:
            # Test equality of coefficients
            f_stat, p_value = _equality_f_test(model, ref_coef, test_coeffs)
            
            test_results.append({
                'Test': 'Leveling-Off',
                'F': f_stat,
                'p.value': p_value
            })
    
    # Convert to expected format for compatibility
    if not test_results:
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(test_results)
    
    # For backward compatibility with tests, also return dict-like access
    # This matches the expected test interface
    result_dict = {}
    
    # Extract p-values for each test type
    for _, row in df.iterrows():
        test_name = row['Test']
        p_value = row['p.value']
        
        # Handle the list of p-values (pyfixest returns list for robust SE)
        if isinstance(p_value, list):
            p_value = p_value[0]  # Take first value
            
        if test_name == 'Pre-Trends':
            result_dict['pretrends'] = p_value
        elif test_name == 'Leveling-Off':
            result_dict['leveling_off'] = p_value
    
    # Return DataFrame with dict-like attributes for backward compatibility
    class TestLinearResults:
        def __init__(self, data, result_dict=None):
            self.data = data
            self._result_dict = result_dict or {}
            # Make DataFrame methods available
            for attr in ['loc', 'iloc', 'index', 'columns', 'shape', 'values', 
                        'iterrows', 'to_dict', 'to_numpy', 'copy']:
                if hasattr(data, attr):
                    setattr(self, attr, getattr(data, attr))
            
        def __getitem__(self, key):
            if key in self._result_dict:
                return self._result_dict[key]
            return self.data[key]
            
        def get(self, key, default=None):
            if key in self._result_dict:
                return self._result_dict[key]
            return getattr(self.data, key, default)
            
        def __contains__(self, key):
            return key in self._result_dict or key in self.data
            
        def __repr__(self):
            return f"TestLinearResults:\n{self.data.__repr__()}"
            
        def __str__(self):
            return str(self.data)
    
    # Create custom results object
    results = TestLinearResults(df, result_dict=result_dict)
    
    return results


def _joint_f_test(model, coef_names: List[str]) -> tuple:
    """
    Perform F-test for joint significance of coefficients.
    
    Parameters
    ----------
    model : pyfixest model
        Regression results
    coef_names : List[str]
        Names of coefficients to test
        
    Returns
    -------
    tuple
        (F-statistic, p-value)
    """
    # Get coefficient estimates and covariance matrix
    all_coefs = model.coef()
    # Use the original model's vcov instead of forcing HC1
    all_vcov = pd.DataFrame(model._vcov, 
                           index=all_coefs.index, 
                           columns=all_coefs.index)
    
    # Find indices of coefficients to test
    coef_indices = [i for i, name in enumerate(all_coefs.index) if name in coef_names]
    
    if not coef_indices:
        return np.nan, np.nan
    
    # Extract relevant coefficients and covariance
    beta = all_coefs.iloc[coef_indices].values
    vcov = all_vcov.iloc[coef_indices, coef_indices].values
    
    # Compute F-statistic: beta' * inv(vcov) * beta / k
    k = len(coef_indices)
    
    try:
        # Use pseudo-inverse for numerical stability
        vcov_inv = np.linalg.pinv(vcov)
        quad_form = np.dot(np.dot(beta, vcov_inv), beta)
        f_stat = quad_form / k
        
        # Degrees of freedom
        df1 = k
        # Get degrees of freedom from model
        # For clustered standard errors, use number of clusters - 1 (like R)
        try:
            # Check if clustering was used - pyfixest specific logic
            if hasattr(model, '_vcov_type') and 'cluster' in str(model._vcov_type).lower():
                # For clustered SEs, use G-1 where G is number of clusters (to match R)
                if hasattr(model, '_G'):
                    # pyfixest stores number of clusters as _G
                    # _G is a list with the number of clusters for each clustering variable
                    if isinstance(model._G, list):
                        df2 = model._G[0] - 1  # Use first clustering variable
                    else:
                        df2 = model._G - 1
                elif hasattr(model, '_N_clust'):
                    df2 = model._N_clust - 1
                    if isinstance(df2, pd.Series):
                        df2 = df2.iloc[0]
                else:
                    # Try to get from cluster info
                    if hasattr(model, '_cluster_df'):
                        n_clusters = model._cluster_df.nunique().iloc[0]
                        df2 = n_clusters - 1
                    else:
                        # Fallback to residual dof
                        df2 = model._N - model._k - model._k_fe
            else:
                # Non-clustered: use residual degrees of freedom
                if hasattr(model, '_k_fe') and model._k_fe is not None:
                    df2 = model._N - model._k - model._k_fe
                else:
                    df2 = model._N - model._k
                
            # Ensure scalar
            if isinstance(df2, pd.Series):
                df2 = df2.iloc[0]
            elif hasattr(df2, '__len__') and not isinstance(df2, (int, float)):
                df2 = float(df2)
                
        except Exception as e:
            # Fallback calculation
            n_obs = model._N
            n_coefs = model._k
            if hasattr(model, '_k_fe') and model._k_fe is not None:
                n_coefs = model._k + model._k_fe
            if isinstance(n_obs, pd.Series):
                n_obs = n_obs.iloc[0]
            if isinstance(n_coefs, pd.Series):
                n_coefs = n_coefs.iloc[0] 
            df2 = n_obs - n_coefs
        
        # P-value from F-distribution
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        return f_stat, p_value
    except:
        return np.nan, np.nan


def _equality_f_test(model, ref_coef: str, test_coefs: List[str]) -> tuple:
    """
    Test equality of coefficients (H0: beta_i = beta_ref for all i).
    
    Parameters
    ----------
    model : pyfixest model
        Regression results
    ref_coef : str
        Reference coefficient name
    test_coefs : List[str]
        Coefficients to test equality with reference
        
    Returns
    -------
    tuple
        (F-statistic, p-value)
    """
    # Get all coefficients
    all_coefs = model.coef()
    # Use the original model's vcov
    all_vcov = pd.DataFrame(model._vcov,
                           index=all_coefs.index,
                           columns=all_coefs.index)
    
    # Find indices
    ref_idx = list(all_coefs.index).index(ref_coef)
    test_indices = [list(all_coefs.index).index(name) for name in test_coefs if name in all_coefs.index]
    
    if not test_indices:
        return np.nan, np.nan
    
    # Construct restriction matrix R such that R*beta = 0 under H0
    # Each row tests beta_i - beta_ref = 0
    n_restrictions = len(test_indices)
    n_coefs = len(all_coefs)
    
    R = np.zeros((n_restrictions, n_coefs))
    for i, test_idx in enumerate(test_indices):
        R[i, test_idx] = 1
        R[i, ref_idx] = -1
    
    # Compute R*beta
    r_beta = np.dot(R, all_coefs.values)
    
    # Compute R*vcov*R'
    r_vcov_r = np.dot(np.dot(R, all_vcov.values), R.T)
    
    try:
        # F-statistic
        r_vcov_r_inv = np.linalg.pinv(r_vcov_r)
        f_stat = np.dot(np.dot(r_beta, r_vcov_r_inv), r_beta) / n_restrictions
        
        # Degrees of freedom
        df1 = n_restrictions
        # Get degrees of freedom from model
        # For clustered standard errors, use number of clusters - 1 (like R)
        try:
            # Check if clustering was used - pyfixest specific logic
            if hasattr(model, '_vcov_type') and 'cluster' in str(model._vcov_type).lower():
                # For clustered SEs, use G-1 where G is number of clusters (to match R)
                if hasattr(model, '_G'):
                    # pyfixest stores number of clusters as _G
                    # _G is a list with the number of clusters for each clustering variable
                    if isinstance(model._G, list):
                        df2 = model._G[0] - 1  # Use first clustering variable
                    else:
                        df2 = model._G - 1
                elif hasattr(model, '_N_clust'):
                    df2 = model._N_clust - 1
                    if isinstance(df2, pd.Series):
                        df2 = df2.iloc[0]
                else:
                    # Try to get from cluster info
                    if hasattr(model, '_cluster_df'):
                        n_clusters = model._cluster_df.nunique().iloc[0]
                        df2 = n_clusters - 1
                    else:
                        # Fallback to residual dof
                        df2 = model._N - model._k - model._k_fe
            else:
                # Non-clustered: use residual degrees of freedom
                if hasattr(model, '_k_fe') and model._k_fe is not None:
                    df2 = model._N - model._k - model._k_fe
                else:
                    df2 = model._N - model._k
        except:
            # Fallback calculation
            n_obs = model._N
            n_coefs = model._k
            if hasattr(model, '_k_fe') and model._k_fe is not None:
                n_coefs = model._k + model._k_fe
            if isinstance(n_obs, pd.Series):
                n_obs = n_obs.iloc[0]
            if isinstance(n_coefs, pd.Series):
                n_coefs = n_coefs.iloc[0] 
            df2 = n_obs - n_coefs
        
        # P-value
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        return f_stat, p_value
    except:
        return np.nan, np.nan