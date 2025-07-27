"""
Data preparation utilities for event study estimation
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Union, Optional, Tuple


def detect_holes(df: pd.DataFrame, idvar: str, timevar: str) -> bool:
    """
    Detect if there are gaps in the time variable within each unit.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data frame to check
    idvar : str
        Name of the unit identifier column
    timevar : str
        Name of the time period column
        
    Returns
    -------
    bool
        True if gaps are detected, False otherwise
    """
    def has_gaps(group):
        time_vals = group[timevar].dropna().sort_values()
        if len(time_vals) < 2:
            return False
        diffs = np.diff(time_vals)
        return any(diffs != 1)
    
    return df.groupby(idvar).apply(has_gaps).any()


def compute_first_differences(
    df: pd.DataFrame,
    idvar: str,
    timevar: str,
    policyvar: str,
    timevar_holes: bool = False
) -> pd.DataFrame:
    """
    Compute first differences of the policy variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data frame
    idvar : str
        Name of the unit identifier column
    timevar : str
        Name of the time period column
    policyvar : str
        Name of the policy variable
    timevar_holes : bool
        Whether there are gaps in the time variable
        
    Returns
    -------
    pd.DataFrame
        Data frame with first differenced policy variable added
    """
    df = df.copy()
    df = df.sort_values([idvar, timevar])
    
    # Compute first differences within each unit
    df[f"{policyvar}_fd"] = df.groupby(idvar)[policyvar].diff()
    
    # If there are holes, set differences to NaN where time gap > 1
    if timevar_holes:
        time_diff = df.groupby(idvar)[timevar].diff()
        df.loc[time_diff != 1, f"{policyvar}_fd"] = np.nan
    
    return df


def _compute_shifts_internal(
    df: pd.DataFrame,
    idvar: str,
    timevar: str,
    shiftvar: str,
    shiftvalues: List[int],
    timevar_holes: bool = False
) -> pd.DataFrame:
    """
    Compute leads and lags of a variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data frame
    idvar : str
        Name of the unit identifier column
    timevar : str
        Name of the time period column
    shiftvar : str
        Name of the variable to shift
    shiftvalues : List[int]
        List of shift values (negative for leads, positive for lags)
    timevar_holes : bool
        Whether there are gaps in the time variable
        
    Returns
    -------
    pd.DataFrame
        Data frame with shifted variables added
    """
    df = df.copy()
    df = df.sort_values([idvar, timevar])
    
    for shift in shiftvalues:
        if shift < 0:
            # Lead
            col_name = f"{shiftvar}_lead{abs(shift)}"
            df[col_name] = df.groupby(idvar)[shiftvar].shift(shift)
            
            # If there are holes, check time consistency
            if timevar_holes:
                time_shifted = df.groupby(idvar)[timevar].shift(shift)
                expected_time = df[timevar] + shift
                df.loc[time_shifted != expected_time, col_name] = np.nan
                
        elif shift > 0:
            # Lag
            col_name = f"{shiftvar}_lag{shift}"
            df[col_name] = df.groupby(idvar)[shiftvar].shift(shift)
            
            # If there are holes, check time consistency
            if timevar_holes:
                time_shifted = df.groupby(idvar)[timevar].shift(shift)
                expected_time = df[timevar] - shift
                df.loc[time_shifted != expected_time, col_name] = np.nan
    
    return df


def prepare_event_study_data(
    df: pd.DataFrame,
    idvar: str,
    timevar: str,
    policyvar: str,
    post: int,
    overidpost: int,
    pre: int,
    overidpre: int,
    static: bool = False
) -> Tuple[pd.DataFrame, List[str], bool]:
    """
    Prepare data for event study estimation by creating necessary leads and lags.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data frame
    idvar : str
        Name of the unit identifier column
    timevar : str
        Name of the time period column  
    policyvar : str
        Name of the policy variable
    post : int
        Number of post-treatment periods
    overidpost : int
        Number of additional post periods
    pre : int
        Number of pre-treatment periods
    overidpre : int
        Number of additional pre periods
    static : bool
        Whether this is a static model
        
    Returns
    -------
    tuple
        (prepared DataFrame, list of policy variable names, timevar_holes flag)
    """
    df = df.copy()
    
    # Check for time variable holes
    timevar_holes = detect_holes(df, idvar, timevar)
    if timevar_holes:
        warnings.warn(f"Note: gaps of more than one unit in the time variable '{timevar}' were detected. "
              f"Treating these as gaps in the panel dimension.")
    
    # Sort data
    df = df.sort_values([idvar, timevar])
    
    if static:
        # For static models, use the original policy variable, not first differences
        return df, [policyvar], timevar_holes
    
    # Compute first differences
    df = compute_first_differences(df, idvar, timevar, policyvar, timevar_holes)
    
    # Compute shifts for first differenced variable
    num_fd_lags = post + overidpost - 1
    num_fd_leads = pre + overidpre
    
    shift_values = []
    if num_fd_lags >= 1 and num_fd_leads >= 1:
        shift_values = list(range(-num_fd_leads, 0)) + list(range(1, num_fd_lags + 1))
    elif num_fd_leads < 1:
        shift_values = list(range(1, num_fd_lags + 1))
    elif num_fd_lags < 1:
        shift_values = list(range(-num_fd_leads, 0))
    
    if shift_values:
        df = _compute_shifts_internal(df, idvar, timevar, f"{policyvar}_fd", shift_values, timevar_holes)
    
    # Compute endpoint variables
    furthest_lag_period = num_fd_lags + 1
    endpoint_shifts = []
    if num_fd_leads > 0:
        endpoint_shifts.append(-num_fd_leads)
    if furthest_lag_period > 0:
        endpoint_shifts.append(furthest_lag_period)
    
    if endpoint_shifts:
        df = _compute_shifts_internal(df, idvar, timevar, policyvar, endpoint_shifts, timevar_holes)
    
    # Invert lead endpoint variable to match R's interpretation
    # z_lead{k} represents "k periods before treatment"
    # If z goes from 0 to 1 at treatment, z.shift(-k) gives us future values
    # We need to invert this to get an indicator for the pre-treatment period
    if num_fd_leads > 0:
        lead_endpoint_var = f"{policyvar}_lead{num_fd_leads}"
        df[lead_endpoint_var] = 1 - df[lead_endpoint_var]
    
    # Collect all policy variables in order
    policy_vars = []
    
    # Lead endpoint
    if num_fd_leads > 0:
        policy_vars.append(f"{policyvar}_lead{num_fd_leads}")
    
    # Lead FD variables
    for i in range(num_fd_leads, 0, -1):
        var_name = f"{policyvar}_fd_lead{i}"
        if var_name in df.columns:
            policy_vars.append(var_name)
    
    # Contemporary FD
    if f"{policyvar}_fd" in df.columns:
        policy_vars.append(f"{policyvar}_fd")
    
    # Lag FD variables
    for i in range(1, num_fd_lags + 1):
        var_name = f"{policyvar}_fd_lag{i}"
        if var_name in df.columns:
            policy_vars.append(var_name)
    
    # Lag endpoint
    if furthest_lag_period > 0 and f"{policyvar}_lag{furthest_lag_period}" in df.columns:
        policy_vars.append(f"{policyvar}_lag{furthest_lag_period}")
    
    return df, policy_vars, timevar_holes


def compute_shifts(
    data: pd.DataFrame,
    shiftvar: str,
    targetvars: List[str],
    idvar: str,
    timevar: str,
    shifts: List[int],
    create_dummies: bool = False
) -> pd.DataFrame:
    """
    Public API for computing leads and lags of variables.
    
    This function creates shifted (lead/lag) versions of variables in a panel dataset.
    It's the Python equivalent of the R package's ComputeShifts function.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input panel data
    shiftvar : str
        Name of the variable to create shifts from (for creating interaction terms)
    targetvars : List[str]
        List of variables to shift
    idvar : str
        Name of the unit identifier column
    timevar : str
        Name of the time period column
    shifts : List[int]
        List of shift values (negative for leads, positive for lags, 0 for contemporary)
    create_dummies : bool
        Whether to create dummy variables (not implemented yet)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new columns for each targetvar-shift combination
        
    Examples
    --------
    >>> df = compute_shifts(data, "treatment", ["outcome"], "id", "time", [-2, -1, 0, 1, 2])
    Creates: outcome_lead2, outcome_lead1, outcome, outcome_lag1, outcome_lag2
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
        
    if not isinstance(shiftvar, str):
        raise TypeError("shiftvar must be a string")
        
    if not isinstance(targetvars, list):
        raise TypeError("targetvars must be a list")
        
    if not isinstance(shifts, list):
        raise TypeError("shifts must be a list")
    
    # Check if columns exist
    if shiftvar not in data.columns:
        raise ValueError(f"shiftvar '{shiftvar}' not found in data")
        
    for targetvar in targetvars:
        if targetvar not in data.columns:
            raise ValueError(f"targetvar '{targetvar}' not found in data")
            
    if idvar not in data.columns:
        raise ValueError(f"idvar '{idvar}' not found in data")
        
    if timevar not in data.columns:
        raise ValueError(f"timevar '{timevar}' not found in data")
    
    result = data.copy()
    
    # Check for time holes
    timevar_holes = detect_holes(result, idvar, timevar)
    
    # Create shifted versions of each targetvar
    for targetvar in targetvars:
        # Only create shifts for non-zero values
        non_zero_shifts = [s for s in shifts if s != 0]
        if non_zero_shifts:
            result = _compute_shifts_internal(result, idvar, timevar, targetvar, non_zero_shifts, timevar_holes)
    
    # Also create first differences if requested (when 0 is in shifts)
    if 0 in shifts:
        for targetvar in targetvars:
            # Sort by id and time
            result = result.sort_values([idvar, timevar])
            # Compute first difference
            result[f"{targetvar}_fd"] = result.groupby(idvar)[targetvar].diff()
            
            # Handle time holes
            if timevar_holes:
                time_diff = result.groupby(idvar)[timevar].diff()
                result.loc[time_diff != 1, f"{targetvar}_fd"] = np.nan
                
            # If create_dummies is True, also create shifted versions of first differences
            if create_dummies:
                fd_shifts = [s for s in shifts if s != 0]
                if fd_shifts:
                    result = _compute_shifts_internal(result, idvar, timevar, f"{targetvar}_fd", fd_shifts, timevar_holes)
    
    return result