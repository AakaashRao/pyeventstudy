"""
Main event study estimation function
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Optional, Dict, Union, Tuple

from .data_prep import prepare_event_study_data
from .estimation import (
    prepare_model_formula,
    event_study_ols,
    get_normalization_column
)


def event_study(
    data: pd.DataFrame,
    outcomevar: str,
    policyvar: str,
    idvar: str,
    timevar: str,
    post: int,
    pre: int,
    estimator: str = "OLS",
    controls: Optional[Union[str, List[str]]] = None,
    proxy: Optional[str] = None,
    proxyIV: Optional[str] = None,
    fe: bool = True,
    tfe: bool = True,
    fixed_effects: Optional[Union[str, List[str]]] = None,
    overidpost: int = 1,
    overidpre: Optional[int] = None,
    normalize: Optional[int] = None,
    cluster: bool = True,
    anticipation_effects_normalization: bool = True
) -> Dict:
    """
    Estimate an event study model using panel data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data containing the variables of interest
    outcomevar : str
        Name of outcome variable column
    policyvar : str
        Name of policy variable column
    idvar : str
        Name of unit identifier column
    timevar : str
        Name of time period column
    post : int
        Number of post-treatment periods to include
    pre : int
        Number of pre-treatment periods to include
    estimator : str, default "OLS"
        Estimation method (only "OLS" supported in this implementation)
    controls : str or List[str], optional
        Control variables
    proxy : str, optional
        Not implemented - will raise error if specified
    proxyIV : str, optional
        Not implemented - will raise error if specified
    fe : bool, default True
        Include unit fixed effects
    tfe : bool, default True
        Include time fixed effects
    fixed_effects : str or List[str], optional
        Additional fixed effects to absorb using pyfixest syntax.
        Can be a single variable or list of variables
    overidpost : int, default 1
        Additional post periods to include
    overidpre : int, optional
        Additional pre periods to include (defaults to post + pre)
    normalize : int, optional
        Event time coefficient to normalize to zero (defaults to -pre-1)
    cluster : bool, default True
        Cluster standard errors by unit
    anticipation_effects_normalization : bool, default True
        Adjust normalization when there are anticipation effects
        
    Returns
    -------
    dict
        Dictionary with two keys:
        - 'output': regression results
        - 'arguments': dict of arguments used
    """
    # Input validation
    if estimator != "OLS":
        raise ValueError("Only 'OLS' estimator is supported in this implementation")
    
    if proxy is not None:
        raise ValueError("proxy parameter is not supported - use proxy=None")
    
    if proxyIV is not None:
        raise ValueError("proxyIV parameter is not supported - use proxyIV=None")
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data should be a pandas DataFrame")
    
    # Handle controls as list
    if controls is None:
        controls_list = None
    elif isinstance(controls, str):
        controls_list = [controls]
    else:
        controls_list = controls
    
    # Handle fixed_effects as list
    if fixed_effects is None:
        fixed_effects_list = None
    elif isinstance(fixed_effects, str):
        fixed_effects_list = [fixed_effects]
    else:
        fixed_effects_list = fixed_effects
    
    # Set defaults
    if overidpre is None:
        overidpre = post + pre
    
    if normalize is None:
        normalize = -1 * (pre + 1)
    
    # Check for static model
    static = (post == 0 and overidpost == 0 and pre == 0 and overidpre == 0)
    
    # Validate normalize parameter
    if normalize == 0 and static:
        raise ValueError("normalize cannot be zero when post = overidpost = pre = overidpre = 0")
    
    if not (-(pre + overidpre + 1) <= normalize <= post + overidpost):
        raise ValueError(f"normalize should be between {-(pre + overidpre + 1)} and {post + overidpost}")
    
    # Check data
    if not pd.api.types.is_numeric_dtype(data[timevar]):
        raise TypeError(f"{timevar} column should be numeric")
    
    if not all(data[timevar] % 1 == 0):
        raise ValueError(f"{timevar} column should contain only integers")
    
    # Check for balanced panel
    n_units = data[idvar].nunique()
    n_periods = data[timevar].nunique()
    n_unique_rows = data[[idvar, timevar]].drop_duplicates().shape[0]
    
    if n_unique_rows != n_units * n_periods:
        warnings.warn("Dataset is unbalanced.")
    
    # Check for sufficient time periods
    num_eventstudy_coeffs = overidpre + pre + post + overidpost
    num_periods = data[timevar].max() - data[timevar].min()
    if num_eventstudy_coeffs > num_periods - 1:
        raise ValueError("overidpre + pre + post + overidpost cannot exceed the data window")
    
    # Prepare data
    prepared_data, policy_vars, timevar_holes = prepare_event_study_data(
        data, idvar, timevar, policyvar, post, overidpost, pre, overidpre, static
    )
    
    # Handle normalization with anticipation effects
    if pre != 0 and normalize == -1 and anticipation_effects_normalization:
        normalize = -pre - 1
        warnings.warn(f"You allowed for anticipation effects {pre} periods before the event, "
                     f"so the coefficient at {normalize} was selected to be normalized to zero. "
                     f"To override this, change anticipation_effects_normalization to False.")
    
    # Get normalization column
    if not static:
        normalization_column = get_normalization_column(
            normalize, pre, overidpre, post, overidpost, policyvar
        )
        # Remove normalized variable from policy vars
        policy_vars_for_regression = [v for v in policy_vars if v != normalization_column]
    else:
        normalization_column = None
        policy_vars_for_regression = policy_vars
    
    # Prepare formula
    formula = prepare_model_formula(
        outcomevar=outcomevar,
        policy_vars=policy_vars_for_regression,
        controls=controls_list,
        fe=fe,
        tfe=tfe,
        idvar=idvar if fe else None,
        timevar=timevar if tfe else None,
        fixed_effects=fixed_effects_list
    )
    
    # Run regression
    # Drop intercept when no fixed effects are included
    drop_intercept = not fe and not tfe and not fixed_effects_list
    output = event_study_ols(
        data=prepared_data,
        formula=formula,
        idvar=idvar,
        cluster=cluster,
        drop_intercept=drop_intercept
    )
    
    # Prepare arguments dictionary
    arguments = {
        "estimator": estimator,
        "data": prepared_data,
        "outcomevar": outcomevar,
        "policyvar": policyvar,
        "idvar": idvar,
        "timevar": timevar,
        "controls": controls_list,
        "fixed_effects": fixed_effects_list,
        "proxy": proxy,
        "proxyIV": proxyIV,
        "fe": fe,
        "tfe": tfe,
        "post": post,
        "overidpost": overidpost,
        "pre": pre,
        "overidpre": overidpre,
        "normalize": normalize,
        "normalization_column": normalization_column,
        "cluster": cluster,
        "eventstudy_coefficients": policy_vars_for_regression,
        "static": static
    }
    
    return {
        "output": output,
        "arguments": arguments
    }