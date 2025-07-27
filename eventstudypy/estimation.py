"""
Estimation functions for event study models using pyfixest
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
import pyfixest as pf


def prepare_model_formula(
    outcomevar: str,
    policy_vars: List[str],
    controls: Optional[List[str]] = None,
    fe: bool = True,
    tfe: bool = True,
    idvar: Optional[str] = None,
    timevar: Optional[str] = None,
    fixed_effects: Optional[List[str]] = None
) -> str:
    """
    Prepare regression formula for pyfixest.
    
    Parameters
    ----------
    outcomevar : str
        Name of outcome variable
    policy_vars : List[str]
        List of policy variables (leads, lags, etc.)
    controls : Optional[List[str]]
        List of control variables
    fe : bool
        Whether to include unit fixed effects
    tfe : bool
        Whether to include time fixed effects
    idvar : Optional[str]
        Name of unit identifier (required if fe=True)
    timevar : Optional[str]
        Name of time identifier (required if tfe=True)
    fixed_effects : Optional[List[str]]
        Additional fixed effects variables to absorb
        
    Returns
    -------
    str
        Formula string for pyfixest
    """
    # Start with outcome variable
    formula_parts = [outcomevar, "~"]
    
    # Add policy variables
    formula_parts.append(" + ".join(policy_vars))
    
    # Add controls
    if controls:
        formula_parts.append(" + " + " + ".join(controls))
    
    # Add fixed effects
    fe_parts = []
    if fe and idvar:
        fe_parts.append(idvar)
    if tfe and timevar:
        fe_parts.append(timevar)
    if fixed_effects:
        fe_parts.extend(fixed_effects)
    
    if fe_parts:
        formula_parts.append(" | " + " + ".join(fe_parts))
    
    return "".join(formula_parts)


def event_study_ols(
    data: pd.DataFrame,
    formula: str,
    idvar: str,
    cluster: bool = True,
    drop_intercept: bool = False
):
    """
    Run OLS regression with fixed effects using pyfixest.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data for estimation
    formula : str
        Regression formula
    idvar : str
        Name of unit identifier for clustering
    cluster : bool
        Whether to cluster standard errors by unit
    drop_intercept : bool
        Whether to drop the intercept term
        
    Returns
    -------
    pf.Fixest
        Estimation results from pyfixest
    """
    # Set up clustering
    if cluster:
        vcov = {"CRV1": idvar}  # Cluster-robust variance
    else:
        vcov = "hetero"  # Heteroskedasticity-robust
    
    # Run regression
    model = pf.feols(
        fml=formula,
        data=data,
        vcov=vcov,
        drop_intercept=drop_intercept
    )
    
    return model


def get_normalization_column(
    normalize: int,
    pre: int,
    overidpre: int,
    post: int,
    overidpost: int,
    policyvar: str
) -> str:
    """
    Determine which coefficient should be normalized to zero.
    
    Parameters
    ----------
    normalize : int
        Event time to normalize
    pre : int
        Number of pre-treatment periods
    overidpre : int
        Number of additional pre periods
    post : int
        Number of post-treatment periods
    overidpost : int
        Number of additional post periods
    policyvar : str
        Name of policy variable
        
    Returns
    -------
    str
        Name of the variable to be excluded (normalized)
    """
    if normalize < 0:
        if normalize == -(pre + overidpre + 1):
            return f"{policyvar}_lead{-1 * (normalize + 1)}"
        else:
            return f"{policyvar}_fd_lead{-1 * normalize}"
    elif normalize == 0:
        if normalize == post + overidpost:
            return f"{policyvar}_lag{normalize}"
        else:
            return f"{policyvar}_fd"
    else:
        if normalize == post + overidpost:
            return f"{policyvar}_lag{normalize}"
        else:
            return f"{policyvar}_fd_lag{normalize}"