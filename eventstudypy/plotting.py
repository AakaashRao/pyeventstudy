"""
Plotting functions for event study visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple, Union
import warnings
from scipy import stats

from .testing import linear_hypothesis_test as test_linear


def add_confidence_intervals(
    input_data: Union[pd.DataFrame, Dict],
    conf_level: float = 0.95
) -> pd.DataFrame:
    """
    Add confidence intervals to coefficient dataframe.
    
    Parameters
    ----------
    input_data : pd.DataFrame or Dict
        Either a DataFrame with coefficient estimates and standard errors,
        or the output dictionary from event_study()
    conf_level : float
        Confidence level (default 0.95)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with CI columns added
    """
    # Validate conf_level
    if conf_level <= 0 or conf_level >= 1:
        raise ValueError("conf_level must be between 0 and 1")
    
    # Validate input type
    if not isinstance(input_data, (pd.DataFrame, dict)):
        raise TypeError("estimate must be a DataFrame or dictionary from event_study()")
    
    # Handle event study output dictionary
    if isinstance(input_data, dict):
        if 'output' not in input_data:
            raise ValueError("estimate dictionary must contain 'output' key from event_study()")
            
        # Check if it's an event study result or a mock DataFrame
        if isinstance(input_data['output'], pd.DataFrame):
            # Mock DataFrame passed for testing
            coef_df = input_data['output'].copy()
            if 'std.error' not in coef_df.columns:
                raise ValueError("DataFrame must contain 'std.error' column")
        else:
            # Real event study result
            model = input_data['output']
            args = input_data['arguments']
            
            # Get coefficient names for event study
            coef_names = args['eventstudy_coefficients']
            
            # Extract coefficients into a DataFrame
            coef_df = pd.DataFrame({
                'term': coef_names,
                'estimate': [model.coef()[name] for name in coef_names],
                'std.error': [model.se()[name] for name in coef_names]
            }, index=coef_names)
            
            # Add normalized coefficient if present
            if args['normalization_column']:
                norm_row = pd.DataFrame({
                    'term': [args['normalization_column']],
                    'estimate': [0.0],
                    'std.error': [0.0]
                }, index=[args['normalization_column']])
                coef_df = pd.concat([coef_df, norm_row])
    else:
        # Assume it's already a DataFrame
        coef_df = input_data.copy()
        
        # Validate required columns
        if 'estimate' not in coef_df.columns:
            raise ValueError("DataFrame must contain 'estimate' column")
        if 'std.error' not in coef_df.columns:
            raise ValueError("DataFrame must contain 'std.error' column")
    
    # Use exact values that match R package behavior
    if conf_level == 0.95:
        critical_value = 1.96
    elif conf_level == 0.90:
        critical_value = 1.645
    elif conf_level == 0.99:
        critical_value = 2.576
    else:
        alpha = 1 - conf_level
        critical_value = stats.norm.ppf(1 - alpha/2)
    
    # Use consistent column names with R package
    coef_df['conf.low'] = coef_df['estimate'] - critical_value * coef_df['std.error']
    coef_df['conf.high'] = coef_df['estimate'] + critical_value * coef_df['std.error']
    
    return coef_df


def add_supt_bands(
    model_output,
    coef_names: List[str],
    num_sim: int = 1000,
    conf_level: float = 0.95
) -> pd.DataFrame:
    """
    Add sup-t confidence bands using simulation.
    
    Parameters
    ----------
    model_output : pyfixest model
        Regression results
    coef_names : List[str]
        Names of event study coefficients
    num_sim : int
        Number of simulations
    conf_level : float
        Confidence level
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sup-t bands added
    """
    # Get coefficient estimates and covariance matrix
    coef_df = model_output.coef().to_frame('estimate')
    coef_df['std_error'] = model_output.se().values
    
    # Filter to event study coefficients
    coef_df = coef_df.loc[coef_df.index.isin(coef_names)]
    
    # Get subset of covariance matrix
    coef_indices = [i for i, name in enumerate(model_output.coef().index) if name in coef_names]
    # Get vcov - returns a new model object with the requested vcov
    model_with_vcov = model_output.vcov('HC1')
    vcov_full = model_with_vcov._vcov
    vcov_subset = vcov_full[np.ix_(coef_indices, coef_indices)]
    
    # Simulate from multivariate normal
    n_coef = len(coef_names)
    np.random.seed(None)  # Use current random state
    simulated = np.random.multivariate_normal(
        mean=np.zeros(n_coef),
        cov=vcov_subset,
        size=num_sim
    )
    
    # Compute t-statistics
    se_vector = coef_df['std_error'].values
    t_stats = simulated / se_vector
    
    # Get sup-t critical value
    max_abs_t = np.max(np.abs(t_stats), axis=1)
    critical_value = np.percentile(max_abs_t, conf_level * 100)
    
    # Add sup-t bands
    coef_df['suptband_lower'] = coef_df['estimate'] - critical_value * coef_df['std_error']
    coef_df['suptband_upper'] = coef_df['estimate'] + critical_value * coef_df['std_error']
    
    return coef_df


def prepare_plotting_data(
    results: Dict,
    conf_level: Optional[float] = 0.95,
    supt: Optional[float] = 0.95,
    num_sim: int = 1000
) -> pd.DataFrame:
    """
    Prepare data for event study plot.
    
    Parameters
    ----------
    results : Dict
        Output from event_study function
    conf_level : float, optional
        Confidence level for standard CIs
    supt : float, optional
        Confidence level for sup-t bands
    num_sim : int
        Number of simulations for sup-t bands
        
    Returns
    -------
    pd.DataFrame
        DataFrame ready for plotting
    """
    model = results['output']
    args = results['arguments']
    
    # Get coefficient names
    coef_names = args['eventstudy_coefficients']
    
    # Extract coefficients
    plot_df = model.coef()[coef_names].to_frame('estimate')
    plot_df['std_error'] = model.se()[coef_names].values
    plot_df['term'] = plot_df.index
    
    # Add normalized coefficient
    if args['normalization_column']:
        norm_row = pd.DataFrame({
            'estimate': [0.0],
            'std_error': [0.0],
            'term': [args['normalization_column']]
        }, index=[args['normalization_column']])
        plot_df = pd.concat([plot_df, norm_row])
    
    # Add confidence intervals
    if conf_level is not None:
        # Create a temporary DataFrame with the expected column names
        temp_df = plot_df.copy()
        temp_df = temp_df.rename(columns={'std_error': 'std.error'})
        result = add_confidence_intervals(temp_df, conf_level)
        plot_df['ci_lower'] = result['conf.low']
        plot_df['ci_upper'] = result['conf.high']
    
    # Add sup-t bands
    if supt is not None:
        supt_df = add_supt_bands(model, coef_names, num_sim, supt)
        plot_df.loc[plot_df['term'].isin(coef_names), 'suptband_lower'] = supt_df['suptband_lower'].values
        plot_df.loc[plot_df['term'].isin(coef_names), 'suptband_upper'] = supt_df['suptband_upper'].values
        # Set sup-t bands to 0 for normalized coefficient
        if args['normalization_column']:
            plot_df.loc[plot_df['term'] == args['normalization_column'], ['suptband_lower', 'suptband_upper']] = 0
    
    # Create event time labels
    plot_df['event_time'] = plot_df['term'].apply(lambda x: _extract_event_time(x, args['policyvar']))
    plot_df = plot_df.sort_values('event_time')
    
    return plot_df


def _extract_event_time(term: str, policyvar: str) -> int:
    """Extract event time from variable name."""
    if f"{policyvar}_lead" in term:
        # Extract number after 'lead'
        num = int(term.split('lead')[-1])
        return -num - 1 if f"{policyvar}_lead" in term and '_fd_' not in term else -num
    elif f"{policyvar}_lag" in term:
        # Extract number after 'lag'
        num = int(term.split('lag')[-1])
        return num if '_fd_' not in term else num
    elif term == f"{policyvar}_fd":
        return 0
    else:
        return 0


def event_study_plot(
    estimates: Dict,
    xtitle: str = "Event time",
    ytitle: str = "Coefficient",
    ybreaks: Optional[List[float]] = None,
    conf_level: Optional[float] = 0.95,
    supt: Optional[float] = 0.95,
    num_sim: int = 1000,
    add_mean: bool = False,
    pre_event_coeffs: bool = True,
    post_event_coeffs: bool = True,
    add_zero_line: bool = True,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Create an event study plot.
    
    Parameters
    ----------
    estimates : Dict
        Output from event_study function
    xtitle : str
        X-axis title
    ytitle : str
        Y-axis title
    ybreaks : List[float], optional
        Custom y-axis breaks
    conf_level : float, optional
        Confidence level for standard intervals
    supt : float, optional
        Confidence level for sup-t bands
    num_sim : int
        Number of simulations for sup-t bands
    add_mean : bool
        Add mean of dependent variable
    pre_event_coeffs : bool
        Test for pre-trends
    post_event_coeffs : bool
        Test for leveling-off
    add_zero_line : bool
        Add horizontal line at y=0
    figsize : tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if static model
    if estimates['arguments']['static']:
        raise ValueError("EventStudyPlot does not support static models")
    
    # Prepare data
    plot_df = prepare_plotting_data(estimates, conf_level, supt, num_sim)
    
    # Run tests if requested
    caption_parts = []
    if pre_event_coeffs or post_event_coeffs:
        test_results = test_linear(
            estimates,
            pretrends=pre_event_coeffs,
            leveling_off=post_event_coeffs
        )
        
        if pre_event_coeffs and 'Pre-Trends' in test_results['Test'].values:
            p_val = test_results[test_results['Test'] == 'Pre-Trends']['p.value'].iloc[0]
            caption_parts.append(f"Pre-trends p-value = {p_val:.2f}")
        
        if post_event_coeffs and 'Leveling-Off' in test_results['Test'].values:
            p_val = test_results[test_results['Test'] == 'Leveling-Off']['p.value'].iloc[0]
            caption_parts.append(f"Leveling-off p-value = {p_val:.2f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add zero line
    if add_zero_line:
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    
    # Plot points
    ax.scatter(plot_df['event_time'], plot_df['estimate'], 
              color='#006600', s=50, zorder=5)
    
    # Add confidence intervals
    if conf_level is not None:
        non_zero = plot_df['estimate'] != 0
        ax.errorbar(plot_df.loc[non_zero, 'event_time'], 
                   plot_df.loc[non_zero, 'estimate'],
                   yerr=[plot_df.loc[non_zero, 'estimate'] - plot_df.loc[non_zero, 'ci_lower'],
                         plot_df.loc[non_zero, 'ci_upper'] - plot_df.loc[non_zero, 'estimate']],
                   fmt='none', color='black', capsize=5, zorder=3)
    
    # Add sup-t bands
    if supt is not None:
        non_zero = plot_df['estimate'] != 0
        for idx in plot_df.loc[non_zero].index:
            ax.plot([plot_df.loc[idx, 'event_time']] * 2,
                   [plot_df.loc[idx, 'suptband_lower'], plot_df.loc[idx, 'suptband_upper']],
                   color='blue', linewidth=3, alpha=0.5, zorder=1)
    
    # Set labels
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    
    # Set y-axis breaks
    if ybreaks is not None:
        ax.set_yticks(ybreaks)
        ax.set_ylim(min(ybreaks), max(ybreaks))
    
    # Add caption
    if caption_parts:
        fig.text(0.1, 0.02, " — ".join(caption_parts), fontsize=10)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Set x-axis to show all event times
    event_times = sorted(plot_df['event_time'].unique())
    ax.set_xticks(event_times)
    
    plt.tight_layout()
    
    return fig