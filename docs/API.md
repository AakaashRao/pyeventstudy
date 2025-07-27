# EventStudyPy API Reference

## Main Functions

### event_study()

The main function for conducting event study analysis.

```python
event_study(
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
) -> Dict
```

**Parameters:**
- `data`: Panel data DataFrame
- `outcomevar`: Name of outcome variable column
- `policyvar`: Name of policy/treatment variable column
- `idvar`: Name of unit identifier column
- `timevar`: Name of time period column
- `post`: Number of post-treatment periods to include
- `pre`: Number of pre-treatment periods to include
- `estimator`: Estimation method (only "OLS" supported)
- `controls`: Control variables (string or list of strings)
- `proxy`: Not implemented (must be None)
- `proxyIV`: Not implemented (must be None)
- `fe`: Include unit fixed effects (default: True)
- `tfe`: Include time fixed effects (default: True)
- `fixed_effects`: Additional fixed effects to absorb (string or list)
- `overidpost`: Additional post periods (default: 1)
- `overidpre`: Additional pre periods (defaults to post + pre)
- `normalize`: Event time to normalize to zero (default: -pre-1)
- `cluster`: Cluster standard errors by unit (default: True)
- `anticipation_effects_normalization`: Adjust normalization for anticipation effects (default: True)

**Returns:**
Dictionary with keys:
- `output`: Regression results from pyfixest
- `arguments`: Dictionary of arguments used

### event_study_plot()

Create publication-ready event study plots.

```python
event_study_plot(
    estimates: Union[pd.DataFrame, Dict],
    xlabel: str = "Time to Treatment",
    ylabel: str = "Estimate",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    style: str = 'seaborn',
    ci_level: float = 0.95,
    add_mean: bool = False,
    supt_conf_level: Optional[float] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]
```

**Parameters:**
- `estimates`: Event study results or DataFrame with estimates
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `title`: Plot title
- `figsize`: Figure size as (width, height)
- `style`: Matplotlib style
- `ci_level`: Confidence level for intervals
- `add_mean`: Add horizontal line at mean effect
- `supt_conf_level`: Add sup-t confidence bands
- `save_path`: Path to save figure
- `dpi`: Resolution for saved figure

**Returns:**
Tuple of (figure, axes) objects

### test_linear()

Test for pre-trends and leveling-off in event study results.

```python
test_linear(
    estimate: Dict,
    test_pretrends: bool = True,
    test_leveling_off: bool = True
) -> pd.DataFrame
```

**Parameters:**
- `estimate`: Event study results dictionary
- `test_pretrends`: Test for pre-treatment trends
- `test_leveling_off`: Test for leveling-off of effects

**Returns:**
DataFrame with columns:
- `test`: Test name ("pretrends" or "leveling_off")
- `statistic`: F-statistic value
- `p_value`: P-value of the test

### add_cis()

Add confidence intervals to event study results.

```python
add_cis(
    estimate: Union[Dict, pd.DataFrame],
    conf_level: float = 0.95
) -> pd.DataFrame
```

**Parameters:**
- `estimate`: Event study results or coefficients DataFrame
- `conf_level`: Confidence level (default: 0.95)

**Returns:**
DataFrame with columns:
- `term`: Coefficient name
- `estimate`: Point estimate
- `std.error`: Standard error
- `conf.low`: Lower confidence bound
- `conf.high`: Upper confidence bound

### compute_shifts()

Generate leads and lags of variables.

```python
compute_shifts(
    data: pd.DataFrame,
    idvar: str,
    timevar: str,
    shiftvar: Union[str, List[str]],
    shiftvalues: Union[int, List[int]],
    donotcreatedummyshift: bool = False
) -> pd.DataFrame
```

**Parameters:**
- `data`: Input DataFrame
- `idvar`: Unit identifier column
- `timevar`: Time period column
- `shiftvar`: Variable(s) to shift
- `shiftvalues`: Number of periods to shift (negative for leads, positive for lags)
- `donotcreatedummyshift`: Skip creating dummy shifts

**Returns:**
DataFrame with additional columns for shifted variables

## Example Usage

```python
import pandas as pd
from eventstudypy import event_study, event_study_plot, test_linear

# Load data
data = pd.read_csv('panel_data.csv')

# Run event study
results = event_study(
    data=data,
    outcomevar='outcome',
    policyvar='treatment',
    idvar='unit_id',
    timevar='period',
    pre=3,
    post=3,
    fe=True,
    tfe=True,
    cluster=True
)

# Test for pre-trends
tests = test_linear(results)
print(f"Pre-trends p-value: {tests.loc[tests['test']=='pretrends', 'p_value'].values[0]:.4f}")

# Create plot
fig, ax = event_study_plot(results, ylabel='Treatment Effect')
```