# EventStudyPy

A Python implementation of event study estimation, providing a Python port of the R eventstudyr package.

## Overview

EventStudyPy is a Python package for conducting event study analyses, commonly used in economics and finance to study the effects of events or interventions on outcomes over time. This package is a faithful port of the R eventstudyr package, implementing the methods recommended in [Freyaldenhoven et al. (2021)](https://www.nber.org/papers/w29170).

## Features

- **Event Study Estimation**: Conduct event studies with panel data using OLS with fixed effects
- **Dynamic and Static Models**: Support for both static (single period) and dynamic (multiple period) specifications
- **Flexible Fixed Effects**: Unit and time fixed effects with additional custom fixed effects support
- **Lead and Lag Analysis**: Automatic generation of leads and lags for dynamic event study specifications
- **Normalization Options**: Flexible normalization of coefficients for identification
- **Hypothesis Testing**: Built-in tests for pre-trends and leveling-off
- **Visualization**: Clean event study plots with confidence intervals and customization options
- **Robust Standard Errors**: Cluster-robust and heteroskedasticity-robust standard errors

## Installation

### From PyPI (when available)

```bash
pip install eventstudypy
```

### From Source

```bash
git clone https://github.com/eventstudypy/eventstudypy.git
cd eventstudypy
pip install -e .
```

### Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- pyfixest >= 0.18.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- statsmodels >= 0.12.0

## Quick Start

```python
import pandas as pd
from eventstudypy import event_study, event_study_plot

# Load your data
data = pd.read_csv('your_data.csv')

# Run event study
results = event_study(
    data=data,
    outcomevar="outcome",      # Outcome variable
    policyvar="treatment",     # Treatment/policy variable
    idvar="unit_id",          # Unit identifier
    timevar="time",           # Time identifier
    pre=3,                    # Pre-treatment periods
    post=3,                   # Post-treatment periods
    normalize=-1,             # Normalize period -1 to zero
    cluster=True              # Cluster standard errors at unit level
)

# Create event study plot
event_study_plot(results, ylabel="Effect on Outcome")
```

## Detailed Usage

### Static Model

For a simple difference-in-differences style estimation:

```python
results = event_study(
    data=data,
    outcomevar="y",
    policyvar="treated",
    idvar="id",
    timevar="period",
    pre=0,
    post=0,
    overidpre=0,
    overidpost=0
)
```

### Dynamic Model with Controls

For a dynamic specification with control variables:

```python
results = event_study(
    data=data,
    outcomevar="y",
    policyvar="policy",
    idvar="firm_id",
    timevar="year",
    controls=["gdp", "unemployment"],  # Control variables
    fe=True,                          # Unit fixed effects
    tfe=True,                         # Time fixed effects
    pre=2,                           # 2 pre-treatment periods
    post=3,                          # 3 post-treatment periods
    normalize=-1,                    # Normalize period -1
    cluster=True                     # Cluster standard errors
)
```

### Additional Fixed Effects

You can include additional fixed effects beyond unit and time:

```python
results = event_study(
    data=data,
    outcomevar="y",
    policyvar="treatment",
    idvar="firm_id",
    timevar="year",
    fe=True,
    tfe=True,
    fixed_effects=["industry", "region"],  # Additional fixed effects
    pre=2,
    post=3,
    normalize=-1,
    cluster=True
)
```

### Hypothesis Testing

Test for pre-trends and leveling-off:

```python
from eventstudypy import test_linear

# Returns DataFrame with test statistics and p-values
test_results = test_linear(results)
print(test_results)
```

### Custom Plotting

```python
from eventstudypy import event_study_plot

# Customize the plot
event_study_plot(
    results,
    xlabel="Time to Treatment",
    ylabel="Treatment Effect",
    title="Event Study Results",
    figsize=(10, 6),
    style='seaborn',
    supt_conf_level=0.95,    # Add sup-t confidence bands
    add_mean=True            # Add mean effect line
)
```

## Examples

See the `examples/` directory for complete examples:

- `example.py`: Basic usage example

## Testing

The package includes comprehensive tests comparing results with the original R implementation:

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_basic.py
pytest tests/test_r_parity.py

# Run with coverage
pytest tests/ --cov=eventstudypy
```

## Validation

This implementation has been thoroughly tested against the R package:
- ✅ All coefficients match exactly between Python and R
- ✅ Both implementations recover true parameters within sampling error
- ✅ Static and dynamic models work correctly
- ✅ Handles heterogeneous effects, staggered treatment, anticipation effects, and long-run dynamics

## Differences from R Package

This Python implementation closely follows the R eventstudyr package with a few differences:

1. **API**: Function names use Python conventions (snake_case instead of camelCase)
2. **Data Structure**: Uses pandas DataFrames instead of R data.frames
3. **Plotting**: Uses matplotlib/seaborn instead of ggplot2
4. **FHS Estimator**: The proxy/IV functionality (FHS estimator) is not included in this version

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{eventstudypy,
  title = {EventStudyPy: Event Study Analysis in Python},
  author = {EventStudyPy Contributors},
  year = {2024},
  url = {https://github.com/eventstudypy/eventstudypy}
}
```

## Acknowledgments

This package is a Python port of the R eventstudyr package. We thank the original authors for their work.

## References

Freyaldenhoven, S., Hansen, C., Pérez, J. P., & Shapiro, J. M. (2021). Visualization, Identification, and Estimation in the Linear Panel Event-Study Design (No. w29170). National Bureau of Economic Research.