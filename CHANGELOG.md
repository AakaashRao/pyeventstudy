# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of EventStudyPy
- Core event study estimation functionality matching R eventstudyr package
- Support for static and dynamic event study models
- Unit and time fixed effects with additional custom fixed effects support
- Lead and lag generation for dynamic specifications
- Flexible normalization options for identification
- Hypothesis testing for pre-trends and leveling-off
- Event study plotting with confidence intervals
- Cluster-robust and heteroskedasticity-robust standard errors
- Comprehensive test suite ensuring parity with R implementation
- Support for unbalanced panels and panels with gaps
- Example dataset and usage examples

### Features
- `event_study()`: Main estimation function
- `event_study_plot()`: Create publication-ready event study plots
- `test_linear()`: Test for pre-trends and leveling-off
- `add_cis()`: Add confidence intervals to results
- `compute_shifts()`: Generate leads and lags of variables

### Notes
- This is a Python port of the R eventstudyr package
- All core functionality has been thoroughly tested against the R implementation
- Results match R output within numerical precision tolerances