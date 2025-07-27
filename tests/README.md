# EventStudyPy Test Suite

This directory contains comprehensive tests for the EventStudyPy package, porting all relevant tests from the R eventstudyr package.

## Test Structure

The test suite is organized into the following files:

### Core Functionality Tests
- **test_event_study.py** - Main EventStudy function tests (ported from test-EventStudy.R)
  - Tests shift value creation
  - Tests normalization handling
  - Tests error conditions
  - Tests all parameter combinations
  - Compares results with R implementation

- **test_basic.py** - Basic functionality tests
  - Simple event study tests
  - Tests with controls
  - Tests with fixed effects
  - Static model tests

### Component Tests
- **test_linear.py** - TestLinear hypothesis testing (ported from test-TestLinear.R)
  - Pre-trends tests
  - Leveling-off tests
  - Input validation
  - R comparison for p-values

- **test_add_cis.py** - AddCIs confidence intervals (ported from test-AddCIs.R)
  - 95% CI calculation accuracy
  - Custom confidence levels
  - Input validation
  - R comparison

- **test_compute_shifts.py** - ComputeShifts lead/lag creation (ported from test-ComputeShifts.R)
  - Shift accuracy with continuous time
  - Handling gaps in time series
  - First difference creation
  - Column naming conventions

### Advanced Tests
- **test_comprehensive_scenarios.py** - Complex scenario testing
  - Heterogeneous treatment effects
  - Staggered adoption
  - Multiple treatment periods
  - Large dataset performance

- **test_parameter_recovery.py** - Parameter recovery validation
  - Known DGP recovery
  - Bias assessment
  - Coverage rates

### Comparison Tests
- **compare_with_R.py** - Direct R package comparison
  - Runs identical specifications in Python and R
  - Compares coefficients with high precision
  - Tests multiple model specifications

## Running the Tests

### Prerequisites
```bash
# Install pytest
pip install pytest

# For R comparison tests, ensure R is installed with:
# install.packages("eventstudyr")
```

### Run All Tests
```bash
# From the tests directory
python run_all_tests.py

# Or using pytest directly
pytest -v
```

### Run Specific Test Categories
```bash
# Run only unit tests
pytest -m unit

# Run only R comparison tests
pytest -m r_comparison

# Run without slow tests
pytest -m "not slow"
```

### Run Individual Test Files
```bash
# Test specific functionality
pytest test_event_study.py -v
pytest test_linear.py -v
pytest test_add_cis.py -v
```

## Test Coverage

The test suite covers:

1. **All EventStudy specifications from R package**
   - Dynamic models with various pre/post periods
   - Static models
   - Models with/without fixed effects
   - Models with/without clustering
   - Models with/without controls
   - Anticipation effects normalization

2. **All relevant R package tests** (excluding FHS/IV/Proxy)
   - EventStudy main functionality
   - EventStudyOLS combinations
   - TestLinear hypothesis tests
   - AddCIs confidence intervals
   - ComputeShifts transformations

3. **Edge cases and error handling**
   - Invalid parameters
   - Out-of-bounds normalization
   - Missing data handling
   - Type validation

4. **Performance and accuracy**
   - Large dataset handling
   - Numerical precision
   - Memory efficiency

## R Comparison

Tests that compare with R implementation:
- Require R and eventstudyr package installed
- Use subprocess to run R code
- Compare results with high precision (typically 1e-6 to 1e-10)
- Can be skipped if R is not available

## Adding New Tests

When adding new tests:
1. Follow pytest conventions
2. Use the existing fixtures for data loading
3. Add appropriate markers (@pytest.mark.slow, @pytest.mark.r_comparison)
4. Include R comparison when applicable
5. Document expected behavior

## Troubleshooting

If tests fail:
1. Check that all dependencies are installed
2. Ensure R and eventstudyr are available for comparison tests
3. Verify the example data file exists
4. Check for version compatibility issues

For R-related failures:
- Install R from https://www.r-project.org/
- Install eventstudyr: `install.packages("eventstudyr")`
- Ensure R is in your system PATH