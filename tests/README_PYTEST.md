# Pytest-Standard Test Organization

This test suite follows pytest best practices while maintaining readability by keeping R comparisons within test files.

## Structure

```
tests/
├── conftest.py                 # Shared fixtures and pytest configuration
├── test_r_parity.py           # Main R comparison tests (comprehensive)
├── test_event_study.py        # EventStudy function tests
├── test_linear.py             # TestLinear hypothesis tests  
├── test_add_cis.py            # AddCIs confidence interval tests
├── test_compute_shifts.py     # ComputeShifts tests
├── test_basic.py              # Basic functionality tests
├── compare_with_R.py          # Standalone R comparison script
└── pytest.ini                 # Pytest configuration
```

## Key Design Decisions

1. **R comparisons in same file**: All R code and comparison logic is kept within the test files for easy reading and maintenance.

2. **Shared fixtures in conftest.py**:
   - `example_data` - Loads data once per session
   - `example_data_path` - Provides path to data
   - `sample_event_study_params` - Common parameters
   - `r_available` - Checks R/eventstudyr availability

3. **Tests require R**: All tests will fail (not skip) if R is not available, ensuring we always validate against R.

4. **Clear test organization**:
   - Each test class focuses on one component
   - R comparison utilities are static methods within test classes
   - Parametrized tests for combinations

## Running Tests

```bash
# Run all tests (requires R and eventstudyr)
pytest -v

# Run specific test file
pytest test_r_parity.py -v

# Run with output
pytest -v -s

# Run specific test
pytest test_r_parity.py::TestRParity::test_basic_dynamic_model -v
```

## Test Markers

- `@pytest.mark.r_required` - Test requires R (currently all tests)
- `@pytest.mark.slow` - Slow running test
- `@pytest.mark.parametrize` - Tests multiple parameter combinations

## Precision Levels

- **1e-6**: Default for most coefficient comparisons
- **1e-8**: High precision for CI and critical tests
- **1e-10**: Ultra-high precision for specific tests
- **0.001**: P-value comparisons

## Adding New Tests

1. Create test in appropriate file or new file
2. Keep R comparison code in same test method/class
3. Use fixtures from conftest.py
4. Assert exact equality with R using appropriate tolerance

Example:
```python
def test_new_feature(self, example_data, example_data_path):
    # Python implementation
    py_result = some_function(example_data)
    
    # R implementation (in same method for clarity)
    r_output = self.run_r_code(f'''
        library(eventstudyr)
        data <- read.csv('{example_data_path}')
        # R code here
    ''')
    
    # Compare results
    self.assert_results_equal(py_result, r_output, tolerance=1e-6)
```