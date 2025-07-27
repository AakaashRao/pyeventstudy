"""
Pytest configuration and shared fixtures for eventstudypy tests.
"""

import pytest
import pandas as pd
import subprocess
from pathlib import Path


@pytest.fixture(scope="session")
def example_data():
    """Load example data once per test session."""
    data_path = Path(__file__).parent.parent / 'eventstudypy' / 'example_data.csv'
    return pd.read_csv(data_path)


@pytest.fixture(scope="session")
def example_data_path():
    """Return path to example data file."""
    return Path(__file__).parent.parent / 'eventstudypy' / 'example_data.csv'


@pytest.fixture
def sample_event_study_params():
    """Standard parameters for event study tests."""
    return {
        'outcomevar': 'y_base',
        'policyvar': 'z',
        'idvar': 'id',
        'timevar': 't',
        'controls': 'x_r',
        'fe': True,
        'tfe': True,
        'post': 3,
        'pre': 2,
        'normalize': -1,
        'cluster': True
    }


@pytest.fixture
def tolerances():
    """Standard tolerances for comparing R and Python results.
    
    Returns:
        dict: Dictionary with tolerance values for different metrics
    """
    return {
        'coefficient': 1e-3,  # Absolute tolerance for coefficient estimates
        'se': 5e-3,           # Absolute tolerance for standard errors
        'pvalue': 5e-3        # Absolute tolerance for p-values
    }


@pytest.fixture(scope="session")
def r_available():
    """Check if R and eventstudyr are available."""
    try:
        # Check R
        result = subprocess.run(['R', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            return False
        
        # Check eventstudyr package
        r_code = "if (!require('eventstudyr', quietly=TRUE)) stop('not installed')"
        result = subprocess.run(['R', '--slave', '-e', r_code], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


# Pytest hooks for better test organization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "r_required: test requires R and eventstudyr package")
    config.addinivalue_line("markers", "slow: slow running test")
    config.addinivalue_line("markers", "unit: unit test")
    config.addinivalue_line("markers", "integration: integration test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection - all tests require R for comparison."""
    try:
        # Check R availability once
        result = subprocess.run(['R', '--version'], capture_output=True, text=True)
        r_available = result.returncode == 0
        
        if r_available:
            # Check eventstudyr
            r_code = "if (!require('eventstudyr', quietly=TRUE)) stop('not installed')"
            result = subprocess.run(['R', '--slave', '-e', r_code], capture_output=True, text=True)
            r_available = result.returncode == 0
    except:
        r_available = False
    
    if not r_available:
        # Instead of skipping, we'll let tests fail naturally
        # This ensures R comparison is always required
        pass