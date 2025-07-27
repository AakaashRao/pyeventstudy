"""
Basic tests to verify the package works correctly.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eventstudypy import event_study


class TestBasicFunctionality:
    """Basic tests for package functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Load example data before each test."""
        self.data = pd.read_csv(Path(__file__).parent.parent / 'eventstudypy' / 'example_data.csv')
    
    def test_data_loading(self):
        """Test that example data loads correctly."""
        assert self.data is not None
        assert self.data.shape[0] > 0
        assert 'y_base' in self.data.columns
        assert 'z' in self.data.columns
        assert 'id' in self.data.columns
        assert 't' in self.data.columns
    
    def test_basic_event_study(self):
        """Test basic event study functionality."""
        results = event_study(
            data=self.data,
            outcomevar="y_base",
            policyvar="z",
            idvar="id",
            timevar="t",
            pre=0,
            post=3,
            normalize=-1
        )
        
        assert results is not None
        assert 'output' in results
        assert results['output'] is not None
        
        # Check coefficients exist
        coefs = results['output'].coef()
        assert len(coefs) > 0
        
        # Check that normalized coefficient is not in results
        assert 'z_fd_lead1' not in coefs.index
    
    def test_event_study_with_controls(self):
        """Test event study with control variables."""
        results = event_study(
            data=self.data,
            outcomevar="y_base",
            policyvar="z",
            idvar="id",
            timevar="t",
            controls="x_r",
            pre=2,
            post=2,
            normalize=-1
        )
        
        assert results is not None
        assert 'output' in results
        
        # Check that control variable is in the model
        coefs = results['output'].coef()
        assert 'x_r' in coefs.index
    
    def test_event_study_with_fixed_effects(self):
        """Test event study with fixed effects."""
        results = event_study(
            data=self.data,
            outcomevar="y_base",
            policyvar="z",
            idvar="id",
            timevar="t",
            fe=True,
            tfe=True,
            pre=1,
            post=2,
            normalize=0
        )
        
        assert results is not None
        assert 'output' in results
        
        # Check that results contain expected shift variables
        coefs = results['output'].coef()
        assert any('z_' in str(idx) for idx in coefs.index)
    
    def test_static_model(self):
        """Test static event study model."""
        results = event_study(
            data=self.data,
            outcomevar="y_jump_m",
            policyvar="z",
            idvar="id",
            timevar="t",
            fe=True,
            tfe=True,
            post=0,
            pre=0,
            overidpost=0,
            overidpre=0,
            cluster=True
        )
        
        assert results is not None
        assert 'output' in results
        
        # Static model should have only z (not z_fd)
        coefs = results['output'].coef()
        z_vars = [idx for idx in coefs.index if idx.startswith('z')]
        assert len(z_vars) == 1
        assert 'z' in z_vars