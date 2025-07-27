"""
eventstudypy: Python implementation of event study estimation

A Python port of the R eventstudyr package for estimating linear panel event study models,
following the recommendations in Freyaldenhoven et al. (2021).
"""

from .event_study import event_study
from .plotting import event_study_plot, add_confidence_intervals
from .testing import linear_hypothesis_test as test_linear
from .data_prep import compute_shifts

# Create an alias for add_cis to match R package naming
add_cis = add_confidence_intervals

__version__ = "0.1.0"
__all__ = ["event_study", "event_study_plot", "test_linear", "add_cis", "compute_shifts", "add_confidence_intervals"]