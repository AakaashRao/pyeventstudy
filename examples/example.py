"""
Example usage of eventstudypy package
"""

import pandas as pd
import matplotlib.pyplot as plt
from eventstudypy import event_study, event_study_plot, test_linear

# Load example data
data = pd.read_csv('eventstudypy/example_data.csv')

print("Example 1: Basic Event Study")
print("-" * 50)

# Minimal example
results_basic = event_study(
    data=data,
    outcomevar="y_base",
    policyvar="z",
    idvar="id",
    timevar="t",
    pre=0,
    post=3,
    normalize=-1
)

print("Model output:")
print(results_basic['output'].summary())

print("\nExample 2: Dynamic Model with Controls")
print("-" * 50)

# Dynamic OLS model with anticipation effects and controls
results_dynamic = event_study(
    data=data,
    outcomevar="y_base",
    policyvar="z",
    idvar="id",
    timevar="t",
    controls="x_r",
    fe=True,
    tfe=True,
    post=3,
    overidpost=5,
    pre=2,
    overidpre=4,
    normalize=-3,
    cluster=True,
    anticipation_effects_normalization=True
)

print("Model output:")
print(results_dynamic['output'].summary())

# Test for pre-trends and leveling-off
test_results = test_linear(results_dynamic)
print("\nHypothesis tests:")
print(test_results)

print("\nExample 3: Static Model")
print("-" * 50)

# Static model
results_static = event_study(
    data=data,
    outcomevar="y_jump_m",
    policyvar="z",
    idvar="id",
    timevar="t",
    fe=True,
    tfe=True,
    post=0,
    overidpost=0,
    pre=0,
    overidpre=0,
    cluster=True
)

print("Model output:")
print(results_static['output'].summary())

print("\nExample 4: Event Study Plot")
print("-" * 50)

# Create event study plot for dynamic model
# Set random seed for reproducible sup-t bands
import numpy as np
np.random.seed(10)

fig = event_study_plot(
    results_dynamic,
    conf_level=0.95,
    supt=0.95,
    pre_event_coeffs=True,
    post_event_coeffs=True
)

plt.savefig('event_study_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'event_study_plot.png'")

# Show the plot
plt.show()

print("\nExample 5: Customized Plot")
print("-" * 50)

# Customized plot
fig2 = event_study_plot(
    results_dynamic,
    xtitle="Relative time",
    ytitle="",
    ybreaks=[-2, -1, 0, 1, 2],
    conf_level=0.95,
    supt=None,  # No sup-t bands
    add_zero_line=True
)

plt.savefig('event_study_plot_custom.png', dpi=300, bbox_inches='tight')
print("Customized plot saved as 'event_study_plot_custom.png'")
plt.show()