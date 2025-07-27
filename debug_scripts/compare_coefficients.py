"""Compare actual regression coefficients between R and Python"""

import pandas as pd
import numpy as np
from eventstudypy import event_study
import subprocess
import tempfile

# Load the actual test data
data = pd.read_csv('eventstudypy/example_data.csv')

# Run a simple model
params = {
    'outcomevar': 'y_base',
    'policyvar': 'z', 
    'idvar': 'id',
    'timevar': 't',
    'fe': True,
    'tfe': True,
    'post': 3,
    'pre': 2,
    'normalize': -1,
    'cluster': True
}

print("Running Python event study...")
py_results = event_study(data=data, **params)

print("\nPython coefficients:")
py_model = py_results['output']
for coef in py_model.coef().index:
    if coef.startswith('z_'):
        print(f"  {coef}: {py_model.coef()[coef]:.10f}")

# Check what normalization was applied
print(f"\nNormalization column: {py_results['arguments']['normalization_column']}")
print(f"Event study coefficients used: {py_results['arguments']['eventstudy_coefficients']}")

# Save data for R
with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
    data.to_csv(f.name, index=False)
    data_path = f.name

# Run R
r_code = f"""
library(eventstudyr)
data <- read.csv('{data_path}')

results <- EventStudy(
    estimator = "OLS", data = data,
    outcomevar = "y_base", policyvar = "z",
    idvar = "id", timevar = "t",
    FE = TRUE, TFE = TRUE,
    post = 3, pre = 2, 
    normalize = -1, cluster = TRUE
)

# Show coefficients
coefs <- estimatr::tidy(results$output)
print(coefs[grep("^z_", coefs$term), c("term", "estimate")])

# Check what variables were created
cat("\\nVariables in regression:\\n")
print(names(results$output$coefficients))
"""

result = subprocess.run(['Rscript', '-e', r_code], capture_output=True, text=True)
print("\n\nR output:")
print(result.stdout)
if result.stderr:
    print("\nR stderr:")
    print(result.stderr)

# Direct comparison
print("\n\nDirect comparison of lead variables:")
print("If Python z_lead3 = -0.099 and R z_lead3 = 0.099, then Python is using (1 - z_lead3)")
print("This suggests the inversion is still happening somewhere")