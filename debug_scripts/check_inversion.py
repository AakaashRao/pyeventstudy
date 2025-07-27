"""Check the lead variable inversion issue"""

import pandas as pd
from eventstudypy.data_prep import prepare_event_study_data

# Create simple test data
data = pd.DataFrame({
    'id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    't': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'z': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # Treatment starts at t=6
    'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

print("Original data:")
print(data)

# Prepare event study data
prepared, policy_vars, _ = prepare_event_study_data(
    data, 'id', 't', 'z', 
    post=2, overidpost=3, 
    pre=1, overidpre=4, 
    static=False
)

print("\nPolicy variables created:", policy_vars)
print("\nPrepared data (z-related columns only):")
z_cols = ['t', 'z'] + [col for col in prepared.columns if col.startswith('z_') and col != 'z']
print(prepared[z_cols])

# Check if z_lead4 is inverted
print("\nChecking z_lead4 values:")
print("At t=2 (4 periods before treatment at t=6):")
print(f"  z at t=2: {data[data['t']==2]['z'].values[0]}")
print(f"  z at t=6: {data[data['t']==6]['z'].values[0]}")
print(f"  z_lead4 at t=2: {prepared[prepared['t']==2]['z_lead4'].values[0]}")
print("  Expected z_lead4 = z[t+4] = z[6] = 1")
print("  But if inverted, z_lead4 = 1 - z[t+4] = 1 - 1 = 0")