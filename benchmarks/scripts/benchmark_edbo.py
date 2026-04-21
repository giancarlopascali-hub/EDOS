import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Add EDBO+ to path
EDBO_PATH = os.path.join(os.getcwd(), 'edboplus')
if EDBO_PATH not in sys.path:
    sys.path.append(EDBO_PATH)

from edbo.plus.optimizer_botorch import EDBOplus

# Ground Truth Definition (Replicated from benchmark_bo.py)
def ground_truth_A(x1, x2, x3, c1, c2):
    cont = 10.0 * np.exp(-0.5 * ((x1 - 5.0)/1.5)**2 - 0.5 * ((x2 - 0.0)/20.0)**2)
    d_map = {5.0: 0.5, 10.0: 1.0, 25.0: 0.7, 100.0: 0.2}
    d_bonus = d_map.get(float(x3), 0.0)
    c_bonus = 0.0
    if c1 == 'L2': c_bonus += 0.5
    if c2 == 'M3': c_bonus += 0.5
    return cont + d_bonus + c_bonus

def ground_truth_B(x1, x2, x3, c1, c2):
    obj1 = ground_truth_A(x1, x2, x3, c1, c2)
    cont2 = 8.0 * np.exp(-0.5 * ((x1 - 5.5)/2.0)**2 - 0.5 * ((x2 - 5.0)/15.0)**2)
    d_map = {5.0: 0.2, 10.0: 1.0, 25.0: 0.8, 100.0: 0.4}
    d_bonus = d_map.get(float(x3), 0.0)
    c_bonus = 0.0
    if c1 == 'L2': c_bonus += 0.4
    if c2 == 'M3': c_bonus += 0.6
    obj2 = cont2 + d_bonus + c_bonus
    return obj1, obj2

def ground_truth_C(x1, x2, x3, c1, c2):
    obj1 = ground_truth_A(x1, x2, x3, c1, c2)
    cont2 = 10.0 * np.exp(-0.5 * ((x1 - 1.0)/2.0)**2 - 0.5 * ((x2 + 40.0)/10.0)**2)
    d_map = {5.0: 1.0, 10.0: 0.2, 25.0: 0.1, 100.0: 0.0}
    d_bonus = d_map.get(float(x3), 0.0)
    c_bonus = 0.0
    if c1 == 'L1': c_bonus += 0.5
    if c2 == 'M1': c_bonus += 0.5
    obj2 = cont2 + d_bonus + c_bonus
    return obj1, obj2

# 1. SCOPE GENERATION
print("Generating search scope...")
x1_vals = np.linspace(1, 10, 10)
x2_vals = np.linspace(-50, 50, 10)
x3_vals = [5.0, 10.0, 25.0, 100.0]
c1_vals = ['L1', 'L2', 'L3']
c2_vals = ['M1', 'M2', 'M3']

components = {
    'x1': x1_vals.tolist(),
    'x2': x2_vals.tolist(),
    'x3': x3_vals,
    'c1': c1_vals,
    'c2': c2_vals
}

# Initial Points (Exactly matching EDOS benchmark)
init_points = [
    {'x1': 2.0, 'x2': -30.0, 'x3': 25.0, 'c1': 'L1', 'c2': 'M1'},
    {'x1': 8.0, 'x2': 40.0, 'x3': 100.0, 'c1': 'L3', 'c2': 'M2'},
    {'x1': 4.0, 'x2': 0.0, 'x3': 5.0, 'c1': 'L2', 'c2': 'M2'}
]

def run_edbo_benchmark(case_name, objective_names, gt_func):
    print(f"\n--- Starting EDBO+ Benchmark: {case_name} ---")
    
    # Create a fresh scope CSV for this run
    scope_file = f"scope_{case_name}.csv"
    edbo = EDBOplus()
    edbo.generate_reaction_scope(components, filename=scope_file, check_overwrite=False)
    
    # Load the scope and inject initial points
    df_scope = pd.read_csv(scope_file)
    
    # Initialize all objective columns with 'PENDING'
    for obj_name in objective_names:
        df_scope[obj_name] = 'PENDING'
    
    # We need to find the closest points in the discrete scope to our requested initial points
    for pt in init_points:
        # Find index of closest match
        diff = (df_scope['x1'] - pt['x1']).abs() + \
               (df_scope['x2'] - pt['x2']).abs() + \
               (df_scope['x3'] - pt['x3']).abs() + \
               (df_scope['c1'].astype(str) != str(pt['c1'])).astype(int) + \
               (df_scope['c2'].astype(str) != str(pt['c2'])).astype(int)
        idx = diff.idxmin()
        
        # Evaluate ground truth for the discrete match
        res = gt_func(df_scope.loc[idx, 'x1'], df_scope.loc[idx, 'x2'], 
                      df_scope.loc[idx, 'x3'], df_scope.loc[idx, 'c1'], df_scope.loc[idx, 'c2'])
        
        if isinstance(res, tuple):
            for i, obj_name in enumerate(objective_names):
                df_scope.loc[idx, obj_name] = res[i]
        else:
            df_scope.loc[idx, objective_names[0]] = res
            
    df_scope.to_csv(scope_file, index=False)
    
    # Pre-check: Verify we have 3 points without 'PENDING'
    df_check = pd.read_csv(scope_file)
    obs_count = len(df_check[~df_check[objective_names[0]].astype(str).str.contains('PENDING', case=False)])
    print(f"DEBUG: Initial observations found in CSV: {obs_count}")
    if obs_count == 0:
        print("ERROR: No observations were written to the CSV correctly!")
        return
    
    # Optimization Loop (20 iterations)
    for i in range(1, 21):
        print(f"Iteration {i}/20...")
        
        # EDBO+ run
        # For single objective, use 'max'
        obj_modes = ['max'] * len(objective_names)
        
        # Run optimizer
        edbo.run(
            objectives=objective_names,
            objective_mode=obj_modes,
            batch=1,
            filename=scope_file,
            seed=42, # Consistent seed
            acquisition_function='NoisyEHVI'
        )
        
        # Read the updated scope to see the next suggestion (priority column)
        df_updated = pd.read_csv(scope_file)
        # The top suggested point has the highest priority
        suggestion = df_updated.sort_values(by='priority', ascending=False).iloc[0]
        
        # Evaluate and update the CSV
        res = gt_func(suggestion['x1'], suggestion['x2'], suggestion['x3'], suggestion['c1'], suggestion['c2'])
        
        # Update the specific row in the CSV
        # We find the row index in the original df_updated
        idx = suggestion.name
        
        if isinstance(res, tuple):
            for j, obj_name in enumerate(objective_names):
                df_updated.loc[idx, obj_name] = res[j]
        else:
            df_updated.loc[idx, objective_names[0]] = res
            
        # Reset priority list for next run (or EDBO handles it by checking PENDING)
        # Actually EDBO+ just looks for 'PENDING'
        df_updated.to_csv(scope_file, index=False)

    # Save final results for comparison
    df_final = pd.read_csv(scope_file)
    # Extract only evaluated points
    evaluated = df_final[df_final[objective_names[0]] != 'PENDING'].copy()
    # Convert objective columns to float
    for obj in objective_names:
        evaluated[obj] = pd.to_numeric(evaluated[obj])
        
    evaluated.to_csv(f"plot_data_edbo_{case_name}.csv", index=False)
    print(f"Benchmark {case_name} complete. Saved to plot_data_edbo_{case_name}.csv")

# Run Benchmarks
run_edbo_benchmark("A", ["Score"], ground_truth_A)
run_edbo_benchmark("B", ["Metric1", "Metric2"], ground_truth_B)
run_edbo_benchmark("C", ["Metric1", "Metric2"], ground_truth_C)

print("\nALL EDBO+ BENCHMARKS COMPLETE.")
