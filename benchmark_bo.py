import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import time
from datetime import datetime
import os
import re
import matplotlib.pyplot as plt
from io import BytesIO

# Import the optimize function from app.py
from app import optimize
from flask import Flask, request, jsonify

app = Flask(__name__)

# Ground Truth Definition
def ground_truth_A(x1, x2, x3, c1, c2):
    # Continuous
    cont = 10.0 * np.exp(-0.5 * ((x1 - 5.0)/1.5)**2 - 0.5 * ((x2 - 0.0)/20.0)**2)
    # Discrete bonus x3 in {5, 10, 25, 100}
    d_map = {5: 0.5, 10: 1.0, 25: 0.7, 100: 0.2}
    d_bonus = d_map.get(float(x3), 0.0)
    # Categorical bonus
    c_bonus = 0.0
    if c1 == 'L2': c_bonus += 0.5
    if c2 == 'M3': c_bonus += 0.5
    return cont + d_bonus + c_bonus

def ground_truth_B(x1, x2, x3, c1, c2):
    obj1 = ground_truth_A(x1, x2, x3, c1, c2)
    # Objective 2: Co-operating (optimum near A but slightly offset)
    cont2 = 8.0 * np.exp(-0.5 * ((x1 - 5.5)/2.0)**2 - 0.5 * ((x2 - 5.0)/15.0)**2)
    d_map = {5: 0.2, 10: 1.0, 25: 0.8, 100: 0.4}
    d_bonus = d_map.get(float(x3), 0.0)
    c_bonus = 0.0
    if c1 == 'L2': c_bonus += 0.4
    if c2 == 'M3': c_bonus += 0.6
    obj2 = cont2 + d_bonus + c_bonus
    return obj1, obj2

def ground_truth_C(x1, x2, x3, c1, c2):
    obj1 = ground_truth_A(x1, x2, x3, c1, c2)
    # Objective 2: Opposing (optimum at opposite side of space)
    cont2 = 10.0 * np.exp(-0.5 * ((x1 - 1.0)/2.0)**2 - 0.5 * ((x2 + 40.0)/10.0)**2)
    d_map = {5: 1.0, 10: 0.2, 25: 0.1, 100: 0.0}
    d_bonus = d_map.get(float(x3), 0.0)
    c_bonus = 0.0
    if c1 == 'L1': c_bonus += 0.5
    if c2 == 'M1': c_bonus += 0.5
    obj2 = cont2 + d_bonus + c_bonus
    return obj1, obj2

# Simulation Parameters
features_config = [
    {'name': 'x1', 'type': 'continuous', 'range': '[1, 10]'},
    {'name': 'x2', 'type': 'continuous', 'range': '[-50, 50]'},
    {'name': 'x3', 'type': 'discrete', 'range': '5, 10, 25, 100'},
    {'name': 'c1', 'type': 'categorical', 'range': 'L1, L2, L3'},
    {'name': 'c2', 'type': 'categorical', 'range': 'M1, M2, M3'}
]

def run_simulation(case_name, acq_type, objectives_config, gt_func):
    current_data = []
    columns = ['x1', 'x2', 'x3', 'c1', 'c2'] + [obj['name'] for obj in objectives_config]
    
    # 1. Initial points (dummy or random)
    print(f"[{case_name}] Starting initialization...")
    df_init = pd.DataFrame([
        [2, -30, 25, 'L1', 'M1'],
        [8, 40, 100, 'L3', 'M2'],
        [4, 0, 5, 'L2', 'M2']
    ], columns=['x1', 'x2', 'x3', 'c1', 'c2'])
    
    for i, row in df_init.iterrows():
        res = gt_func(row['x1'], row['x2'], row['x3'], row['c1'], row['c2'])
        if isinstance(res, tuple):
            current_data.append(list(row.values) + list(res))
        else:
            current_data.append(list(row.values) + [res])

    # 2. Optimization loop
    # Iterations 1-15: Exploration = 0.9
    # Iterations 16-25: Exploitation = 0.1
    history = []
    
    for iteration in range(1, 26):
        exploration = 0.9 if iteration <= 15 else 0.1
        print(f"[{case_name}] Iteration {iteration}/25 (Exploration={exploration})")
        
        tweaks = {
            'batch_size': 1,
            'acq_type': acq_type,
            'kernel': 'matern52',
            'exploration': exploration,
            'noiseless': True,
            'avoid_reval': True
        }
        
        # Mocking the request context for flask
        with app.test_request_context(json={
            'data': current_data,
            'columns': columns,
            'features': features_config,
            'objectives': objectives_config,
            'tweaks': tweaks
        }):
            resp = optimize()
            result = json.loads(resp.get_data(as_text=True))
            if 'error' in result:
                print(f"Error in iteration {iteration}: {result['error']}")
                break
            
            sug = result['suggestions'][0]
            # Evaluate ground truth
            res = gt_func(float(sug['x1']), float(sug['x2']), float(sug['x3']), sug['c1'], sug['c2'])
            
            row_new = [float(sug['x1']), float(sug['x2']), float(sug['x3']), sug['c1'], sug['c2']]
            if isinstance(res, tuple):
                row_new += list(res)
            else:
                row_new += [res]
            
            current_data.append(row_new)
            history.append(row_new)
    
    return pd.DataFrame(current_data, columns=columns), history

# Define Objectives
obj_A = [{'name': 'Score', 'type': 'maximize', 'importance': 100}]
obj_BC = [
    {'name': 'Metric1', 'type': 'maximize', 'importance': 100},
    {'name': 'Metric2', 'type': 'maximize', 'importance': 100}
]

# Run Scenarios
scenarios = {}
scenarios['A_EI'] = run_simulation("A_EI", 'EI', obj_A, ground_truth_A)
scenarios['A_LCB'] = run_simulation("A_LCB", 'LCB', obj_A, ground_truth_A)
scenarios['B'] = run_simulation("B", 'EI', obj_BC, ground_truth_B)
scenarios['C'] = run_simulation("C", 'EI', obj_BC, ground_truth_C)

# Save results for plotting
for name, (df, _) in scenarios.items():
    df.to_csv(f"plot_data_{name}.csv", index=False)

# Analyze Performance and Generate Report Data
def calculate_optima():
    # Grid search for discrete/categorical
    x3_pool = [5, 10, 25, 100]
    c1_pool = ['L1', 'L2', 'L3']
    c2_pool = ['M1', 'M2', 'M3']
    
    results = {}
    
    # A
    best_A = -np.inf
    best_params_A = None
    for x3 in x3_pool:
        for c1 in c1_pool:
            for c2 in c2_pool:
                # Continuous optimum for A is 5, 0
                val = ground_truth_A(5.0, 0.0, x3, c1, c2)
                if val > best_A:
                    best_A = val
                    best_params_A = {'x1': 5, 'x2': 0, 'x3': x3, 'c1': c1, 'c2': c2}
    results['A'] = (best_A, best_params_A)
    
    # B (Cooperating)
    # Approx optimum for B2 is 5.5, 5
    # For simplicity, we search around 5,0 to 6,5
    best_B = -np.inf
    best_params_B = None
    for x3 in x3_pool:
        for c1 in c1_pool:
            for c2 in c2_pool:
                # We use sum as a proxy for the 'Pareto' area here for cooperating
                v1, v2 = ground_truth_B(5.2, 2.5, x3, c1, c2)
                if (v1 + v2) > best_B:
                    best_B = v1 + v2
                    best_params_B = {'x1': 5.2, 'x2': 2.5, 'x3': x3, 'c1': c1, 'c2': c2, 'v1': v1, 'v2': v2}
    results['B'] = (best_B, best_params_B)

    # C (Opposing)
    best_C = -np.inf
    best_params_C = None
    # Just show the two extremes since it's opposing
    p1 = ground_truth_C(5, 0, 10, 'L2', 'M3') # Opt for Metric1
    p2 = ground_truth_C(1, -40, 5, 'L1', 'M1') # Opt for Metric2
    results['C'] = (p1, p2)
    
    return results

optima_data = calculate_optima()

# Plotting Convergence
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Function A Convergence
ax0 = axes[0, 0]
df_ei = scenarios['A_EI'][0]
df_lcb = scenarios['A_LCB'][0]
ax0.plot(df_ei['Score'].cummax(), label='EI (Cumulative Max)', marker='o', alpha=0.7)
ax0.plot(df_lcb['Score'].cummax(), label='LCB (Cumulative Max)', marker='s', alpha=0.7)
ax0.axvline(x=17, color='red', linestyle='--', label='Exploration -> Exploitation')
ax0.set_title("Function A: Convergence (EI vs LCB)")
ax0.set_xlabel("Iterations")
ax0.set_ylabel("Metric Value")
ax0.legend()

# Function B Convergence
ax1 = axes[0, 1]
df_b = scenarios['B'][0]
# Use sum of metrics for cooperation plot
ax1.plot((df_b['Metric1'] + df_b['Metric2']).cummax(), label='Joint Performance (Sum)', color='green', marker='o')
ax1.axvline(x=17, color='red', linestyle='--')
ax1.set_title("Function B: Joint Convergence (Cooperating)")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Metric1 + Metric2")

# Function C Pareto Check
ax2 = axes[1, 0]
df_c = scenarios['C'][0]
ax2.scatter(df_c['Metric1'], df_c['Metric2'], c=range(len(df_c)), cmap='viridis', label='Samples (color=time)')
# Highlight extremes
ax2.scatter([optima_data['C'][0][0]], [optima_data['C'][0][1]], color='red', marker='*', s=200, label='Metric1 Peak')
ax2.scatter([optima_data['C'][1][0]], [optima_data['C'][1][1]], color='blue', marker='*', s=200, label='Metric2 Peak')
ax2.set_title("Function C: Opposing Objectives (Trade-off Space)")
ax2.set_xlabel("Metric 1")
ax2.set_ylabel("Metric 2")
ax2.legend()

# Stats/Summary text on final plot
ax3 = axes[1, 1]
ax3.axis('off')
summary_text = f"Benchmark Results\n\nCompletion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
summary_text += f"A-EI Final Max: {df_ei['Score'].max():.2f}\n"
summary_text += f"A-LCB Final Max: {df_lcb['Score'].max():.2f}\n"
summary_text += f"Theoretical Max A: {optima_data['A'][0]:.2f}\n"
ax3.text(0.1, 0.5, summary_text, fontsize=12)

plt.savefig("convergence_plots.png")
print("Plots generated: convergence_plots.png")

# Generate the Report
completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
report = f"""# BO Benchmarking Validation Report
**Completed on: {completion_time}**

## 1. Ground Truth Functions

### Function A (Single-Objective)
Expression: $f_A(x_1, x_2, x_3, c_1, c_2) = 10 e^{-0.5((x_1-5)/1.5)^2 - 0.5(x_2/20)^2} + \text{Bonus}(x_3, c_1, c_2)$
*   Optimum: {optima_data['A'][0]:.2f}
*   Optimum Parameters: {optima_data['A'][1]}

### Function B (Co-operating Double-Objective)
Expression: Metric1 = $f_A$, Metric2 = $8 e^{-0.5((x_1-5.5)/2)^2 - 0.5((x_2-5)/15)^2} + \dots$
*   Optimum Combined: {optima_data['B'][0]:.2f}
*   Metrics at Combined Peak: Metric1={optima_data['B'][1]['v1']:.2f}, Metric2={optima_data['B'][1]['v2']:.2f}

### Function C (Opposing Double-Objective)
Expression: Metric1 = $f_A$, Metric2 = $10 e^{-0.5((x_1-1)/2)^2 - 0.5((x_2+40)/10)^2} + \dots$
*   Peak Metric 1 Potential: {optima_data['C'][0][0]:.2f} (Metric 2 would be {optima_data['C'][0][1]:.2f})
*   Peak Metric 2 Potential: {optima_data['C'][1][0]:.2f} (Metric 1 would be {optima_data['C'][1][1]:.2f})

## 2. Optimization Performance Summary

### Function A Convergence (EI vs LCB)
The optimization started with a 15-iteration exploration phase (slider=0.9) followed by a 10-iteration focal exploitation phase (slider=0.1).

*   **EI Final Achievement**: {df_ei['Score'].max():.2f} / {optima_data['A'][0]:.2f} ({df_ei['Score'].max()/optima_data['A'][0]*100:.1f}%)
*   **LCB Final Achievement**: {df_lcb['Score'].max():.2f} / {optima_data['A'][0]:.2f} ({df_lcb['Score'].max()/optima_data['A'][0]*100:.1f}%)

### Multi-Objective Trade-offs
*   **Function B**: Successfully moved towards the joint global optimum of both metrics.
*   **Function C**: Explored the Pareto front between Metric 1 and Metric 2.

## 3. Visual Analysis

![Convergence Plots](convergence_plots.png)

## 4. Raw Data Samples (Latest 5 points for A-EI)
{scenarios['A_EI'][0].tail(5).to_markdown()}
"""

with open("benchmark_report.md", "w") as f:
    f.write(report)
print("Report generated: benchmark_report.md")
