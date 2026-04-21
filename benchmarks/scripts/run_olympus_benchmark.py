import os
import sys
import numpy as np
import pandas as pd
import torch
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Patch matplotlib for olympus
plt.register_cmap = colormaps.register

import olympus
from olympus.datasets import Dataset
from olympus.emulators import Emulator

# Add EDBO+ to path
EDBO_PATH = os.path.join(os.getcwd(), 'edboplus')
if EDBO_PATH not in sys.path:
    sys.path.append(EDBO_PATH)
from edbo.plus.optimizer_botorch import EDBOplus

# Add app to path for EDOS-native
from app import optimize
from flask import Flask

app_mock = Flask(__name__)

# DEVICE Setup (from app.py)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_benchmark(dataset_name, budget=30, n_init=3):
    print(f"\n--- Benchmarking on {dataset_name} ---")
    
    # 1. Load Olympus Emulator
    try:
        emulator = Emulator(dataset=dataset_name, model='BayesNeuralNet')
        print(f"Loaded emulator for {dataset_name}")
    except Exception as e:
        print(f"Error loading emulator: {e}")
        # Fallback to Dataset
        dataset = Dataset(kind=dataset_name)
        # Olympus Dataset doesn't have a direct predict, it's a fixed data table
        # We'll use the closest point or something. 
        # But Emulator is better for continuous.
        return None

    param_space = emulator.param_space
    features_config = []
    
    for param in param_space:
        name = param.name
        p_type = param.type
        if p_type == 'continuous':
            features_config.append({'name': name, 'type': 'continuous', 'range': f"[{param.low}, {param.high}]"})
        elif p_type == 'categorical':
            features_config.append({'name': name, 'type': 'categorical', 'range': ", ".join(param.options)})
        elif p_type == 'discrete':
            features_config.append({'name': name, 'type': 'discrete', 'range': ", ".join([str(x) for x in param.options])})

    objectives_config = [{'name': 'yield', 'type': 'maximize', 'importance': 1}]
    
    results = {}

    # --- EDOS-native Benchmark ---
    print("Running EDOS-native...")
    current_data = []
    columns = [f['name'] for f in features_config] + ['yield']
    
    # Initial points
    for _ in range(n_init):
        # Random initial point
        sample = {}
        row = []
        for f in features_config:
            if f['type'] == 'continuous':
                low, high = eval(f['range'])
                val = np.random.uniform(low, high)
            elif f['type'] == 'categorical':
                opts = [x.strip() for x in f['range'].split(',')]
                val = np.random.choice(opts)
            else: # discrete
                opts = [float(x.strip()) for x in f['range'].split(',')]
                val = np.random.choice(opts)
            sample[f['name']] = val
            row.append(val)
        
        # Evaluate
        # Olympus expected input is list of dicts or array
        obs = emulator.run([list(sample.values())])[0][0]
        row.append(obs)
        current_data.append(row)

    edos_history = [row[-1] for row in current_data]
    
    for i in range(budget - n_init):
        print(f"  Iteration {i+1}/{budget-n_init}")
        tweaks = {'batch_size': 1, 'acq_type': 'EI', 'exploration': 0.5, 'noiseless': True, 'avoid_reval': True}
        
        with app_mock.test_request_context(json={
            'data': current_data,
            'columns': columns,
            'features': features_config,
            'objectives': objectives_config,
            'tweaks': tweaks
        }):
            resp = optimize()
            res_json = json.loads(resp.get_data(as_text=True))
            if 'suggestions' in res_json:
                sug = res_json['suggestions'][0]
                obs = emulator.run([list(sug.values())])[0][0]
                edos_history.append(obs)
                current_data.append([sug[f['name']] for f in features_config] + [obs])
    
    results['EDOS-native'] = edos_history

    # --- EDBO+ Benchmark ---
    print("Running EDBO+...")
    edbo = EDBOplus()
    
    # For EDBO+, transform continuous to regular with 50 steps
    edbo_features = []
    components = {}
    for f in features_config:
        if f['type'] == 'continuous':
            low, high = eval(f['range'])
            steps = 50
            edbo_features.append({'name': f['name'], 'type': 'regular', 'range': f"[{low}, {high}, {steps}]"})
            components[f['name']] = np.linspace(low, high, steps).tolist()
        else:
            edbo_features.append(f)
            if f['type'] == 'categorical':
                components[f['name']] = [x.strip() for x in f['range'].split(',')]
            else:
                components[f['name']] = [float(x.strip()) for x in f['range'].split(',')]

    scope_file = f"scope_edbo_{dataset_name}.csv"
    edbo.generate_reaction_scope(components, filename=scope_file, check_overwrite=False)
    df_scope = pd.read_csv(scope_file)
    df_scope['yield'] = 'PENDING'
    
    # Inject same initial points (find closest in scope)
    for row in current_data[:n_init]:
        # Simple closest point matching
        # ... (implementation from benchmark_edbo.py)
        diff = 0
        for idx, f in enumerate(features_config):
            if f['type'] == 'categorical':
                diff += (df_scope[f['name']].astype(str) != str(row[idx])).astype(int)
            else:
                diff += (df_scope[f['name']] - float(row[idx])).abs()
        best_idx = diff.idxmin()
        df_scope.loc[best_idx, 'yield'] = row[-1]
    
    df_scope.to_csv(scope_file, index=False)
    
    edbo_history = [row[-1] for row in current_data[:n_init]]
    
    for i in range(budget - n_init):
        print(f"  Iteration {i+1}/{budget-n_init}")
        edbo.run(objectives=['yield'], objective_mode=['max'], batch=1, filename=scope_file, seed=42)
        df_updated = pd.read_csv(scope_file)
        suggestion = df_updated.sort_values(by='priority', ascending=False).iloc[0]
        
        # Evaluate Suggestion
        sug_dict = {f['name']: suggestion[f['name']] for f in features_config}
        obs = emulator.run([list(sug_dict.values())])[0][0]
        edbo_history.append(obs)
        
        # Update CSV
        idx = suggestion.name
        df_updated.loc[idx, 'yield'] = obs
        df_updated.to_csv(scope_file, index=False)
        
    results['EDBO+'] = edbo_history
    
    return results

# Datasets to test
datasets = ['suzuki_i', 'suzuki_iv']
all_results = {}

for ds in datasets:
    res = run_benchmark(ds, budget=40, n_init=3)
    if res:
        all_results[ds] = res

# Save results
with open('olympus_benchmark_results.json', 'w') as f:
    json.dump(all_results, f)

# Generate Plots
for ds, res in all_results.items():
    plt.figure(figsize=(10, 6))
    for model, history in res.items():
        cum_max = np.maximum.accumulate(history)
        plt.plot(range(1, len(history)+1), cum_max, label=model, marker='o')
    
    # Mock some data for Figure S2.2 style (GPyOpt, Phoenics)
    # These are illustrative baselines if I can't read the PDF
    # But I'll try to find better ones if possible.
    # For now, let's just plot our results.
    
    plt.xlabel('Number of Experiments')
    plt.ylabel('Yield (%)')
    plt.title(f'Benchmarking on {ds}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'convergence_{ds}.png')
    plt.close()

# Report Generation
print("\nBenchmark tests completed.")
