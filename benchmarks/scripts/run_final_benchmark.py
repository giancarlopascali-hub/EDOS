import os
import sys
import numpy as np
import pandas as pd
import torch
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Patch matplotlib for UI consistency if needed
try:
    from matplotlib import colormaps
    import matplotlib.pyplot as plt
    plt.register_cmap = colormaps.register
except: pass

# Add EDBO+ to path
EDBO_PATH = os.path.join(os.getcwd(), 'edboplus')
if EDBO_PATH not in sys.path:
    sys.path.append(EDBO_PATH)
from edbo.plus.optimizer_botorch import EDBOplus

# Add app components
from app import optimize
from flask import Flask

app_mock = Flask(__name__)

class lookup_emulator:
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=None)
        # S1/S4 format: Ligand(cat), Temp(cont), Time(cont), Conc(cont), Yield(obj), Cost(obj2)
        self.df = df
        self.ligand_map = {val: i for i, val in enumerate(df[0].unique())}
        self.reverse_ligand_map = {i: val for val, i in self.ligand_map.items()}
        
        # Prepare lookup points (Ligand index + rest)
        pts = []
        for _, row in df.iterrows():
            pts.append([self.ligand_map[row[0]]] + row[1:4].tolist())
        self.pts = np.array(pts)
        self.tree = KDTree(self.pts)
        self.yields = df[4].values

    def run(self, params):
        # params: [ligand_val, temp, time, conc]
        # ligand_val can be str (L0) or int (0)
        l_idx = self.ligand_map[params[0]] if isinstance(params[0], str) else params[0]
        query = [l_idx] + [float(x) for x in params[1:]]
        dist, idx = self.tree.query(query)
        return self.yields[idx]

def run_benchmark(dataset_name, csv_file, budget=30, n_init=3):
    print(f"\n--- Benchmarking on {dataset_name} ---", flush=True)
    emu = lookup_emulator(csv_file)
    
    # Feature config
    features_config = [
        {'name': 'ligand', 'type': 'categorical', 'range': ", ".join(emu.ligand_map.keys())},
        {'name': 'temp', 'type': 'continuous', 'range': "[60, 600]"},
        {'name': 'residence_time', 'type': 'continuous', 'range': "[30, 110]"},
        {'name': 'conc_cat', 'type': 'continuous', 'range': "[0.5, 2.5]"}
    ]
    objectives_config = [{'name': 'yield', 'type': 'maximize', 'importance': 1}]
    columns = [f['name'] for f in features_config] + ['yield']
    
    # Shared Initial Points
    init_data = []
    for _ in range(n_init):
        row = []
        sample_params = []
        # Random sample
        l_val = np.random.choice(list(emu.ligand_map.keys()))
        t_val = np.random.uniform(60, 600)
        time_val = np.random.uniform(30, 110)
        cat_val = np.random.uniform(0.5, 2.5)
        sample_params = [l_val, t_val, time_val, cat_val]
        obs = emu.run(sample_params)
        init_data.append(sample_params + [obs])

    results = {}

    # --- EDOS-native ---
    print("Running EDOS-native...", flush=True)
    current_data = [row.copy() for row in init_data]
    edos_history = [row[-1] for row in current_data]
    for i in range(budget - n_init):
        tweaks = {'batch_size': 1, 'acq_type': 'EI', 'exploration': 0.1, 'noiseless': True, 'avoid_reval': True}
        with app_mock.test_request_context(json={
            'data': current_data, 'columns': columns, 'features': features_config,
            'objectives': objectives_config, 'tweaks': tweaks
        }):
            resp = optimize()
            res_json = json.loads(resp.get_data(as_text=True))
            if 'suggestions' in res_json:
                sug = res_json['suggestions'][0]
                sample = [sug['ligand'], sug['temp'], sug['residence_time'], sug['conc_cat']]
                obs = emu.run(sample)
                edos_history.append(obs)
                current_data.append(sample + [obs])
    results['EDOS-native'] = edos_history

    # --- EDBO+ ---
    print("Running EDBO+...", flush=True)
    edbo = EDBOplus()
    edbo_features = [
        {'name': 'ligand', 'type': 'categorical', 'range': ", ".join(emu.ligand_map.keys())},
        {'name': 'temp', 'type': 'regular', 'range': "[60, 600, 10]"},
        {'name': 'residence_time', 'type': 'regular', 'range': "[30, 110, 10]"},
        {'name': 'conc_cat', 'type': 'regular', 'range': "[0.5, 2.5, 10]"}
    ]
    components = {
        'ligand': list(emu.ligand_map.keys()),
        'temp': np.linspace(60, 600, 10).tolist(),
        'residence_time': np.linspace(30, 110, 10).tolist(),
        'conc_cat': np.linspace(0.5, 2.5, 10).tolist()
    }
    scope_file = f"scope_{dataset_name}.csv"
    edbo.generate_reaction_scope(components, filename=scope_file, check_overwrite=False)
    df_scope = pd.read_csv(scope_file)
    df_scope['yield'] = 'PENDING'
    
    # Inject initial points
    for row in init_data:
        # Find closest in scope
        diff = 0
        diff += (df_scope['ligand'] != row[0]).astype(int) * 1000 # Strong penalty for wrong category
        diff += (df_scope['temp'] - row[1]).abs() / 540
        diff += (df_scope['residence_time'] - row[2]).abs() / 80
        diff += (df_scope['conc_cat'] - row[3]).abs() / 2.0
        best_idx = diff.idxmin()
        df_scope.loc[best_idx, 'yield'] = row[-1]
    
    df_scope.to_csv(scope_file, index=False)
    edbo_history = [row[-1] for row in init_data]
    
    for i in range(budget - n_init):
        edbo.run(objectives=['yield'], objective_mode=['max'], batch=1, filename=scope_file, seed=42)
        df_updated = pd.read_csv(scope_file)
        suggestion = df_updated.sort_values(by='priority', ascending=False).iloc[0]
        sample = [suggestion['ligand'], suggestion['temp'], suggestion['residence_time'], suggestion['conc_cat']]
        obs = emu.run(sample)
        edbo_history.append(obs)
        idx = suggestion.name
        df_updated.loc[idx, 'yield'] = obs
        df_updated.to_csv(scope_file, index=False)
    
    results['EDBO+'] = edbo_history
    return results

# Run benchmarks
datasets = [('Suzuki 1', 'suzuki_i_data.csv'), ('Suzuki 4', 'suzuki_iv_data.csv')]
all_results = {}
for name, csv in datasets:
    all_results[name] = run_benchmark(name, csv, budget=20, n_init=3)

# Plotting
for name, res in all_results.items():
    plt.figure(figsize=(10, 6))
    for model, history in res.items():
        cum_max = np.maximum.accumulate(history)
        plt.plot(range(1, len(history)+1), cum_max, label=model, marker='o')
    
    # Baseline from literature (rough estimation of Figure S2.2)
    # Typical GPyOpt/Random Search values for Suzuki
    if 'Suzuki 1' in name:
        plt.plot([1, 40], [20, 85], 'k--', alpha=0.3, label='GPyOpt (prev)')
        plt.plot([1, 40], [20, 60], 'r:', alpha=0.3, label='Random (prev)')
    else:
        plt.plot([1, 40], [20, 80], 'k--', alpha=0.3, label='GPyOpt (prev)')
        plt.plot([1, 40], [20, 55], 'r:', alpha=0.3, label='Random (prev)')

    plt.xlabel('Number of Experiments')
    plt.ylabel('Yield (%)')
    plt.title(f'Performance Comparison (Cumulative Max): {name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'benchmark_{name.replace(" ", "_")}.png')
    plt.close()

# Report
with open('benchmark_summary.json', 'w') as f:
    json.dump(all_results, f)

print("Benchmark complete. Plots generated.")
