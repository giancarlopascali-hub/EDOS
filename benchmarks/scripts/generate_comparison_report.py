import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_clean(filepath, objective_cols):
    df = pd.read_csv(filepath)
    # Ensure objectives are numeric
    for col in objective_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Files
edos_files = {
    'A': 'plot_data_A_EI.csv',
    'B': 'plot_data_B.csv',
    'C': 'plot_data_C.csv'
}
edbo_files = {
    'A': 'plot_data_edbo_A.csv',
    'B': 'plot_data_edbo_B.csv',
    'C': 'plot_data_edbo_C.csv'
}

# Plot setup
fig, axes = plt.subplots(3, 1, figsize=(12, 18))
plt.subplots_adjust(hspace=0.4)

# CASE A: SINGLE OBJECTIVE
ax = axes[0]
try:
    df_edos_a = load_and_clean(edos_files['A'], ['Score'])
    df_edbo_a = load_and_clean(edbo_files['A'], ['Score'])
    
    # We plot raw points
    ax.scatter(range(len(df_edos_a)), df_edos_a['Score'], label='EDOS (Raw Points)', color='blue', alpha=0.5, s=20)
    ax.plot(range(len(df_edos_a)), df_edos_a['Score'].cummax(), color='blue', linestyle='--', label='EDOS (Cumulative Max)')
    
    ax.scatter(range(len(df_edbo_a)), df_edbo_a['Score'], label='EDBO+ (Raw Points)', color='red', alpha=0.5, s=20)
    ax.plot(range(len(df_edbo_a)), df_edbo_a['Score'].cummax(), color='red', linestyle='--', label='EDBO+ (Cumulative Max)')
    
    ax.set_title("Benchmark A: Single-Objective Optimization", fontsize=14)
    ax.set_xlabel("Iteration (Initial 3 + 20 Proposals)")
    ax.set_ylabel("Score")
    ax.legend()
except Exception as e:
    ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')

# CASE B: COOPERATING MULTI-OBJECTIVE
ax = axes[1]
try:
    df_edos_b = load_and_clean(edos_files['B'], ['Metric1', 'Metric2'])
    df_edbo_b = load_and_clean(edbo_files['B'], ['Metric1', 'Metric2'])
    
    # Joint score: Metric 1 + Metric 2
    joint_edos = df_edos_b['Metric1'] + df_edos_b['Metric2']
    joint_edbo = df_edbo_b['Metric1'] + df_edbo_b['Metric2']
    
    ax.scatter(range(len(joint_edos)), joint_edos, label='EDOS (Raw Points)', color='green', alpha=0.5, s=20)
    ax.plot(range(len(joint_edos)), joint_edos.cummax(), color='green', linestyle='--', label='EDOS (Cumulative Max)')
    
    ax.scatter(range(len(joint_edbo)), joint_edbo, label='EDBO+ (Raw Points)', color='orange', alpha=0.5, s=20)
    ax.plot(range(len(joint_edbo)), joint_edbo.cummax(), color='orange', linestyle='--', label='EDBO+ (Cumulative Max)')
    
    ax.set_title("Benchmark B: Cooperating Objectives (Joint Score)", fontsize=14)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric 1 + Metric 2")
    ax.legend()
except Exception as e:
    ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')

# CASE C: OPPOSING MULTI-OBJECTIVE
ax = axes[2]
try:
    df_edos_c = load_and_clean(edos_files['C'], ['Metric1', 'Metric2'])
    df_edbo_c = load_and_clean(edbo_files['C'], ['Metric1', 'Metric2'])
    
    # For opposing, we use hypervolume as a proxy or just show Metric 1
    # Let's show Metric 1 vs Iteration
    ax.scatter(range(len(df_edos_c)), df_edos_c['Metric1'], label='EDOS Metric1', color='purple', alpha=0.4, s=20)
    ax.scatter(range(len(df_edbo_c)), df_edbo_c['Metric1'], label='EDBO+ Metric1', color='brown', alpha=0.4, s=20)
    
    ax.set_title("Benchmark C: Opposing Objectives (Exploration of Metric 1)", fontsize=14)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric 1 Value")
    ax.legend()
except Exception as e:
    ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')

plt.savefig("benchmark_comparison_plots.png")
print("Comparison plots generated: benchmark_comparison_plots.png")

# Generate Summary Report Data
results = []
if 'df_edos_a' in locals() and 'df_edbo_a' in locals():
    results.append(['A (Max Score)', df_edos_a['Score'].max(), df_edbo_a['Score'].max()])
if 'joint_edos' in locals() and 'joint_edbo' in locals():
    results.append(['B (Max Joint)', joint_edos.max(), joint_edbo.max()])
if 'df_edos_c' in locals() and 'df_edbo_c' in locals():
    results.append(['C (Max Metric1)', df_edos_c['Metric1'].max(), df_edbo_c['Metric1'].max()])

report_df = pd.DataFrame(results, columns=['Case', 'EDOS (Continuous)', 'EDBO+ (Discrete/Grid)'])

# Save Report as Markdown
with open("comparison_report.md", "w") as f:
    f.write("# BO Benchmark Comparison: EDOS vs. EDBO+\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## 1. Summary Metrics\n\n")
    f.write(report_df.to_markdown(index=False))
    f.write("\n\n## 2. Convergence Analysis\n\n")
    f.write("![Comparison Plots](benchmark_comparison_plots.png)\n\n")
    f.write("## 3. Findings\n\n")
    f.write("- **Exploration Pattern**: Observe how EDOS (blue/green) shows more variance in later stages due to its continuous exploration logic vs. EDBO+'s grid updates.\n")
    f.write("- **Optimization Precision**: Note if EDBO+ hits the same peaks despite the 10x10 discretization.\n")

print("Comparison report generated: comparison_report.md")
