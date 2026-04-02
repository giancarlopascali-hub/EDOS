import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Ground Truth Definitions
def ground_truth_A(x1, x2, x3, c1, c2):
    cont = 10.0 * np.exp(-0.5 * ((x1 - 5.0)/1.5)**2 - 0.5 * ((x2 - 0.0)/20.0)**2)
    d_map = {5: 0.5, 10: 1.0, 25: 0.7, 100: 0.2}
    d_bonus = d_map.get(float(x3), 0.0)
    c_bonus = 0.0
    if c1 == 'L2': c_bonus += 0.5
    if c2 == 'M3': c_bonus += 0.5
    return cont + d_bonus + c_bonus

def ground_truth_B(x1, x2, x3, c1, c2):
    obj1 = ground_truth_A(x1, x2, x3, c1, c2)
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
    cont2 = 10.0 * np.exp(-0.5 * ((x1 - 1.0)/2.0)**2 - 0.5 * ((x2 + 40.0)/10.0)**2)
    d_map = {5: 1.0, 10: 0.2, 25: 0.1, 100: 0.0}
    d_bonus = d_map.get(float(x3), 0.0)
    c_bonus = 0.0
    if c1 == 'L1': c_bonus += 0.5
    if c2 == 'M1': c_bonus += 0.5
    obj2 = cont2 + d_bonus + c_bonus
    return obj1, obj2

# Load Data
df_ei = pd.read_csv("plot_data_A_EI.csv")
df_lcb = pd.read_csv("plot_data_A_LCB.csv")
df_b = pd.read_csv("plot_data_B.csv")
df_c = pd.read_csv("plot_data_C.csv")

# Add Phase Info (Init + 25 iterations)
def add_phase_label(df):
    df['Phase'] = 'Init'
    df.iloc[0:3, df.columns.get_loc('Phase')] = 'Init'
    df.iloc[3:18, df.columns.get_loc('Phase')] = 'Exploration'
    df.iloc[18:, df.columns.get_loc('Phase')] = 'Exploitation'
    return df

df_ei = add_phase_label(df_ei)
df_lcb = add_phase_label(df_lcb)
df_b = add_phase_label(df_b)
df_c = add_phase_label(df_c)

def calculate_optima():
    x3_pool = [5, 10, 25, 100]
    c1_pool = ['L1', 'L2', 'L3']
    c2_pool = ['M1', 'M2', 'M3']
    results = {}
    best_A = -np.inf
    best_params_A = None
    for x3 in x3_pool:
        for c1 in c1_pool:
            for c2 in c2_pool:
                val = ground_truth_A(5.0, 0.0, x3, c1, c2)
                if val > best_A:
                    best_A = val
                    best_params_A = {'x1': 5, 'x2': 0, 'x3': x3, 'c1': c1, 'c2': c2}
    results['A'] = (best_A, best_params_A)
    best_B = -np.inf
    best_params_B = None
    for x3 in x3_pool:
        for c1 in c1_pool:
            for c2 in c2_pool:
                v1, v2 = ground_truth_B(5.2, 2.5, x3, c1, c2)
                if (v1 + v2) > best_B:
                    best_B = v1 + v2
                    best_params_B = {'x1': 5.2, 'x2': 2.5, 'x3': x3, 'c1': c1, 'c2': c2, 'v1': v1, 'v2': v2}
    results['B'] = (best_B, best_params_B)
    p1 = ground_truth_C(5, 0, 10, 'L2', 'M3') # Opt for Metric1
    p2 = ground_truth_C(1, -40, 5, 'L1', 'M1') # Opt for Metric2
    results['C_ext'] = (p1, p2)
    pareto_x, pareto_y = [], []
    for t in np.linspace(0.1, 0.9, 100):
        x1_t = 1 + (5 - 1) * t
        x2_t = -40 + (0 - (-40)) * t
        # Calculate a point that is likely Pareto efficient
        # We use a compromise bonus
        v1, v2 = ground_truth_C(x1_t, x2_t, 25, 'L1', 'M1')
        pareto_x.append(v1)
        pareto_y.append(v2)
    results['C_pareto'] = (np.array(pareto_x), np.array(pareto_y))
    return results

optima_data = calculate_optima()

# Enhanced Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Function A
ax0 = axes[0, 0]
ax0.scatter(df_ei.index, df_ei['Score'], label='EI Raw Trials', color='blue', alpha=0.3)
ax0.plot(df_ei.index, df_ei['Score'].cummax(), label='EI Best', color='blue')
ax0.scatter(df_lcb.index, df_lcb['Score'], label='LCB Raw Trials', color='orange', alpha=0.3, marker='s')
ax0.plot(df_lcb.index, df_lcb['Score'].cummax(), label='LCB Best', color='orange')
ax0.axhline(y=optima_data['A'][0], color='green', linestyle='--', label='Optimum')
ax0.axvline(x=17.5, color='red', linestyle=':')
ax0.set_title("Function A (1 Objective)")
ax0.legend()

# Function B
ax1 = axes[0, 1]
b_sum = df_b['Metric1'] + df_b['Metric2']
ax1.scatter(df_b.index, b_sum, label='Raw Trials (Sum)', color='green', alpha=0.3)
ax1.plot(df_b.index, b_sum.cummax(), label='Best (Sum)', color='green')
ax1.axhline(y=optima_data['B'][0], color='darkgreen', linestyle='--', label='Optimum')
ax1.axvline(x=17.5, color='red', linestyle=':')
ax1.set_title("Function B (Cooperating Multi-Obj)")
ax1.legend()

# Function C
ax2 = axes[1, 0]
m1 = df_c['Metric1']
m2 = df_c['Metric2']
ax2.scatter(m1[df_c['Phase']=='Exploration'], m2[df_c['Phase']=='Exploration'], c='cyan', edgecolors='k', label='Exploration Phase')
ax2.scatter(m1[df_c['Phase']=='Exploitation'], m2[df_c['Phase']=='Exploitation'], c='magenta', edgecolors='k', label='Exploitation Phase')
px, py = optima_data['C_pareto']
sort_idx = np.argsort(px)
ax2.plot(px[sort_idx], py[sort_idx], color='red', linestyle=':', label='Approx. Pareto Front')
ax2.set_title("Function C (Opposing Multi-Obj)")
ax2.set_xlabel("Metric 1")
ax2.set_ylabel("Metric 2")
ax2.legend()

ax3 = axes[1, 1]
ax3.axis('off')
ax3.text(0.1, 0.5, "Enhanced Performance Analysis\nPhase Contrast Visualization", fontsize=14, fontweight='bold')

plt.savefig("convergence_plots.png")

# Metrics Calculation
def calc_metrics(df, col, opt):
    m = {}
    for phase in ['Exploration', 'Exploitation']:
        sub = df[df['Phase'] == phase]
        if sub.empty: m[phase] = ("N/A", "N/A", "N/A")
        else:
            avg = sub[col].mean()
            mx = sub[col].max()
            dist = opt - mx
            m[phase] = (f"{avg:.2f}", f"{mx:.2f}", f"{dist:.2f}")
    return m

m_ei = calc_metrics(df_ei, 'Score', optima_data['A'][0])
m_lcb = calc_metrics(df_lcb, 'Score', optima_data['A'][0])
m_b = calc_metrics(df_b, 'Metric1', 11.5) # Using Metric 1 approx opt for B
m_c = calc_metrics(df_c, 'Metric1', 12.0) # Using Metric 1 opt for C

# Report Writing
header = f"# Enhanced BO Benchmarking Validation Report\n**Completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**\n\n"

body = r"""## 1. Ground Truth Functions (Detailed Expressions)

### Function A (Single-Objective)
$$f_A(x_1, x_2, x_3, c_1, c_2) = 10 \cdot \exp\left(-0.5 \cdot \left(\frac{x_1 - 5.0}{1.5}\right)^2 - 0.5 \cdot \left(\frac{x_2}{20.0}\right)^2\right) + D_{bonus}(x_3) + C_{bonus}(c_1, c_2)$$
*   **Discrete Bonus ($x_3$):** {5: 0.5, 10: 1.0, 25: 0.7, 100: 0.2}
*   **Categorical Bonus:** {L2: 0.5, M3: 0.5}
*   **Optimum:** 12.00 at (5, 0, 10, L2, M3)

### Function B (Cooperating Multi-Objective)
*   **Metric 1:** $f_A(x_1, x_2, x_3, c_1, c_2)$
*   **Metric 2:** $8 \cdot \exp\left(-0.5 \cdot \left(\frac{x_1 - 5.5}{2.0}\right)^2 - 0.5 \cdot \left(\frac{x_2 - 5.0}{15.0}\right)^2\right) + D_{bonus2}(x_3) + C_{bonus2}(c_1, c_2)$
*   **Discrete Bonus 2 ($x_3$):** {5: 0.2, 10: 1.0, 25: 0.8, 100: 0.4}
*   **Categorical Bonus 2:** {L2: 0.4, M3: 0.6}

### Function C (Opposing Multi-Objective)
*   **Metric 1:** $f_A(x_1, x_2, x_3, c_1, c_2)$ (Max at (5,0))
*   **Metric 2:** $10 \cdot \exp\left(-0.5 \cdot \left(\frac{x_1 - 1.0}{2.0}\right)^2 - 0.5 \cdot \left(\frac{x_2 + 40.0}{10.0}\right)^2\right) + D_{bonus3}(x_3) + C_{bonus3}(c_1, c_2)$ (Max at (1,-40))
*   **Discrete Bonus 3 ($x_3$):** {5: 1.0, 10: 0.2, 25: 0.1, 100: 0.0}
*   **Categorical Bonus 3:** {L1: 0.5, M1: 0.5}

## 2. Performance Metrics Comparison

| Scenario | Phase | Mean Score | Max Score | Dist. from Opt |
| :--- | :--- | :---: | :---: | :---: |
"""
table = []
table.append(f"| **A (EI)** | Exploration | {m_ei['Exploration'][0]} | {m_ei['Exploration'][1]} | {m_ei['Exploration'][2]} |")
table.append(f"| | Exploitation | {m_ei['Exploitation'][0]} | {m_ei['Exploitation'][1]} | {m_ei['Exploitation'][2]} |")
table.append(f"| **A (LCB)** | Exploration | {m_lcb['Exploration'][0]} | {m_lcb['Exploration'][1]} | {m_lcb['Exploration'][2]} |")
table.append(f"| | Exploitation | {m_lcb['Exploitation'][0]} | {m_lcb['Exploitation'][1]} | {m_lcb['Exploitation'][2]} |")
table.append(f"| **B** | Exploration | {m_b['Exploration'][0]} | {m_b['Exploration'][1]} | {m_b['Exploration'][2]} |")
table.append(f"| | Exploitation | {m_b['Exploitation'][0]} | {m_b['Exploitation'][1]} | {m_b['Exploitation'][2]} |")
table.append(f"| **C** | Exploration | {m_c['Exploration'][0]} | {m_c['Exploration'][1]} | {m_c['Exploration'][2]} |")
table.append(f"| | Exploitation | {m_c['Exploitation'][0]} | {m_c['Exploitation'][1]} | {m_c['Exploitation'][2]} |")

visual = "\n## 3. Visual Analysis\n\n![Convergence Plots](convergence_plots.png)\n"

with open("benchmark_report.md", "w") as f:
    f.write(header + body + "\n".join(table) + visual)
print("Enhanced report generated successfully.")
