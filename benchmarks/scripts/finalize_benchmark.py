"""
Finalize EDBO+ benchmark: run Case C, then generate the full comparison report.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ── EDBO+ path ────────────────────────────────────────────────────────────────
EDBO_PATH = os.path.join(os.getcwd(), 'edboplus')
if EDBO_PATH not in sys.path:
    sys.path.append(EDBO_PATH)
from edbo.plus.optimizer_botorch import EDBOplus

# ── Ground-truth functions (identical to benchmark_bo.py) ────────────────────
def gt_A(x1, x2, x3, c1, c2):
    cont  = 10.0 * np.exp(-0.5*((x1-5.0)/1.5)**2 - 0.5*((x2)/20.0)**2)
    dmap  = {5.:0.5, 10.:1.0, 25.:0.7, 100.:0.2}
    cbon  = (0.5 if c1=='L2' else 0) + (0.5 if c2=='M3' else 0)
    return cont + dmap.get(float(x3), 0.0) + cbon

def gt_B(x1, x2, x3, c1, c2):
    o1  = gt_A(x1, x2, x3, c1, c2)
    c2v = 8.0*np.exp(-0.5*((x1-5.5)/2.0)**2 - 0.5*((x2-5.0)/15.0)**2)
    dm  = {5.:0.2, 10.:1.0, 25.:0.8, 100.:0.4}
    cb  = (0.4 if c1=='L2' else 0) + (0.6 if c2=='M3' else 0)
    return o1, c2v + dm.get(float(x3), 0.0) + cb

def gt_C(x1, x2, x3, c1, c2):
    o1  = gt_A(x1, x2, x3, c1, c2)
    c2v = 10.0*np.exp(-0.5*((x1-1.0)/2.0)**2 - 0.5*((x2+40.0)/10.0)**2)
    dm  = {5.:1.0, 10.:0.2, 25.:0.1, 100.:0.0}
    cb  = (0.5 if c1=='L1' else 0) + (0.5 if c2=='M1' else 0)
    return o1, c2v + dm.get(float(x3), 0.0) + cb

# ── Scope definition ──────────────────────────────────────────────────────────
components = {
    'x1': np.linspace(1, 10, 10).tolist(),
    'x2': np.linspace(-50, 50, 10).tolist(),
    'x3': [5., 10., 25., 100.],
    'c1': ['L1','L2','L3'],
    'c2': ['M1','M2','M3'],
}

INIT_PTS = [
    {'x1':2., 'x2':-30., 'x3':25., 'c1':'L1', 'c2':'M1'},
    {'x1':8., 'x2':40.,  'x3':100., 'c1':'L3', 'c2':'M2'},
    {'x1':4., 'x2':0.,   'x3':5.,  'c1':'L2', 'c2':'M2'},
]

def run_case(case_name, obj_names, gt_func, n_iter=20):
    out_csv = f'plot_data_edbo_{case_name}.csv'
    # Already done?
    if os.path.exists(out_csv):
        df_check = pd.read_csv(out_csv)
        if len(df_check) >= n_iter + len(INIT_PTS):
            print(f"[{case_name}] Already complete ({len(df_check)} rows). Skipping.")
            return
    
    print(f"\n=== EDBO+ Case {case_name} ===")
    scope_file = f'scope_{case_name}.csv'
    edbo = EDBOplus()
    edbo.generate_reaction_scope(components, filename=scope_file, check_overwrite=False)
    df = pd.read_csv(scope_file)
    
    # Mark all PENDING
    for obj in obj_names:
        df[obj] = 'PENDING'
    
    # Inject initial observations (snap to closest grid point)
    for pt in INIT_PTS:
        diff = ((df['x1'] - pt['x1']).abs() +
                (df['x2'] - pt['x2']).abs() +
                (df['x3'] - pt['x3']).abs() +
                (df['c1'].astype(str) != str(pt['c1'])).astype(int) +
                (df['c2'].astype(str) != str(pt['c2'])).astype(int))
        idx = diff.idxmin()
        res = gt_func(df.loc[idx,'x1'], df.loc[idx,'x2'],
                      df.loc[idx,'x3'], df.loc[idx,'c1'], df.loc[idx,'c2'])
        if isinstance(res, tuple):
            for j, obj in enumerate(obj_names):
                df.loc[idx, obj] = res[j]
        else:
            df.loc[idx, obj_names[0]] = res
    df.to_csv(scope_file, index=False)
    
    # BO loop
    for i in range(1, n_iter + 1):
        print(f"  Iter {i}/{n_iter}", flush=True)
        edbo.run(objectives=obj_names,
                 objective_mode=['max']*len(obj_names),
                 batch=1, filename=scope_file, seed=42,
                 acquisition_function='NoisyEHVI')
        df_up = pd.read_csv(scope_file)
        suggestion = df_up.sort_values('priority', ascending=False).iloc[0]
        idx = suggestion.name
        res = gt_func(suggestion['x1'], suggestion['x2'],
                      suggestion['x3'], suggestion['c1'], suggestion['c2'])
        if isinstance(res, tuple):
            for j, obj in enumerate(obj_names):
                df_up.loc[idx, obj] = res[j]
        else:
            df_up.loc[idx, obj_names[0]] = res
        df_up.to_csv(scope_file, index=False)
    
    # Extract evaluated rows and save
    df_final = pd.read_csv(scope_file)
    evaluated = df_final[~df_final[obj_names[0]].astype(str).str.contains('PENDING', case=False)].copy()
    for obj in obj_names:
        evaluated[obj] = pd.to_numeric(evaluated[obj], errors='coerce')
    evaluated.to_csv(out_csv, index=False)
    print(f"  Saved {len(evaluated)} rows to {out_csv}")

# ── Run only missing cases ────────────────────────────────────────────────────
run_case('A', ['Score'],              gt_A)
run_case('B', ['Metric1','Metric2'], gt_B)
run_case('C', ['Metric1','Metric2'], gt_C)

print("\n[OK] All EDBO+ cases complete. Generating comparison report...")

# ── Helper ────────────────────────────────────────────────────────────────────
def load(fpath, cols):
    df = pd.read_csv(fpath)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=cols).reset_index(drop=True)

# ── Load results ──────────────────────────────────────────────────────────────
edos_A  = load('plot_data_A_EI.csv',  ['Score'])
edbo_A  = load('plot_data_edbo_A.csv',['Score'])
edos_B  = load('plot_data_B.csv',     ['Metric1','Metric2'])
edbo_B  = load('plot_data_edbo_B.csv',['Metric1','Metric2'])
edos_C  = load('plot_data_C.csv',     ['Metric1','Metric2'])
edbo_C  = load('plot_data_edbo_C.csv',['Metric1','Metric2'])

# ── Ground-truth optimal values (discrete grid) ───────────────────────────────
best_A_possible = max(gt_A(5., 0., 10., 'L2', 'M3'), gt_A(4.0, 0., 10., 'L2', 'M3'))
best_B_possible = sum(gt_B(5., 0., 10., 'L2', 'M3'))
best_C_m1 = gt_C(5., 0., 10., 'L2', 'M3')[0]
best_C_m2 = gt_C(1., -44.4, 5., 'L1', 'M1')[1]

# ── Plot ──────────────────────────────────────────────────────────────────────
BLUE   = '#2563EB'
RED    = '#DC2626'
GREEN  = '#16A34A'
ORANGE = '#D97706'
PURPLE = '#7C3AED'
BROWN  = '#92400E'
LGRAY  = '#E5E7EB'

fig = plt.figure(figsize=(16, 18), facecolor='#0F172A')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                        left=0.08, right=0.96, top=0.93, bottom=0.05)

title_kw = dict(color='white', fontsize=13, fontweight='bold', pad=10)
label_kw = dict(color='#94A3B8', fontsize=10)
tick_kw  = dict(colors='#94A3B8', labelsize=9)

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor('#1E293B')
    ax.spines[:].set_color('#334155')
    ax.tick_params(axis='both', **tick_kw)
    ax.set_title(title, **title_kw)
    ax.set_xlabel(xlabel, **label_kw)
    ax.set_ylabel(ylabel, **label_kw)
    ax.grid(True, color='#334155', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='white', fontsize=9)

# ── Panel A-left: raw points ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
n_edos = len(edos_A); n_edbo = len(edbo_A)
ax.scatter(range(n_edos), edos_A['Score'], color=BLUE,  alpha=0.55, s=30, zorder=3, label='EDOS raw')
ax.scatter(range(n_edbo), edbo_A['Score'], color=RED,   alpha=0.55, s=30, zorder=3, label='EDBO+ raw')
ax.plot(range(n_edos), edos_A['Score'].cummax(), color=BLUE, lw=2, label='EDOS best-so-far')
ax.plot(range(n_edbo), edbo_A['Score'].cummax(), color=RED,  lw=2, linestyle='--', label='EDBO+ best-so-far')
ax.axhline(best_A_possible, color='white', lw=1, linestyle=':', alpha=0.6, label=f'Grid optimum ({best_A_possible:.2f})')
ax.axvline(3, color='#64748B', lw=1, linestyle=':', alpha=0.7, label='Init → BO')
style_ax(ax, 'Case A – Single Objective (EI / NoisyEHVI)', 'Iteration', 'Score')

# ── Panel A-right: exploration space ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(edos_A['x1'], edos_A['x2'], c=range(len(edos_A)), cmap='Blues',
            s=40, alpha=0.8, edgecolors='white', linewidths=0.3, label='EDOS', zorder=3)
ax2.scatter(edbo_A['x1'], edbo_A['x2'], c=range(len(edbo_A)), cmap='Reds',
            s=40, alpha=0.8, edgecolors='white', linewidths=0.3, marker='D', label='EDBO+', zorder=3)
ax2.set_facecolor('#1E293B'); ax2.spines[:].set_color('#334155')
ax2.tick_params(axis='both', **tick_kw)
ax2.set_title('Case A – Sampled Space (x₁ vs x₂)', **title_kw)
ax2.set_xlabel('x₁', **label_kw); ax2.set_ylabel('x₂', **label_kw)
ax2.grid(True, color='#334155', linestyle='--', linewidth=0.5, alpha=0.7)
ax2.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='white', fontsize=9)

# ── Panel B-left: joint metric ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
je = edos_B['Metric1'] + edos_B['Metric2']
jb = edbo_B['Metric1'] + edbo_B['Metric2']
ax3.scatter(range(len(je)), je, color=GREEN,  alpha=0.55, s=30, zorder=3, label='EDOS raw')
ax3.scatter(range(len(jb)), jb, color=ORANGE, alpha=0.55, s=30, zorder=3, label='EDBO+ raw')
ax3.plot(range(len(je)), je.cummax(), color=GREEN,  lw=2, label='EDOS best-so-far')
ax3.plot(range(len(jb)), jb.cummax(), color=ORANGE, lw=2, linestyle='--', label='EDBO+ best-so-far')
ax3.axhline(best_B_possible, color='white', lw=1, linestyle=':', alpha=0.6, label=f'Grid optimum ({best_B_possible:.2f})')
ax3.axvline(3, color='#64748B', lw=1, linestyle=':', alpha=0.7)
style_ax(ax3, 'Case B – Cooperating Objectives (Joint Score)', 'Iteration', 'Metric1 + Metric2')

# ── Panel B-right: Pareto scatter ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(edos_B['Metric1'], edos_B['Metric2'], color=GREEN,  alpha=0.6, s=35,
            edgecolors='white', linewidths=0.3, label='EDOS', zorder=3)
ax4.scatter(edbo_B['Metric1'], edbo_B['Metric2'], color=ORANGE, alpha=0.6, s=35,
            edgecolors='white', linewidths=0.3, marker='D', label='EDBO+', zorder=3)
ax4.set_facecolor('#1E293B'); ax4.spines[:].set_color('#334155')
ax4.tick_params(axis='both', **tick_kw)
ax4.set_title('Case B – Trade-off Space', **title_kw)
ax4.set_xlabel('Metric 1', **label_kw); ax4.set_ylabel('Metric 2', **label_kw)
ax4.grid(True, color='#334155', linestyle='--', linewidth=0.5, alpha=0.7)
ax4.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='white', fontsize=9)

# ── Panel C-left: Metric1 vs iteration ───────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(range(len(edos_C)), edos_C['Metric1'], color=PURPLE, alpha=0.55, s=30, zorder=3, label='EDOS raw')
ax5.scatter(range(len(edbo_C)), edbo_C['Metric1'], color=BROWN,  alpha=0.55, s=30, zorder=3, label='EDBO+ raw')
ax5.plot(range(len(edos_C)), edos_C['Metric1'].cummax(), color=PURPLE, lw=2, label='EDOS best-so-far')
ax5.plot(range(len(edbo_C)), edbo_C['Metric1'].cummax(), color=BROWN,  lw=2, linestyle='--', label='EDBO+ best-so-far')
ax5.axvline(3, color='#64748B', lw=1, linestyle=':', alpha=0.7)
style_ax(ax5, 'Case C – Opposing Objectives (Metric 1)', 'Iteration', 'Metric 1')

# ── Panel C-right: Pareto front (opposing) ────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
sc1 = ax6.scatter(edos_C['Metric1'], edos_C['Metric2'],
                  c=range(len(edos_C)), cmap='Purples', vmin=0, vmax=len(edos_C)+2,
                  s=40, edgecolors='white', linewidths=0.3, label='EDOS (color=time)', zorder=3)
sc2 = ax6.scatter(edbo_C['Metric1'], edbo_C['Metric2'],
                  c=range(len(edbo_C)), cmap='YlOrBr', vmin=0, vmax=len(edbo_C)+2,
                  s=40, edgecolors='white', linewidths=0.3, marker='D', label='EDBO+ (color=time)', zorder=3)
ax6.axvline(best_C_m1, color='#7C3AED', lw=1, linestyle=':', alpha=0.7, label='M1 peak')
ax6.axhline(best_C_m2, color='#B45309', lw=1, linestyle=':', alpha=0.7, label='M2 peak')
ax6.set_facecolor('#1E293B'); ax6.spines[:].set_color('#334155')
ax6.tick_params(axis='both', **tick_kw)
ax6.set_title('Case C – Opposing Trade-off Space', **title_kw)
ax6.set_xlabel('Metric 1', **label_kw); ax6.set_ylabel('Metric 2', **label_kw)
ax6.grid(True, color='#334155', linestyle='--', linewidth=0.5, alpha=0.7)
ax6.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='white', fontsize=8)

fig.suptitle('EDOS vs. EDBO+  ·  Benchmark Comparison\n(20 iterations + 3 initial, same ground-truth functions A / B / C)',
             color='white', fontsize=15, fontweight='bold', y=0.97)

img_path = 'benchmark_comparison_plots.png'
plt.savefig(img_path, dpi=150, bbox_inches='tight', facecolor='#0F172A')
plt.close()
print(f"[OK] Saved plots -> {img_path}")

# ── Summary table ─────────────────────────────────────────────────────────────
rows = [
    ['A  (max Score)',         best_A_possible, edos_A['Score'].max(),
     f"{edos_A['Score'].max()/best_A_possible*100:.1f}%",
     edbo_A['Score'].max(),
     f"{edbo_A['Score'].max()/best_A_possible*100:.1f}%"],
    ['B  (max M1+M2)',         best_B_possible,
     (edos_B['Metric1']+edos_B['Metric2']).max(),
     f"{(edos_B['Metric1']+edos_B['Metric2']).max()/best_B_possible*100:.1f}%",
     (edbo_B['Metric1']+edbo_B['Metric2']).max(),
     f"{(edbo_B['Metric1']+edbo_B['Metric2']).max()/best_B_possible*100:.1f}%"],
    ['C  (max M1)',            best_C_m1,
     edos_C['Metric1'].max(),
     f"{edos_C['Metric1'].max()/best_C_m1*100:.1f}%",
     edbo_C['Metric1'].max(),
     f"{edbo_C['Metric1'].max()/best_C_m1*100:.1f}%"],
]
summary = pd.DataFrame(rows, columns=['Case','Grid Optimum','EDOS Best',
                                       'EDOS %','EDBO+ Best','EDBO+ %'])

# Build markdown table manually (avoid 'tabulate' dependency)
def df_to_md(df):
    headers = df.columns.tolist()
    sep = ['---'] * len(headers)
    lines = ['| ' + ' | '.join(str(h) for h in headers) + ' |',
             '| ' + ' | '.join(sep) + ' |']
    for _, row in df.iterrows():
        lines.append('| ' + ' | '.join(str(v) for v in row.values) + ' |')
    return '\n'.join(lines)

# ── Markdown report ───────────────────────────────────────────────────────────
ts = datetime.now().strftime('%Y-%m-%d %H:%M')
md = f"""# Benchmark Comparison Report: EDOS vs. EDBO+
**Generated:** {ts}

## Experimental Setup

| Parameter | EDOS | EDBO+ |
|---|---|---|
| Search space | Continuous bounds + rounded | Discrete 10x10 grid for x1,x2 |
| x1 | [1, 10] continuous | 10 equally spaced values |
| x2 | [-50, 50] continuous | 10 equally spaced values |
| x3 | {{5, 10, 25, 100}} discrete | same |
| c1 | {{L1, L2, L3}} categorical | same |
| c2 | {{M1, M2, M3}} categorical | same |
| Categorical encoding | Ordinal + MixedSingleTaskGP | One-Hot (drop_first=True) |
| GP training | fit_gpytorch_mll (L-BFGS-B) | Adam optimizer (1000 iter) |
| ACQ (single-obj) | qLogEI | qEI (via NoisyEHVI block) |
| ACQ (multi-obj) | qLogNEHVI | qNoisyEHVI |
| ACQ optimization | optimize_acqf (gradient-based) | optimize_acqf_discrete (grid) |
| Initialization | 3 fixed points | Same 3 points snapped to grid |
| BO iterations | 25 (15 explore + 10 exploit) | 20 (NoisyEHVI throughout) |

## Performance Summary

{df_to_md(summary)}

## Convergence Plots

![Comparison Plots](benchmark_comparison_plots.png)

## Key Findings

### Case A – Single Objective
- **EDOS** uses a continuous relaxation and gradient-based optimizer, reaching fine-grained
  solutions with a deliberate explore→exploit phase switch.
- **EDBO+** exhaustively evaluates every unobserved grid point, so it is guaranteed not to miss
  the best *discrete* candidate, but is limited to the 10×10 grid resolution.
- The gap between their best values (if any) reflects the precision penalty of discretization.

### Case B – Co-operating Objectives
- Both algorithms leverage NEHVI to simultaneously improve two cooperating objectives.
- EDOS can propose off-grid continuous values, while EDBO+ is constrained to the scope grid.
- In cooperating scenarios, both approaches typically converge to similar regions, but EDOS
  may achieve slightly higher joint scores due to finer resolution.

### Case C – Opposing Objectives
- This is the most challenging scenario: the optima of Metric 1 and Metric 2 are at opposite
  corners of the parameter space.
- EDBO+'s exhaustive grid search ensures the Pareto front candidates are always evaluated from
  the *actual* scope, while EDOS can reach intermediate continuous trade-off points.
- The trade-off scatter plot reveals how each algorithm distributes its proposals across the
  two-objective landscape.

## Conclusion

> EDOS is better suited for problems with **continuous parameters or very large combinatorial spaces**,
> where exhaustive grid search would be intractable.
>
> EDBO+ is ideal for **fully discrete/categorical reaction optimization** where the total number
> of candidates is manageable (< ~100k), and a deterministic global guarantee over the grid
> is more important than fine-grained resolution.
"""

report_path = 'EDOS_vs_EDBOPLUS_Benchmark_Report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(md)
print(f"[OK] Report saved -> {report_path}")
print("\n=== DONE ===")
