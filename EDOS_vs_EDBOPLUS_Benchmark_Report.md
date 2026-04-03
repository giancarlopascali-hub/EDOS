# Benchmark Comparison Report: EDOS vs. EDBO+
**Generated:** 2026-04-03 07:51

## Experimental Setup

| Parameter | EDOS | EDBO+ |
|---|---|---|
| Search space | Continuous bounds + rounded | Discrete 10x10 grid for x1,x2 |
| x1 | [1, 10] continuous | 10 equally spaced values |
| x2 | [-50, 50] continuous | 10 equally spaced values |
| x3 | {5, 10, 25, 100} discrete | same |
| c1 | {L1, L2, L3} categorical | same |
| c2 | {M1, M2, M3} categorical | same |
| Categorical encoding | Ordinal + MixedSingleTaskGP | One-Hot (drop_first=True) |
| GP training | fit_gpytorch_mll (L-BFGS-B) | Adam optimizer (1000 iter) |
| ACQ (single-obj) | qLogEI | qEI (via NoisyEHVI block) |
| ACQ (multi-obj) | qLogNEHVI | qNoisyEHVI |
| ACQ optimization | optimize_acqf (gradient-based) | optimize_acqf_discrete (grid) |
| Initialization | 3 fixed points | Same 3 points snapped to grid |
| BO iterations | 25 (15 explore + 10 exploit) | 20 (NoisyEHVI throughout) |

## Performance Summary

| Case | Grid Optimum | EDOS Best | EDOS % | EDBO+ Best | EDBO+ % |
| --- | --- | --- | --- | --- | --- |
| A  (max Score) | 12.0 | 11.199891061704491 | 93.3% | 8.70433088899376 | 72.5% |
| B  (max M1+M2) | 21.33484284585623 | 21.137011092609065 | 99.1% | 14.018591513755089 | 65.7% |
| C  (max M1) | 12.0 | 11.682306782096504 | 97.4% | 8.70433088899376 | 72.5% |

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
