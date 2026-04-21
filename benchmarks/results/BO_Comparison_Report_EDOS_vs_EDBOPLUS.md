# Bayesian Optimization Comparative Report: EDOS vs. EDBO+

This report provides a technical comparison between the **BO module of EDOS** and **EDBO+** (Doyle Lab), focusing on implementation details in the BoTorch/GPyTorch framework.

## Executive Summary

| Feature | EDOS (Local App) | EDBO+ (Doyle Lab) |
| :--- | :--- | :--- |
| **Primary Domain** | Mixed spaces (Continuous, Discrete, Cat) | Strictly Discrete/Categorical spaces |
| **Search Space** | Continuous bounds with post-hoc rounding | Exhaustive discrete grid ("Reaction Scope") |
| **Categorical Encoding** | Ordinal (Integer) + `MixedSingleTaskGP` | One-Hot Encoding (OHE) |
| **Acquisition Optimization** | Gradient-based (`optimize_acqf`) | Exhaustive evaluation (`optimize_acqf_discrete`) |
| **BO Engine** | BoTorch (Modern `qLog`-based) | BoTorch (Custom training loop) |
| **Constraint Handling** | Dynamic torch-compiled expressions | Filtered from pre-generated grid |

---

## 1. Categorical Parameter Handling

The most significant architectural difference lies in how categorical variables are interpreted by the Gaussian Process (GP).

### **EDBO+: One-Hot Encoding (OHE)**
EDBO+ treats every categorical level as a separate binary dimension (using `pd.get_dummies(drop_first=True)`).
*   **Pros**: Explicitly models the distance between any two categories as equal; very robust for small sets of categories.
*   **Cons**: Leads to "Dimensionality Explosion." If you have 5 solvents and 5 ligands, you add 8 binary dimensions. The GP becomes harder to train as the feature space grows.

### **EDOS: Ordinal Encoding + `MixedSingleTaskGP`**
EDOS maps categories to integers `[0, 1, 2, ...]` and utilizes BoTorch's `MixedSingleTaskGP`.
*   **Pros**: Maintains a low-dimensional input space. `MixedSingleTaskGP` handles categories using specialized kernels (often kernel-level masking or distance metrics) rather than expanding the feature matrix.
*   **Cons**: Requires careful handling of the "continuous relaxation" during optimization.

---

## 2. BoTorch Algorithm & Model Settings

| Setting | EDOS | EDBO+ |
| :--- | :--- | :--- |
| **Model** | `MixedSingleTaskGP` / `ModelListGP` | `SingleTaskGP` / `ModelListGP` |
| **Kernel** | `MaternKernel(nu=2.5)` + `ScaleKernel` | `MaternKernel(nu=2.5)` with ARD |
| **Priors** | GPyTorch Defaults | Strict Gamma Priors (Lengthscale, Outputscale, Noise) |
| **Training** | `fit_gpytorch_mll` (L-BFGS-B) | Custom Adam Optimizer (1000 iterations) |
| **Acquisition** | `qLogEI`, `qUCB`, `qLogNEHVI` | `qEI`, `qNEHVI` |

**Differentiator:** EDBO+ uses a fixed **Adam optimizer** for GP training, which is more common in deep learning but less standard for GPs than the L-BFGS-B approach used in EDOS. EDBO+ also uses very specific Gamma priors to bias the model towards smoother functions, whereas EDOS relies more on the data to inform hyperparameter fitting via standard BoTorch utilities.

---

## 3. Exploration/Exploitation Balance

### **EDBO+ (Standard Acquisition)**
The balance is strictly governed by the math of EI or NEHVI. EDBO+ does not expose a "jitter" or "exploration" coefficient for EI in its typical UI, relying on the acquisition function's inherent property to sample areas of high uncertainty.

### **EDOS (Dynamic & Safety-First)**
*   **UCB Beta Control**: EDOS exposes an "Exploration" slider that maps directly to the $\beta$ parameter: `beta = (1.0 - exploration) * 10.0 + 0.1`. High exploration = high $\beta$.
*   **Avoid Re-evaluation Logic**: EDOS adds a unique layer of "Exploitation Safety." If the optimizer proposes a point already in the dataset, EDOS applies a **micro-jitter** (10% span) to continuous variables or a **categorical rotation** to force exploration.

---

## 4. Finding Proposed Conditions (Optimization Strategy)

This is the core "philosophical" difference between the two tools.

### **EDBO+: Grid-Search (Exhaustive)**
EDBO+ defines a "Scope" (all possible combinations of your discrete/categorical variables). It then computes the Acquisition Value for **every single point** in that scope and picks the best ones.
*   **Finding Strategy**: Deterministic and global. It cannot "miss" the best discrete point because it checks them all.
*   **Scale Limit**: If your scope has billions of combinations, EDBO+ becomes extremely slow or runs out of memory.

### **EDOS: Continuous Optimization + Rounding**
EDOS treats the world as continuous. It uses gradient-based optimization (`optimize_acqf`) to find the mathematical peak of the acquisition function.
*   **Finding Strategy**: It identifies the "ideal" theoretical point, then **moves it to the nearest valid point** (round to discrete, map to category).
*   **Scale Limit**: Virtually unlimited. It handles millions of possibilities easily because it only optimizes the surface, not the points.
*   **Deterministic Finding**: EDOS uses `SobolQMCNormalSampler` for quasi-Monte Carlo sampling during ACQ optimization, making it more stable, but requires restarts to avoid local optima.

---

## 5. Conclusion & Recommendations

*   **When to use EDBO+**: When your problem is entirely combinatorial (e.g., picking 1 of 12 catalysts, 1 of 8 bases, and 1 of 4 temperatures) and the total number of combinations is $< 100,000$.
*   **When to use EDOS**: When you have continuous variables (e.g., exact concentrations, precise time, or flow rates) mixed with categoricals, or when the combinatorial space is too large for exhaustive search.

---
*Report generated on April 2, 2026.*
