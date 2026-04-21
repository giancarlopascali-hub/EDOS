# Enhanced BO Benchmarking Validation Report
**Completed on: 2026-04-02 13:48:29**

## 1. Ground Truth Functions (Detailed Expressions)

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
| **A (EI)** | Exploration | 9.78 | 11.20 | 0.80 |
| | Exploitation | 10.69 | 11.20 | 0.80 |
| **A (LCB)** | Exploration | 10.14 | 11.66 | 0.34 |
| | Exploitation | 9.82 | 11.70 | 0.30 |
| **B** | Exploration | 9.58 | 10.94 | 0.56 |
| | Exploitation | 10.88 | 11.48 | 0.02 |
| **C** | Exploration | 8.78 | 11.44 | 0.56 |
| | Exploitation | 8.68 | 11.68 | 0.32 |
## 3. Visual Analysis

![Convergence Plots](convergence_plots.png)
