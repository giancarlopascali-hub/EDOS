# EDOS

**EDOS** (Experimental Design and Optimization System) is a powerful, web-based platform designed for researchers and engineers to perform **Design of Experiments (DoE)**, **Bayesian Optimization (BO)** and **Statistical Analysis (SA)** with ease.

## Key Features

- **Advanced DoE Module**: Generate Box-Behnken, Central Composite, Fractional Factorial and Definitive Screening designs.
- **Scientific Quality Metrics**: Real-time calculation of Orthogonality, D-Efficiency, Resolution, and Curvature detection with a visual "Quality Dashboard".
- **Bayesian Optimizer**: Suggest the best tests to perform next and improve your peformance, visualizing objective trends and Pareto front. It supports categorical parameters and multi-objective optimization.
- **Statistical Analysis**: Perform basic analysis on experimental data, including feature importance, correlation, and success-based optimization.
- **Standalone Capability**: Bundled into a Windows executable for easy deployment.

## Installation and Running (Windows Standalone)

1. **Download**: You can download the latest standalone version from the [Releases](https://github.com/giancarlopascali-hub/EDOS/releases) page.
2. **Extract**: Right-click the `EDOS_v1.0_Windows.zip` file and select "Extract All...".
3. **Run**: 
   - Open the extracted folder.
   - **Note**: The application lives entirely within its folder structure; do not move the `.exe` file out of the `EDOS_v1.0` folder.
   - Double-click **`EDOS_v1.0.exe`**.
4. **Access**: The application will automatically start a local server. If your browser doesn't open automatically, go to `http://127.0.0.1:5000`.
5. **Shutdown**: To close the application properly, press the **"Shutdown"** button within the web interface before closing the window.


## Technical Stack


- **Backend**: Python (Flask, BoTorch, GPyTorch, Scikit-Learn, pyDOE2)
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (ES6+)
- **Packaging**: PyInstaller

## Developer
Giancarlo Pascali
UNSW, Sydney

---
*Created with the help of Antigravity AI.*
