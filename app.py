import os
import sys

# Detect if running as a PyInstaller bundle and set base path accordingly
if getattr(sys, 'frozen', False):
    # If frozen, sys._MEIPASS is the temp folder where files are extracted
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Robust path detection for Flask resources
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# In some PyInstaller configurations with --onedir, templates might be inside _internal
if not os.path.exists(os.path.join(TEMPLATE_DIR, 'index.html')):
    if os.path.exists(os.path.join(BASE_DIR, '_internal', 'templates')):
        TEMPLATE_DIR = os.path.join(BASE_DIR, '_internal', 'templates')
        STATIC_DIR = os.path.join(BASE_DIR, '_internal', 'static')
import json
import pandas as pd
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, send_file
# BoTorch/GPyTorch imports for Bayesian Optimization
from botorch.models import SingleTaskGP, MixedSingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, qExpectedImprovement
try:
    from botorch.acquisition import qLogExpectedImprovement
except ImportError:
    qLogExpectedImprovement = qExpectedImprovement

from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement, qExpectedHypervolumeImprovement
try:
    from botorch.acquisition.multi_objective.monte_carlo import qLogNoisyExpectedHypervolumeImprovement
except ImportError:
    qLogNoisyExpectedHypervolumeImprovement = qNoisyExpectedHypervolumeImprovement

from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.priors import NormalPrior
from io import BytesIO
import pyDOE2 as doe
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import re

# Set Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("cpu"))
# For DirectML or specific GPU hardware, we might have COMPUTE_DEVICE
COMPUTE_DEVICE = DEVICE 
try:
    import torch_directml
    if torch_directml.is_available():
        DEVICE = torch_directml.device()
        COMPUTE_DEVICE = torch.device("cpu") # Fallback for some ops
except: pass

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# --- Initialization ---
# Ensure necessary directories for templates and static assets exist (only needed in dev mode).
if not getattr(sys, 'frozen', False):
    os.makedirs(os.path.join(BASE_DIR, 'templates'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)

@app.route('/')
def index():
    """Serves the main single-page application."""
    return render_template('index.html')

@app.route('/guide')
def guide():
    """Serves the quick guide page."""
    return render_template('guide.html')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shuts down the Flask server."""
    import threading
    def _stop():
        import time, os, signal
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_stop, daemon=True).start()
    return jsonify({'status': 'shutting down'})

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handles CSV file uploads.
    Parses the CSV into a pandas DataFrame and returns a JSON-friendly 'split' format
    which includes columns and data rows separately for easy frontend rendering.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        df = pd.read_csv(file)
        # Convert to JSON-friendly format
        data = df.to_dict(orient='split')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        features_config = req_data.get('features')
        objectives_config = req_data.get('objectives')
        tweaks = req_data.get('tweaks', {})

        df = pd.DataFrame(data, columns=columns)
        
        # 1. Prepare Features (X)
        train_x_list = []
        bounds = [] 
        categorical_features = []
        
        for f in features_config:
            name = f['name']
            f_range = f['range']
            
            if f['type'] == 'continuous':
                import re
                matches = re.findall(r"[-+]?\d*\.?\d+", f_range)
                low, high = sorted([float(matches[0]), float(matches[1])]) if len(matches) >= 2 else (0.0, 1.0)
                feat_vals = pd.to_numeric(df[name], errors='coerce').fillna(0).values
                train_x_list.append(feat_vals)
                bounds.append([low, high])
            elif f['type'] == 'discrete':
                # Handle cases like [10, 20, 30] or 10, 20, 30
                clean_range = f_range.replace('[', '').replace(']', '')
                values = sorted([float(x.strip()) for x in clean_range.split(',') if x.strip()])
                if not values: values = [0.0]
                feat_vals = pd.to_numeric(df[name], errors='coerce').fillna(values[0]).values
                train_x_list.append(feat_vals)
                bounds.append([values[0], values[-1]])
            elif f['type'] == 'categorical':
                choices = [str(x).strip() for x in f_range.split(',')]
                # Use string mapping to be robust against numeric categorical data
                mapping = {choice: i for i, choice in enumerate(choices)}
                # Ensure data is compared as strings
                encoded = df[name].astype(str).map(mapping).fillna(0).values
                train_x_list.append(encoded)
                bounds.append([0, len(choices) - 1])
                categorical_features.append(len(train_x_list) - 1)

        # Use float32 (Single Precision) for universal GPU support (Intel/AMD/NVIDIA)
        train_x = torch.tensor(np.stack(train_x_list, axis=1), dtype=torch.float32).to(DEVICE)
        bounds_t = torch.tensor(bounds, dtype=torch.float32).T.to(DEVICE)
        
        # 2. Prepare Objectives (Y)
        train_y_list = []
        weights_list = []

        for obj in objectives_config:
            weights_list.append(float(obj.get('importance', 1)))
            if not df.empty:
                vals = pd.to_numeric(df[obj['name']], errors='coerce').fillna(0).values
                if obj['type'] == 'minimize': train_y_list.append(-vals)
                elif obj['type'] == 'maximize': train_y_list.append(vals)
                elif obj['type'] == 'target':
                    target = float(obj.get('target', 0))
                    # Vectorized variance calculation for speed
                    variance = np.var(vals) if np.var(vals) > 0 else 1.0
                    train_y_list.append(-((vals - target)**2) / variance)

        if not df.empty:
            train_y = torch.tensor(np.stack(train_y_list, axis=1), dtype=torch.float32).to(DEVICE)
        else:
            train_y = torch.zeros((0, len(objectives_config)), dtype=torch.float32).to(DEVICE)

        weights_t = torch.tensor(weights_list, dtype=torch.float32).to(DEVICE)
        batch_size = int(tweaks.get('batch_size', 1))
        avoid_reval = tweaks.get('avoid_reval', True)
        acq_type = tweaks.get('acq_type', 'EI')
        exploration = float(tweaks.get('exploration', 0.5))
        
        # 3. Optimization Logic
        if df.empty:
            candidates = draw_sobol_samples(bounds=bounds_t, n=1, q=batch_size).squeeze(0)
        else:
            # HOT START
            kernel_name = tweaks.get('kernel', 'matern52')
            is_noiseless = tweaks.get('noiseless', False)

            def get_kernel(k_name):
                if k_name == 'rbf': return ScaleKernel(RBFKernel()).to(DEVICE)
                if k_name == 'matern32': return ScaleKernel(MaternKernel(nu=1.5)).to(DEVICE)
                if k_name == 'matern12': return ScaleKernel(MaternKernel(nu=0.5)).to(DEVICE)
                return ScaleKernel(MaternKernel(nu=2.5)).to(DEVICE)

            def build_single_model(tx, ty_slice):
                all_dims = list(range(tx.shape[1]))
                cont_dims = [i for i in all_dims if i not in categorical_features]
                
                # Input transform and models moved to COMPUTE_DEVICE (CPU fallback for DML)
                input_tf = Normalize(d=tx.shape[1], bounds=bounds_t.to(COMPUTE_DEVICE)[:, cont_dims], indices=cont_dims).to(COMPUTE_DEVICE) if cont_dims else None
                
                if categorical_features:
                    m = MixedSingleTaskGP(tx.to(COMPUTE_DEVICE), ty_slice.to(COMPUTE_DEVICE), cat_dims=categorical_features, outcome_transform=Standardize(m=1), input_transform=input_tf).to(COMPUTE_DEVICE)
                else:
                    m = SingleTaskGP(tx.to(COMPUTE_DEVICE), ty_slice.to(COMPUTE_DEVICE), covar_module=get_kernel(kernel_name).to(COMPUTE_DEVICE), outcome_transform=Standardize(m=1), input_transform=input_tf).to(COMPUTE_DEVICE)
                
                if is_noiseless: 
                    m.likelihood.noise_covar.register_prior("noise_prior", NormalPrior(1e-4, 1e-5), "noise")
                
                return m.to(COMPUTE_DEVICE)

            if train_y.shape[1] > 1:
                models = [build_single_model(train_x, train_y[:, i:i+1]) for i in range(train_y.shape[1])]
                model = ModelListGP(*models).to(COMPUTE_DEVICE)
                mll = SumMarginalLogLikelihood(model.likelihood, model).to(COMPUTE_DEVICE)
            else:
                model = build_single_model(train_x, train_y).to(COMPUTE_DEVICE)
                mll = ExactMarginalLogLikelihood(model.likelihood, model).to(COMPUTE_DEVICE)
            
            fit_gpytorch_mll(mll)
            
            # SAFE HARDWARE STRATEGY: Only offload to GPU if it's CUDA. 
            # DirectML (PrivateUse1) has kernel stability issues with in-place ops.
            OPT_DEVICE = DEVICE if DEVICE.type == 'cuda' else COMPUTE_DEVICE
            model = model.to(OPT_DEVICE)

            # Parallelization & Intensity Setup
            intensity = 2 if train_y.shape[1] > 1 else 1
            if DEVICE.type == 'cuda':
                intensity *= 6  # Aggressive for CUDA
            else:
                intensity *= 2  # Conservative for CPU/DML to maintain OS responsiveness
            
            # Reduce base samples for multi-objective to prevent OOM
            is_mo = train_y.shape[1] > 1
            base_mc_samples = 64 if is_mo else 256
            mc_samples = int(base_mc_samples * intensity)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

            if train_y.shape[1] > 1:
                # Multi-objective: use qNEHVI for tradeoffs
                mo_obj = WeightedMCMultiOutputObjective(weights=weights_t.to(OPT_DEVICE))
                ty_compute = train_y.to(OPT_DEVICE)
                mins, maxs = ty_compute.min(dim=0).values, ty_compute.max(dim=0).values
                ref_point = (mins - 0.2 * (maxs - mins + 1e-6)) * weights_t.to(OPT_DEVICE)
                acq_func = qLogNoisyExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=ref_point,
                    X_baseline=train_x.to(OPT_DEVICE),
                    sampler=sampler, # Use vectorized sampler
                    objective=mo_obj,
                    prune_baseline=True
                ).to(OPT_DEVICE)
            else:
                # Single-objective
                if acq_type == 'EI':
                    acq_func = qLogExpectedImprovement(
                        model=model,
                        best_f=train_y.to(OPT_DEVICE).max().item(),
                        sampler=sampler, 
                        objective=None
                    ).to(OPT_DEVICE)
                else:
                    beta = (1.0 - exploration) * 10.0 + 0.1
                    acq_func = UpperConfidenceBound(model, beta=beta).to(OPT_DEVICE)

            p_constraints = []
            if tweaks.get('constraints'):
                try:
                    raw_expr = tweaks.get('constraints').strip().replace('<= 0', '')
                    # Pre-compile vectorized torch expression
                    indices = {f['name']: i for i, f in enumerate(features_config)}
                    vec_expr = raw_expr
                    # Replace names with X[..., i] to allow full vectorization on GPU
                    for name, idx in indices.items():
                        vec_expr = re.sub(rf'\b{re.escape(name)}\b', f'X[..., {idx}]', vec_expr)
                    
                    code = compile(f"torch.negative({vec_expr})", '<string>', 'eval')
                    def constraint_func(X):
                        # X is provided on OPT_DEVICE during optimization
                        return eval(code, {"X": X, "torch": torch, "np": np, "__builtins__": {}})
                    p_constraints = [constraint_func]
                except Exception as ex: print(f"Constraint error: {ex}")

            # Optimization with controlled memory usage
            base_restarts = 8 if is_mo else 32
            base_raw_samples = 128 if is_mo else 512
            
            candidates, _ = optimize_acqf(
                acq_function=acq_func, 
                bounds=bounds_t.to(OPT_DEVICE), 
                q=batch_size,
                num_restarts=int(base_restarts * intensity), 
                raw_samples=int(base_raw_samples * intensity), 
                nonlinear_inequality_constraints=p_constraints if p_constraints else None
            )
            # Ensure candidates are on CPU for decoding
            candidates = candidates.cpu()
        
        # 4. Decode Results
        feat_names = [f['name'] for f in features_config]
        def get_fingerprint(row_dict):
            parts = []
            for name in feat_names:
                val = row_dict[name]
                parts.append(f"{float(val):.3f}" if isinstance(val, (int, float)) else str(val))
            return "|".join(parts)

        existing_fps = set()
        for _, row in df[feat_names].iterrows():
            existing_fps.add(get_fingerprint(row.to_dict()))

        suggestions = []
        for cand in candidates:
            def decode_cand(c_tensor):
                row = {}
                for i, f in enumerate(features_config):
                    val = c_tensor[i].item()
                    if f['type'] == 'discrete':
                        d_vals = sorted([float(x.strip()) for x in f['range'].split(',') if x.strip()])
                        val = min(d_vals, key=lambda x: abs(x - val)) if d_vals else val
                    elif f['type'] == 'categorical':
                        choices = [x.strip() for x in f['range'].split(',')]
                        idx = max(0, min(len(choices)-1, int(round(val))))
                        val = choices[idx]
                    if isinstance(val, (int, float)): val = round(float(val), 3)
                    row[f['name']] = val
                return row

            suggestion = decode_cand(cand)
            if avoid_reval:
                attempts = 0
                t_cand = cand.clone()
                while get_fingerprint(suggestion) in existing_fps and attempts < 20:
                    attempts += 1
                    for i, f in enumerate(features_config):
                        if f['type'] == 'continuous':
                            span = bounds[i][1] - bounds[i][0]
                            t_cand[i] = torch.clamp(t_cand[i] + (torch.rand(1).item() - 0.5) * span * 0.1, bounds[i][0], bounds[i][1])
                        elif f['type'] == 'discrete':
                            d_vals = sorted([float(x.strip()) for x in f['range'].split(',') if x.strip()])
                            if len(d_vals) > 1:
                                cur_idx = d_vals.index(suggestion[f['name']])
                                t_cand[i] = d_vals[(cur_idx + attempts) % len(d_vals)]
                        elif f['type'] == 'categorical':
                            choices = [x.strip() for x in f['range'].split(',')]
                            if len(choices) > 1:
                                cur_idx = choices.index(suggestion[f['name']])
                                t_cand[i] = float((cur_idx + attempts) % len(choices))
                    suggestion = decode_cand(t_cand)
            suggestions.append(suggestion)
            existing_fps.add(get_fingerprint(suggestion))

        return jsonify({'suggestions': suggestions})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/doe', methods=['POST'])
def run_doe():
    try:
        req_data = request.get_json()
        features_config = req_data.get('features', [])
        tweaks = req_data.get('tweaks', {})
        model_type = tweaks.get('model', 'bbdesign') # Default to Box-Behnken
        max_runs = int(tweaks.get('max_runs', 50))

        # Filter features with valid configurations
        valid_features = []
        for f in features_config:
            name = f.get('name', 'Unnamed')
            f_type = f.get('type', 'continuous')
            f_range = f.get('range', '')

            if not f_range:
                continue

            if f_type == 'categorical':
                choices = [x.strip() for x in f_range.split(',') if x.strip()]
                if choices:
                    valid_features.append({
                        'name': name, 
                        'choices': choices, 
                        'type': 'categorical',
                        'range': f_range
                    })
            else:
                # Support both [min, max] and min, max formats
                matches = [float(m) for m in re.findall(r"[-+]?\d*\.?\d+", f_range)]
                if len(matches) >= 2:
                    low, high = min(matches), max(matches)
                    valid_features.append({
                        'name': name, 
                        'low': low, 
                        'high': high, 
                        'type': f_type,
                        'range': f_range
                    })
        
        n_factors = int(len(valid_features))
        print(f"DEBUG: DoE Run - Model: {model_type}, n_factors: {n_factors}, Features: {[f['name'] for f in valid_features]}")

        if n_factors < 2:
            return jsonify({'error': f'DoE requires at least 2 features. Found {n_factors} valid features.'}), 400
        
        if model_type == 'bbdesign' and n_factors < 3:
            return jsonify({'error': f'Box-Behnken design requires at least 3 factors. Found {n_factors}.'}), 400

        # Generate Design
        try:
            if model_type == 'ccdesign':
                design = doe.ccdesign(n_factors)
            elif model_type == 'bbdesign':
                design = doe.bbdesign(n_factors)
            elif model_type == 'fracfact':
                if n_factors <= 4:
                    levels = [int(2)] * n_factors
                    design = doe.fullfact(levels).astype(float)
                    design = design * 2 - 1
                else:
                    design = doe.pbdesign(n_factors).astype(float)
            elif model_type == 'def_screening' or model_type == 'pbdesign':
                design = doe.pbdesign(n_factors).astype(float)
            elif model_type == 'lhs':
                from pyDOE2 import lhs as lhs_sampler
                n_numeric = sum(1 for f in valid_features if f['type'] != 'categorical')
                n_cat = len(valid_features) - n_numeric
                n_samples = min(max_runs, max_runs)  # use max_runs directly
                if n_numeric > 0:
                    # LHS on numeric features in [0, 1], then scale to [-1, 1]
                    lhs_numeric = lhs_sampler(n_numeric, samples=n_samples, criterion='maximin')
                    lhs_numeric = lhs_numeric * 2 - 1  # scale to [-1, 1] for compatibility
                else:
                    lhs_numeric = np.empty((n_samples, 0))
                # Build full design: insert numeric columns in order, categorical handled separately
                design = np.zeros((n_samples, len(valid_features)))
                num_col = 0
                for col_idx, feat in enumerate(valid_features):
                    if feat['type'] != 'categorical':
                        design[:, col_idx] = lhs_numeric[:, num_col]
                        num_col += 1
                    # categorical col stays 0; handled in the conversion loop below
            else:
                design = doe.bbdesign(n_factors)
        except Exception as design_err:
            print(f"Design generation error: {design_err}")
            # Fallback to PB design which is very robust
            design = doe.pbdesign(n_factors).astype(float)
            model_type = "Plackett-Burman (Fallback)"

        # Scale and Limit
        original_len = len(design)
        if original_len > max_runs:
            design = design[:max_runs]

        # Convert to Feature Space
        suggested_table = []
        import random
        for row_idx, row in enumerate(design):
            entry = {}
            for i, feat in enumerate(valid_features):
                if feat['type'] == 'categorical':
                    if model_type == 'lhs':
                        # Stratified assignment: cycle through choices, then shuffle within groups
                        n_choices = len(feat['choices'])
                        n_rows = len(design)
                        # Build a stratified list: repeat choices as evenly as possible
                        strat = [feat['choices'][j % n_choices] for j in range(n_rows)]
                        # Shuffle with a fixed seed for reproducibility per feature
                        rng = random.Random(i)
                        rng.shuffle(strat)
                        entry[feat['name']] = strat[row_idx]
                    else:
                        # Original approach for classical designs
                        idx = 0
                        if row[i] > 0.5: idx = len(feat['choices']) - 1
                        elif row[i] < -0.5: idx = 0
                        else: idx = len(feat['choices']) // 2
                        entry[feat['name']] = feat['choices'][idx]
                else:
                    # Map design space [-1, 1] to user space [low, high]
                    val = feat['low'] + (row[i] + 1) / 2 * (feat['high'] - feat['low'])
                    
                    if feat['type'] == 'discrete':
                        d_vals = sorted([float(x.strip()) for x in feat['range'].split(',') if x.strip()])
                        if d_vals:
                            val = min(d_vals, key=lambda x: abs(x - val))
                        else:
                            val = round(float(val))
                    else:
                        val = round(float(val), 3)
                    entry[feat['name']] = val
            suggested_table.append(entry)

        # Calculate Quality Metrics
        try:
            # Prepare numeric design matrix for calculations in [-1, 1] space
            X = design.astype(float)
            N_runs, k_factors = X.shape
            
            # 1. Orthogonality (Correlation)
            if model_type == 'lhs':
                ortho_score = "N/A"
            elif k_factors > 1:
                # Correlations between columns (features)
                corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
                np.fill_diagonal(corr_matrix, 0)
                # Handle possible NaNs from zero-variance columns
                max_corr = np.nanmax(corr_matrix) if not np.all(np.isnan(corr_matrix)) else 1.0
                ortho_score = max(0, 100 * (1 - max_corr))
            else:
                ortho_score = 100
            
            # 2. D-Efficiency
            # Standard D-eff = (|X'X|^(1/k)) / N
            try:
                if model_type == 'lhs':
                    d_eff = "N/A"
                else:
                    xtx = np.dot(X.T, X)
                    det = np.linalg.det(xtx)
                    if det > 0:
                        d_eff = (det**(1.0/k_factors)) / N_runs * 100
                    else:
                        d_eff = 0
            except:
                d_eff = 0
                
            # 3. Resolution
            # More nuanced logic based on model and truncation
            is_truncated = (len(design) < original_len)
            
            # Check rank to see if we can at least solve for main effects
            # (Intercept + k factors)
            design_with_intercept = np.column_stack([np.ones(N_runs), X])
            rank = np.linalg.matrix_rank(design_with_intercept)
            
            if rank < (k_factors + 1):
                resolution = "Poor (Insuff. Runs)"
            elif is_truncated:
                resolution = "Reduced (Truncated)"
            elif model_type == 'lhs':
                resolution = "N/A (Space-Filling)"
            elif model_type == 'bbdesign' or model_type == 'ccdesign':
                resolution = "V+ (High)"
            elif model_type == 'fracfact' or 'Plackett' in model_type:
                resolution = "III (Screening)"
            else:
                resolution = "IV (Moderate)"

            
            # 4. Curvature Detection
            # Check if there are center points (0) or multiple levels
            levels = len(np.unique(design))
            has_center = np.any(np.all(np.abs(design) < 0.1, axis=1))
            if levels > 2 and has_center:
                curvature = "Excellent"
            elif levels > 2:
                curvature = "Partial"
            else:
                curvature = "None (Linear only)"

            metrics = {
                'orthogonality': round(ortho_score, 1) if isinstance(ortho_score, (int, float)) else ortho_score,
                'efficiency': round(d_eff, 1) if isinstance(d_eff, (int, float)) else d_eff,
                'resolution': resolution,
                'curvature': curvature,
                'runs': len(suggested_table),
                'model': model_type
            }
        except Exception as met_err:
            print(f"Metrics calculation error: {met_err}")
            metrics = {
                'orthogonality': 0, 'efficiency': 0, 'resolution': "Error", 
                'curvature': "Error", 'runs': len(suggested_table), 'model': model_type
            }
        
        return jsonify({
            'suggestions': suggested_table,
            'metrics': metrics
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/sa', methods=['POST'])
def run_sa():
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        tweaks = req_data.get('tweaks', {})
        
        df = pd.DataFrame(data, columns=columns)
        
        # Ensure categorical columns are treated as strings to avoid mixed-type issues
        for f in features_config:
            if f.get('type') == 'categorical':
                df[f['name']] = df[f['name']].astype(str)
        
        # Prepare X and Y
        feat_names = [f['name'] for f in features_config if f.get('range')]
        obj_names = [o['name'] for o in objectives_config]
        
        if not feat_names or not obj_names:
            return jsonify({'error': 'Missing features or objectives for analysis.'}), 400

        # Extract raw features and objectives
        X_raw = df[feat_names].copy()
        Y_raw = df[obj_names].copy()
        
        # 1. Goal Transformation & Global Success Index
        # We transform objectives into "Success Scores" (Desirability)
        success_data = {}
        processed_success = []
        weights = []
        
        for obj in objectives_config:
            name = obj['name']
            w = float(obj.get('importance', 100))
            weights.append(w)
            
            raw_y = pd.to_numeric(Y_raw[name], errors='coerce').fillna(0).values
            
            if obj['type'] == 'maximize':
                transformed = raw_y
            elif obj['type'] == 'minimize':
                transformed = -raw_y
            else: # target
                t = float(obj.get('target', 0))
                transformed = -np.abs(raw_y - t)
            
            # Normalize for weighting: S' = (S - min) / (max - min)
            # This ensures weights are meaningful across different scales/units
            s_min, s_max = transformed.min(), transformed.max()
            if s_max > s_min:
                normalized = (transformed - s_min) / (s_max - s_min)
            else:
                normalized = np.ones_like(transformed) # All points equally successful if no variation
                
            success_data[name] = normalized
            processed_success.append(normalized)
        
        # Calculate Weighted Global Success Index
        if len(processed_success) >= 2:
            succ_matrix = np.column_stack(processed_success)
            w_arr = np.array(weights)
            if np.sum(w_arr) > 0:
                global_success = np.average(succ_matrix, axis=1, weights=w_arr)
            else:
                global_success = np.mean(succ_matrix, axis=1)
            success_data['global_success'] = global_success
        else:
            global_success = None
                
        # Build the Success DataFrame for analysis
        Y_success = pd.DataFrame(success_data)
        extended_obj_names = list(success_data.keys())

        # 2. One-Hot Encoding for categorical features
        X_encoded = pd.get_dummies(X_raw, drop_first=False)
        encoded_feat_names = list(X_encoded.columns)

        X = X_encoded.fillna(0).astype(float)
        Y = Y_success.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        
        # Correlation Matrix now reflects "Success Correlation"
        corr_matrix = pd.concat([X, Y], axis=1).corr().fillna(0).to_dict()
        
        # Model and Metrics (Analyzing Success)
        model_name = tweaks.get('model', 'linear') 
        params = tweaks.get('params', {})
        
        results = {}
        for obj in extended_obj_names:
            y = Y[obj]
            if model_name == 'random_forest':
                model = ensemble.RandomForestRegressor(
                    n_estimators=int(params.get('n_estimators', 100)),
                    max_depth=int(params.get('max_depth', 10)) if params.get('max_depth') else None
                )
            elif model_name == 'mlp':
                hidden_layers = str(params.get('mlp_layers', '100'))
                layers = [int(x.strip()) for x in hidden_layers.split(',') if x.strip()]
                model = MLPRegressor(
                    hidden_layer_sizes=tuple(layers),
                    max_iter=int(params.get('mlp_iter', 500)),
                    random_state=42
                )
            else:
                model = lm.LinearRegression()
            
            # Validation (Bootstrapping style)
            ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            scores = []
            
            # MLP/NN is scale sensitive
            X_data = X
            if model_name == 'mlp':
                scaler = StandardScaler()
                X_data = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            for train_idx, test_idx in ss.split(X_data):
                X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model.fit(X_train, y_train)
                scores.append(model.score(X_test, y_test))
            
            reliability = np.mean(scores)
            
            # Fit final model for importance
            model.fit(X_data, y)
            if model_name == 'linear':
                importance = np.abs(model.coef_)
            elif model_name == 'mlp':
                # Use permutation importance for Neural Network
                imp_res = permutation_importance(model, X_data, y, n_repeats=5, random_state=42)
                importance = imp_res.importances_mean
            else:
                importance = model.feature_importances_
            
            imp_sum = np.sum(importance)
            importance_pct = (importance / imp_sum * 100).tolist() if imp_sum > 0 else [0.0] * len(importance)

            # Group dummy importance back to original features for cleaner plots
            grouped_importance = {}
            for f_name in feat_names:
                # Find all dummy columns derived from this original feature
                # (Either the name itself for numeric, or f_name + "_" for categoricals)
                rel_indices = [i for i, c in enumerate(encoded_feat_names) if c == f_name or f"{f_name}_" in c]
                val = sum(importance_pct[i] for i in rel_indices)
                grouped_importance[f_name] = round(val, 2)
            
            results[obj] = {
                'reliability': None if np.isnan(reliability) else round(float(reliability), 3),
                'importance': grouped_importance,
                'fit_flag': 'Good' if reliability > 0.7 else ('Limited' if reliability >= 0.4 else 'Poor')
            }
            
        # Categorical vs Numerical features identification based on frontend config
        cat_features = [f['name'] for f in features_config if f.get('type') == 'categorical']
        num_features = [f['name'] for f in features_config if f.get('type') in ['continuous', 'discrete']]
        
        cat_interactions = {}
        for i in range(len(cat_features)):
            for j in range(i + 1, len(cat_features)):
                f1, f2 = cat_features[i], cat_features[j]
                pair_key = f"{f1} vs {f2}"
                cat_interactions[pair_key] = {}
                for obj in extended_obj_names:
                    # Pivot table for MAX objective value (Best Achieved)
                    combo_df = pd.concat([df[[f1, f2]], Y_success[[obj]]], axis=1)
                    pivot = combo_df.groupby([f1, f2])[obj].max().unstack().fillna(0)
                    cat_interactions[pair_key][obj] = {
                        'z': pivot.values.tolist(),
                        'x': pivot.columns.tolist(),
                        'y': pivot.index.tolist()
                    }

        # Numerical Correlation: standard Pearson for numerical features vs success targets
        num_df = df[num_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        num_corr_matrix = pd.concat([num_df, Y_success], axis=1).corr().loc[extended_obj_names, num_features].fillna(0).to_dict()

        return jsonify({
            'numerical_correlation': num_corr_matrix,
            'cat_interactions': cat_interactions,
            'results': results,
            'cat_features': cat_features,
            'num_features': num_features,
            'extended_obj_names': extended_obj_names
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/estimate', methods=['POST'])
def estimate_sa():
    try:
        req_data = request.get_json()
        data = req_data.get('data')
        columns = req_data.get('columns')
        input_values = req_data.get('inputs') # {feat_name: val}
        features_config = req_data.get('features', [])
        objectives_config = req_data.get('objectives', [])
        model_name = req_data.get('model', 'linear')
        params = req_data.get('params', {})
        
        df = pd.DataFrame(data, columns=columns)
        feat_names = [f['name'] for f in features_config if f.get('range')]
        obj_names = [o['name'] for o in objectives_config]
        
        # We must fit the model exactly identically to run_sa, capturing the one-hot structure
        X_raw_train = df[feat_names].copy()
        X_train_encoded = pd.get_dummies(X_raw_train)
        encoded_cols = X_train_encoded.columns
        X_train = X_train_encoded.fillna(0).astype(float)
        
        # Handle target values
        for o in obj_names:
            if df[o].dtype == object or str(df[o].dtype) in ['string', 'category']:
                df[o] = df[o].astype(str).astype('category').cat.codes
        Y = df[obj_names].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        
        # Prepare the prediction input point with proper Categorical alignment
        input_data_raw = pd.DataFrame([input_values], columns=feat_names)
        for col in feat_names:
            if df[col].dtype == object or str(df[col].dtype) in ['string', 'category']:
                unique_train_vals = df[col].astype(str).unique()
                input_data_raw[col] = pd.Categorical(input_data_raw[col].astype(str), categories=unique_train_vals)
        
        input_df_encoded = pd.get_dummies(input_data_raw)
        input_df_aligned = input_df_encoded.reindex(columns=encoded_cols, fill_value=0).astype(float)
        
        # MLP requires scaling
        X_train_data = X_train
        input_data = input_df_aligned
        if model_name == 'mlp':
            scaler = StandardScaler()
            X_train_data = pd.DataFrame(scaler.fit_transform(X_train), columns=encoded_cols)
            input_data = pd.DataFrame(scaler.transform(input_df_aligned), columns=encoded_cols)

        predictions = {}
        for obj in obj_names:
            if model_name == 'random_forest':
                model = ensemble.RandomForestRegressor(
                    n_estimators=int(params.get('n_estimators', 100)),
                    max_depth=int(params.get('max_depth', 10)) if params.get('max_depth') else None
                )
            elif model_name == 'mlp':
                hidden_layers = str(params.get('mlp_layers', '100'))
                layers = [int(x.strip()) for x in hidden_layers.split(',') if x.strip()]
                model = MLPRegressor(
                    hidden_layer_sizes=tuple(layers),
                    max_iter=int(params.get('mlp_iter', 500)),
                    random_state=42
                )
            else:
                model = lm.LinearRegression()
            
            model.fit(X_train_data, Y[obj])
            pred = model.predict(input_data)[0]
            predictions[obj] = round(float(pred), 3)
            
        # Calculate Success Feedback
        success_feedback = {}
        processed_success = []
        weights = []
        
        for obj in objectives_config:
            name = obj['name']
            val = predictions[name]
            w = float(obj.get('importance', 100))
            weights.append(w)
            
            # Transformation
            if obj['type'] == 'maximize':
                s = val
            elif obj['type'] == 'minimize':
                s = -val
            else: # target
                t = float(obj.get('target', 0))
                s = -abs(val - t)
            
            # Normalize based on training data success range
            # Note: We recalculate training success range for consistency
            train_raw = Y[name].values
            if obj['type'] == 'maximize': train_s = train_raw
            elif obj['type'] == 'minimize': train_s = -train_raw
            else: train_s = -np.abs(train_raw - float(obj.get('target', 0)))
            
            s_min, s_max = train_s.min(), train_s.max()
            if s_max > s_min:
                norm_s = (s - s_min) / (s_max - s_min)
            else:
                norm_s = 1.0
            
            success_feedback[name] = round(float(norm_s * 100), 1)
            processed_success.append(norm_s)
            
        if processed_success:
            w_arr = np.array(weights)
            if np.sum(w_arr) > 0:
                global_score = np.average(processed_success, weights=w_arr)
            else:
                global_score = np.mean(processed_success)
        else:
            global_score = 0
            
        return jsonify({
            'predictions': predictions,
            'success_scores': success_feedback,
            'global_success': round(float(global_score * 100), 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export_csv():
    req_data = request.get_json()
    data = req_data.get('data')
    columns = req_data.get('columns')
    
    df = pd.DataFrame(data, columns=columns)
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='edos_export.csv'
    )


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 7860))
    
    # Only open browser if running locally on default port
    if port == 5000:
        import threading
        import webbrowser
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(f'http://127.0.0.1:{port}')
        threading.Thread(target=open_browser, daemon=True).start()
        
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

