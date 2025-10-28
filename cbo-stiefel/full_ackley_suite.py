# filename: full_ackley_suite.py

import numpy as np
import time
import sys
import multiprocessing
import inspect # Still useful for getting solver name

# --- Set start method early (Crucial for C++ extensions) ---
if __name__ == '__main__':
    try:
        # Force the 'spawn' start method to prevent GIL issues with C++ extensions
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Ignore if already set or context prevents changing (e.g., interactive session)
        pass

# --- Imports (AFTER setting start method) ---
sys.path.append('.')
import cbo_module # C++ kernel module
# ExperimentRunner class should be defined in ackley_experiment_parallel.py
from experiment_runner import ExperimentRunner
from visualisation import plot_cbo_results, save_cbo_data_to_pickle  # Plotting function
# Note: ackley.py (Python AckleyObjective) is no longer needed here,
# as the C++ module now contains the objective definition.


# ==============================================================================
# BASE PARAMETER DEFINITIONS
# ==============================================================================
NUM_PARTICLES = 50
DT_VAL = 0.05
NUM_TRIALS = 8*13 # For full replication

# --- C++ Solver Parameter Dictionaries ---
# These must match the parameter names in the C++ constructors bound by Pybind11

# Kim's Solver Parameters (Matches C++ constructor args)
KIM_SOLVER_PARAMS = {
    'beta_val': 5000.0,
    'lambda_val': 1.0,
    # 'sigma_val' will be added per experiment
    # 'backend' will be added per experiment run
}

# Cormac's Solver Parameters (Matches C++ constructor args - full list)
COR_SOLVER_PARAMS = {
    'beta_val': 5000.0,
    'lambda_initial': 1.0,
    'lambda_min': 0.1,
    'lambda_max': 5.0,
    'lambda_increase_factor': 1.1,
    'lambda_decrease_factor': 0.98,
    'lambda_adapt_interval': 20,
    'lambda_stagnation_thresh': 0.005,
    'lambda_convergence_thresh': 0.3,
    'sigma_initial': 0.5,
    'sigma_final': 0.0,
    'sigma_max': 1.0,
    'annealing_rate': 10.0,
    'reheat_check_interval': 50_000,
    'reheat_window': 100_000,
    'reheat_threshold': 1e-4,
    'reheat_sigma_boost': 0.2,
    'reheat_lambda_reset': True,
}

# Ackley Objective Parameters for C++ constructor (if needed, otherwise defaults used)
# AckleyObjective in C++ uses defaults, so this is empty for now.
ACKLEY_OBJ_PARAMS = {}


# ==============================================================================
# EXPERIMENT SETUP FUNCTIONS
# ==============================================================================

def setup_and_run_ackley_experiment(solver_class, n, k, sigma_config, T, success_tolerance, backend):
    """
    General function to configure and run an Ackley experiment case using the C++ kernel.
    """
    objective_name_cpp = "Ackley" # String identifier for the C++ objective

    # Select and configure base solver parameters
    if solver_class == cbo_module.KimCBOSolver:
        solver_params = KIM_SOLVER_PARAMS.copy()
        solver_params['sigma_val'] = sigma_config # Kim uses 'sigma_val'
    elif solver_class == cbo_module.CormacsCBOSolver:
        solver_params = COR_SOLVER_PARAMS.copy()
        solver_params['sigma_initial'] = sigma_config # Cormac uses 'sigma_initial'
    else:
        raise ValueError(f"Unknown solver class: {solver_class}")

    # Add the backend selection
    solver_params['backend'] = backend

    # Create the ExperimentRunner instance
    # Note: objective_params is passed but might be empty if C++ objective uses defaults
    runner = ExperimentRunner(
        solver_class=solver_class,
        objective_name=objective_name_cpp, # Pass name for C++ objective instantiation
        n=n,
        k=k,
        num_particles=NUM_PARTICLES,
        T=T,
        dt=DT_VAL,
        trials=NUM_TRIALS,
        success_tolerance=success_tolerance,
        solver_params=solver_params, # Pass the prepared dictionary
        objective_params=ACKLEY_OBJ_PARAMS
    )

    # Run the parallel trials using the C++ kernel
    runner.run_trials_parallel()

    return runner

# --- Specific Experiment Cases ---

def run_v3_5(solver_class, backend):
    """V(3, 5) case: n=5, k=3, sigma=0.5, T=100.0, success_tol=2e-3."""
    print("\n--- Running Experiment: V(3, 5) ---")
    return setup_and_run_ackley_experiment(
        solver_class=solver_class, n=5, k=3, sigma_config=0.5,
        T=100.0, success_tolerance=2.0e-3, backend=backend
    )

def run_v10_20(solver_class, backend):
    """V(10, 20) case: n=20, k=10, sigma=0.11, T=1500.0, success_tol=0.5."""
    print("\n--- Running Experiment: V(10, 20) ---")
    return setup_and_run_ackley_experiment(
        solver_class=solver_class, n=20, k=10, sigma_config=0.11,
        T=1500.0, success_tolerance=0.5, backend=backend
    )

def run_v1_3(solver_class, backend):
    """V(1, 3) case: n=3, k=1, sigma=0.17, T=10.0, success_tol=1e-2."""
    print("\n--- Running Experiment: V(1, 3) ---")
    return setup_and_run_ackley_experiment(
        solver_class=solver_class, n=3, k=1, sigma_config=0.17,
        T=10.0, success_tolerance=1.0e-2, backend=backend
    )

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':

    # --- CHOOSE SOLVER AND BACKEND ---
    SOLVER_TO_USE = cbo_module.KimCBOSolver # C++ Kim solver reference
    # SOLVER_TO_USE = cbo_module.CormacsCBOSolver # C++ Cormac solver reference

    ACCELERATION_BACKEND = 'eigen' # Choose 'eigen' (CPU) or 'torch' (MPS/GPU SVD)
    # ACCELERATION_BACKEND = 'torch' # Choose 'eigen' (CPU) or 'torch' (MPS/GPU SVD)

    # --- Check if tqdm is available ---
    try:
        from tqdm import tqdm
    except ImportError:
        print("\n[INFO] 'tqdm' not found. Progress bars disabled. Install with 'pip install tqdm'.")

    print(f"\n--- STARTING FULL ACKLEY SUITE ---")
    print(f"--- Using Solver: {SOLVER_TO_USE.__name__} (C++) ---")
    print(f"--- Using Backend: {ACCELERATION_BACKEND.upper()} ---")

    start_suite_time = time.time()

    # --- Run V(3, 5) ---
    runner_v3_5 = run_v3_5(SOLVER_TO_USE, ACCELERATION_BACKEND)
    time_pts, dynamics, asymptotics, obj_instance = runner_v3_5.get_results_for_plotting()
    if time_pts is not None: # Check if run succeeded
        save_cbo_data_to_pickle("ackley", time_pts, dynamics, asymptotics, obj_instance, SOLVER_TO_USE.__name__, k=3, n=5)
        plot_cbo_results("ackley", time_pts, dynamics, asymptotics, obj_instance, SOLVER_TO_USE.__name__, k=3, n=5)

    # --- Run V(10, 20) ---
    runner_v10_20 = run_v10_20(SOLVER_TO_USE, ACCELERATION_BACKEND)
    time_pts, dynamics, asymptotics, obj_instance = runner_v10_20.get_results_for_plotting()
    if time_pts is not None:
        save_cbo_data_to_pickle("ackley", time_pts, dynamics, asymptotics, obj_instance, SOLVER_TO_USE.__name__, k=10, n=20)
        plot_cbo_results("ackley", time_pts, dynamics, asymptotics, obj_instance, SOLVER_TO_USE.__name__, k=10, n=20)

    # --- Run V(1, 3) ---
    runner_v1_3 = run_v1_3(SOLVER_TO_USE, ACCELERATION_BACKEND)
    time_pts, dynamics, asymptotics, obj_instance = runner_v1_3.get_results_for_plotting()
    if time_pts is not None:
        save_cbo_data_to_pickle("ackley", time_pts, dynamics, asymptotics, obj_instance, SOLVER_TO_USE.__name__, k=1, n=3)
        plot_cbo_results("ackley", time_pts, dynamics, asymptotics, obj_instance, SOLVER_TO_USE.__name__, k=1, n=3)

    end_suite_time = time.time()
    print("\n--- All Ackley experiments complete. ---")
    print(f"Total Suite Runtime: {end_suite_time - start_suite_time:.2f} seconds.")