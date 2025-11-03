# filename: full_gep_suite.py
import os

import numpy as np
import time
import sys
import multiprocessing

# --- Set start method early (Crucial for C++ extensions) ---
if __name__ == '__main__':
    try:
        # Force 'spawn' to prevent GIL issues with C++ extensions
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# --- Imports (AFTER setting start method) ---
# C++ Kernel: Contains GEPObjective and Solvers
# 1. Get the current directory (demo/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory (the project root, which contains cbo_module.so)
parent_dir = os.path.join(current_dir, '..')

# 3. Insert the parent directory path to Python's module search path
sys.path.insert(0, parent_dir)
import cbo_module

# Python Runner: Contains ExperimentRunner class
# Assumes ExperimentRunner is in ackley_experiment_parallel.py
from experiment_runner import ExperimentRunner

# Plotting Logic
from visualisation import plot_cbo_results


# ==============================================================================
# GEP EXPERIMENT CONFIGURATION
# ==============================================================================

# 1. Experiment Parameters
D_DIMENSION = 10       # d: Dimension of the orthogonal group O(d) (n=k=d)
NUM_PARTICLES = 50     # N: Number of particles
# Note: Sigma might need tuning for GEP. Start with a moderate value.
SIGMA_VAL = 0.08        # sigma: Noise coefficient (adjust as needed)
T_VAL = 100.0          # T: Total simulation time (adjust as needed)
DT_VAL = 0.01          # dt: Time step
NUM_TRIALS = 8 * 13       # Number of independent trials

# 2. Success Tolerance (Absolute error from true minimum)
SUCCESS_TOLERANCE_ABS = 0.1

# 3. Solver Parameter Dictionaries
KIM_SOLVER_PARAMS = {
    'beta_val': 100.0, # Beta often lower for quadratic-like objectives
    'lambda_val': 1.0,
    # 'sigma_val' will be added later
}

COR_SOLVER_PARAMS = {
    'beta_val': 10,
    'lambda_initial': 1,
    'lambda_min': 0.1,
    'lambda_max': 5.0,
    'lambda_increase_factor': 1.1,
    'lambda_decrease_factor': 0.98,
    'lambda_adapt_interval': 50,
    'lambda_stagnation_thresh': 0.005,
    'lambda_convergence_thresh': 0.3,
    'sigma_initial': 0.5,
    'sigma_final': 0.0,
    'sigma_max': 1.0,
    'annealing_rate': 8,
    'reheat_check_interval': 5000000,
    'reheat_window': 50000 * 2,
    'reheat_threshold': 1e-4,
    'reheat_sigma_boost': 0.002,
    'reheat_lambda_reset': True,
}

# 4. Objective Parameters (None needed, A generated internally)
GEP_OBJ_PARAMS = {}

# ==============================================================================
# MAIN EXECUTION FUNCTION
# ==============================================================================

def setup_and_run_gep_experiment(solver_class, backend):
    """
    Sets up and runs the GEP experiment case O(d) = V(d, d).
    """

    # Select and configure base solver parameters
    if solver_class == cbo_module.KimCBOSolver:
        solver_params = KIM_SOLVER_PARAMS.copy()
        solver_params['sigma_val'] = SIGMA_VAL
    elif solver_class == cbo_module.CormacsCBOSolver:
        solver_params = COR_SOLVER_PARAMS.copy()
        solver_params['sigma_initial'] = SIGMA_VAL
    else:
        raise ValueError(f"Unknown solver class: {solver_class.__name__}")

    solver_params['backend'] = backend

    OBJECTIVE_NAME_CPP = "GEP" # String identifier for C++ objective

    print("=" * 60)
    print(f"--- STARTING GEP SUITE: {solver_class.__name__} (C++) ---")
    print(f"--- Target O(d) = V({D_DIMENSION}, {D_DIMENSION}) ---")
    print(f"--- ACCELERATION BACKEND: {backend.upper()} ---")
    print("=" * 60)

    # Instantiate objective once to get the true minimum for reporting
    temp_gep_obj = cbo_module.GEPObjective(D_DIMENSION) # Pass only 'd'
    true_min_for_reporting = temp_gep_obj.get_true_minimum()
    print(f"[INFO] Approximate True Minimum for GEP (based on one instance): {true_min_for_reporting:.4f}")

    # Define success based on absolute error from this reported minimum
    calculated_success_tolerance = true_min_for_reporting + SUCCESS_TOLERANCE_ABS

    # Create and Run ExperimentRunner
    runner = ExperimentRunner(
        solver_class=solver_class,
        objective_name=OBJECTIVE_NAME_CPP,
        n=D_DIMENSION, # Pass d as n
        k=D_DIMENSION, # Pass d as k
        num_particles=NUM_PARTICLES,
        T=T_VAL,
        dt=DT_VAL,
        trials=NUM_TRIALS,
        success_tolerance=calculated_success_tolerance, # Check against absolute threshold
        solver_params=solver_params,
        objective_params=GEP_OBJ_PARAMS
    )

    runner.run_trials_parallel()

    # Override true_minimum in runner for accurate summary print
    runner.true_minimum = true_min_for_reporting

    return runner


if __name__ == '__main__':

    # --- 1. CHOOSE SOLVER AND BACKEND ---
    # SOLVER_TO_RUN = cbo_module.KimCBOSolver
    SOLVER_TO_RUN = cbo_module.CormacsCBOSolver

    ACCELERATION_BACKEND = 'eigen'

    # --- 2. EXECUTE SUITE ---
    gep_runner = setup_and_run_gep_experiment(SOLVER_TO_RUN, ACCELERATION_BACKEND)

    # --- 3. PLOT RESULTS ---
    time_pts, dynamics, asymptotics, obj_instance = gep_runner.get_results_for_plotting()

    if time_pts is not None and obj_instance:
        print("\nPlotting results...")
        plot_cbo_results(
            "gep",
            time_pts,
            dynamics,
            asymptotics,
            obj_instance,
            SOLVER_TO_RUN.__name__,
            D_DIMENSION,
            D_DIMENSION
        )

    print("\n--- GEP experiment suite execution finished. ---")