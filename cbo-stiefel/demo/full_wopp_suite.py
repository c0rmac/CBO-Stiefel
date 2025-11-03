# filename: full_wopp_suite.py
import os

import numpy as np
import time
import sys
import multiprocessing

# --- Set start method early (Crucial for C++ extensions) ---
if __name__ == '__main__':
    try:
        # Force the 'spawn' start method to prevent GIL deadlocks
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# --- Imports (AFTER setting start method) ---
# C++ Kernel: Contains WOPPObjective and Solvers
# 1. Get the current directory (demo/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory (the project root, which contains cbo_module.so)
parent_dir = os.path.join(current_dir, '..')

# 3. Insert the parent directory path to Python's module search path
sys.path.insert(0, parent_dir)
import cbo_module

# Python Runner: Contains ExperimentRunner class
from experiment_runner import ExperimentRunner

# Plotting Logic
from visualisation import plot_cbo_results

# ==============================================================================
# WOPP EXPERIMENT CONFIGURATION
# ==============================================================================

# 1. Experiment Parameters (V(10, 20) case)
N_AMBIENT = 20  # n: Ambient dimension
K_RANK = 10  # k: Rank
NUM_PARTICLES = 50  # N: Particles used for convergence analysis
SIGMA_VAL = 0.1  # sigma: Noise coefficient specified
# T_VAL = 500.0  # T: Asymptotic time used
T_VAL = 500.0
DT_VAL = 0.01  # dt: Standard time step
NUM_TRIALS = 8*13  # Standardizing to 100 trials

# 2. Success Tolerance
SUCCESS_TOLERANCE = 1.0

# 3. Solver Parameter Dictionaries (Must match C++ constructor args)
KIM_SOLVER_PARAMS = {
    'beta_val': 5000.0,
    'lambda_val': 1.0,
}

COR_SOLVER_PARAMS = {
    'beta_val': 5000.0,
    'lambda_initial': 1.0,
    'lambda_min': 0.1,
    'lambda_max': 5.0,
    'lambda_increase_factor': 1.1,
    'lambda_decrease_factor': 0.98,
    'lambda_adapt_interval': 50,
    'lambda_stagnation_thresh': 0.005,
    'lambda_convergence_thresh': 0.3,
    'sigma_initial': 0.5,
    'sigma_final': 0.001,
    'sigma_max': 1.0,
    'annealing_rate': 10.0,
    'reheat_check_interval': 50_000,
    'reheat_window': 100_000,
    'reheat_threshold': 1e-4,
    'reheat_sigma_boost': 0.2,
    'reheat_lambda_reset': True,
}

# 4. Objective Parameters
WOPP_OBJ_PARAMS = {}  # No explicit parameters, relies on random generation


# ==============================================================================
# MAIN EXECUTION FUNCTION
# ==============================================================================

def setup_and_run_wopp_experiment(solver_class, backend):
    """
    Sets up and runs the single WOPP experiment case with the selected solver/backend.
    """

    # Select and configure base solver parameters
    if solver_class == cbo_module.KimCBOSolver:
        solver_params = KIM_SOLVER_PARAMS.copy()
        solver_params['sigma_val'] = SIGMA_VAL  # Kim uses 'sigma_val'
    elif solver_class == cbo_module.CormacsCBOSolver:
        solver_params = COR_SOLVER_PARAMS.copy()
        solver_params['sigma_initial'] = SIGMA_VAL  # Cormac uses 'sigma_initial'
    else:
        raise ValueError(f"Unknown solver class: {solver_class.__name__}")

    # Add the common backend and run parameters
    solver_params['backend'] = backend

    OBJECTIVE_NAME_CPP = "WOPP"  # String identifier for C++ objective

    print("=" * 60)
    print(f"--- STARTING WOPP SUITE: {solver_class.__name__} (C++) ---")
    print(f"--- Target V(k,n): V({K_RANK}, {N_AMBIENT}) ---")
    print(f"--- ACCELERATION BACKEND: {backend.upper()} ---")
    print("=" * 60)

    # --- Create and Run ExperimentRunner ---
    runner = ExperimentRunner(
        solver_class=solver_class,
        objective_name=OBJECTIVE_NAME_CPP,
        n=N_AMBIENT,
        k=K_RANK,
        num_particles=NUM_PARTICLES,
        T=T_VAL,
        dt=DT_VAL,
        trials=NUM_TRIALS,
        success_tolerance=SUCCESS_TOLERANCE,
        solver_params=solver_params,
        objective_params=WOPP_OBJ_PARAMS
    )

    runner.run_trials_parallel()

    return runner


if __name__ == '__main__':

    # --- 1. CHOOSE SOLVER AND BACKEND ---
    # Uncomment the configuration you wish to run:

    # Configuration A: Fixed Parameters (Kim's Solver)
    SOLVER_TO_RUN = cbo_module.KimCBOSolver

    # Configuration B: Adaptive Parameters (Cormac's Solver)
    # SOLVER_TO_RUN = cbo_module.CormacsCBOSolver

    ACCELERATION_BACKEND = 'eigen'  # Use 'torch' for MPS SVD

    # --- 2. EXECUTE SUITE ---

    wopp_runner = setup_and_run_wopp_experiment(SOLVER_TO_RUN, ACCELERATION_BACKEND)

    # --- 3. PLOT RESULTS ---
    time_pts, dynamics, asymptotics, obj_instance = wopp_runner.get_results_for_plotting()

    if time_pts is not None and obj_instance:
        plot_cbo_results(
            "wopp",
            time_pts,
            dynamics,
            asymptotics,
            obj_instance,
            SOLVER_TO_RUN.__name__,
            n=N_AMBIENT,
            k=K_RANK
        )
        plot_cbo_results("wopp", time_pts, dynamics, asymptotics, obj_instance, SOLVER_TO_RUN.__name__, k=K_RANK,
                         n=N_AMBIENT)

    print("\n--- WOPP experiment suite execution finished. ---")