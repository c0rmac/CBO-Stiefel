# filename: full_qap_suite.py
import os

import numpy as np
import time
import sys
import multiprocessing

# --- Set start method early (Crucial for C++ extensions) ---
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# --- Imports (AFTER setting start method) ---
# C++ Kernel: Contains QAPObjective and Solvers
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
# QAP EXPERIMENT CONFIGURATION
# ==============================================================================

# 1. Experiment Parameters (Based on paper's example, Section 8.1.2)
N_DIM = 10  # n and k (QAP is on V(n,n))
N_AMBIENT = N_DIM  # n = 10
K_RANK = N_DIM  # k = 10
NUM_PARTICLES = 50  # N: Particles used
SIGMA_VAL = 0.19  # sigma: Noise coefficient specified
T_VAL = 500.0  # T: Asymptotic time used [cite: 1473]
DT_VAL = 0.05  # dt: Standard time step
NUM_TRIALS = 8*13  # Standardizing to 100 trials

# 2. Success Tolerance (Based on 1% of the typical range of the problem)
# Paper's calculated range for specific A, B was [-14.2861, 28.49].
# Tolerance = 0.4278. We set the max acceptable final f(M) value.
# Max f(M) for Success = -14.2861 + 0.4278 = -13.8583
SUCCESS_TOLERANCE = -13.8583

# 3. Solver Parameter Dictionaries
KIM_SOLVER_PARAMS = {
    'beta_val': 5000.0,
    'lambda_val': 1.0,
    # 'sigma_val' will be added later
}

COR_SOLVER_PARAMS = {
    'beta_val': 5000,
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
    'annealing_rate': 10,
    'reheat_check_interval': 5000000,
    'reheat_window': 50000 * 2,
    'reheat_threshold': 1e-4,
    'reheat_sigma_boost': 0.002,
    'reheat_lambda_reset': True,
}

# 4. Objective Parameters
QAP_OBJ_PARAMS = {}  # A and B matrices are randomly generated internally


# ==============================================================================
# MAIN EXECUTION FUNCTION
# ==============================================================================

def setup_and_run_qap_experiment(solver_class, backend):
    """
    Sets up and runs the single QAP experiment case with the selected solver/backend.
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

    OBJECTIVE_NAME_CPP = "QAP"  # String identifier for C++ objective

    print("=" * 60)
    print(f"--- STARTING QAP SUITE: {solver_class.__name__} (C++) ---")
    print(f"--- Target V(n,n): V({K_RANK}, {N_AMBIENT}) ---")
    print(f"--- ACCELERATION BACKEND: {backend.upper()} ---")
    print("=" * 60)

    # --- Create and Run ExperimentRunner ---
    # The runner will instantiate cbo_module.QAPObjective inside the worker processes
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
        objective_params=QAP_OBJ_PARAMS
    )

    runner.run_trials_parallel()

    return runner


if __name__ == '__main__':

    # --- 1. CHOOSE SOLVER AND BACKEND ---
    # Set the desired solver here
    # SOLVER_TO_RUN = cbo_module.KimCBOSolver
    SOLVER_TO_RUN = cbo_module.CormacsCBOSolver

    ACCELERATION_BACKEND = 'eigen'

    # --- 2. EXECUTE SUITE ---

    qap_runner = setup_and_run_qap_experiment(SOLVER_TO_RUN, ACCELERATION_BACKEND)

    # --- 3. PLOT RESULTS ---
    time_pts, dynamics, asymptotics, obj_instance = qap_runner.get_results_for_plotting()

    if time_pts is not None and obj_instance:
        plot_cbo_results(
            "qap",
            time_pts,
            dynamics,
            asymptotics,
            obj_instance,
            SOLVER_TO_RUN.__name__,
            n=N_AMBIENT,
            k=K_RANK
        )
        plot_cbo_results("qap", time_pts, dynamics, asymptotics, obj_instance, SOLVER_TO_RUN.__name__, k=K_RANK, n=N_AMBIENT)

    print("\n--- QAP experiment suite execution finished. ---")