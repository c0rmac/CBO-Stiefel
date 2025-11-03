# filename: full_rotation_averaging_suite.py
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
# C++ Kernel: Contains RotationAveragingObjective and Solvers
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
# DATA GENERATION FOR ROTATION AVERAGING
# ==============================================================================

def generate_so_d_matrix(d):
    """Helper function to generate a random d x d Special Orthogonal matrix (det= +1)."""
    A = np.random.randn(d, d)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1  # Flip one column to ensure determinant is +1
    return Q


def create_rotation_averaging_data(d, num_measurements, noise_level=0.1):
    """
    Generates a dataset for the simplified Rotation Averaging problem.
    """
    # 1. Create the secret true rotation X_true
    X_true = generate_so_d_matrix(d)

    references_X_star = []
    measurements_Z = []

    for _ in range(num_measurements):
        # 2. Create a random reference rotation
        X_j_star = generate_so_d_matrix(d)
        references_X_star.append(X_j_star)

        # 3. Create the perfect measurement Z_j = X_true^T @ X_j*
        Z_j_perfect = X_true.T @ X_j_star

        # 4. Add some noise
        Noise = np.random.randn(d, d) * noise_level
        Z_j_noisy = Z_j_perfect + Noise
        measurements_Z.append(Z_j_noisy)

    return measurements_Z, references_X_star, X_true


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

# 1. Experiment Parameters
D_DIMENSION = 3  # d: Dimension of the rotation group SO(d) (e.g., SO(3))
NUM_MEASUREMENTS = 20  # Number of (Z_j, X_j*) pairs in the dataset
NUM_PARTICLES = 50  # N: Number of particles
SIGMA_VAL = 0.5  # sigma: Noise coefficient (needs tuning)
T_VAL = 100.0  # T: Total simulation time
DT_VAL = 0.01  # dt: Time step
NUM_TRIALS = 100  # Number of independent trials (to see if it finds X_true)

# 2. Success Tolerance
# Since true_minimum is noise-dependent (NaN), we can't use it.
# We'll set a high tolerance just to let the runner complete.
SUCCESS_TOLERANCE_PLACEHOLDER = 1000.0

# 3. Solver Parameter Dictionaries
KIM_SOLVER_PARAMS = {
    'beta_val': 100.0,
    'lambda_val': 1.0,
}

COR_SOLVER_PARAMS = {
    'beta_val': 100,
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


# ==============================================================================
# MAIN EXECUTION FUNCTION
# ==============================================================================

def setup_and_run_rotation_averaging_experiment(solver_class, backend):
    """
    Sets up and runs the Rotation Averaging experiment case SO(d).
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
    solver_params['enforce_SOd'] = True

    OBJECTIVE_NAME_CPP = "RotationAveraging"  # String identifier for C++ objective

    print("=" * 60)
    print(f"--- STARTING Rotation Averaging SUITE: {solver_class.__name__} (C++) ---")
    print(f"--- Target SO(d) = SO({D_DIMENSION}) ---")
    print(f"--- ACCELERATION BACKEND: {backend.upper()} ---")
    print("=" * 60)

    # --- Generate the Dataset for this Run ---
    print(f"Generating dataset with {NUM_MEASUREMENTS} measurements...")
    Z_data, X_star_data, X_true = create_rotation_averaging_data(D_DIMENSION, NUM_MEASUREMENTS)

    # 4. Objective Parameters (Pass the generated data lists)
    ROT_AVG_OBJ_PARAMS = {
        'measurements_Z': Z_data,
        'references_X_star': X_star_data
    }

    # --- Create and Run ExperimentRunner ---
    runner = ExperimentRunner(
        solver_class=solver_class,
        objective_name=OBJECTIVE_NAME_CPP,
        n=D_DIMENSION,
        k=D_DIMENSION,
        num_particles=NUM_PARTICLES,
        T=T_VAL,
        dt=DT_VAL,
        trials=NUM_TRIALS,
        success_tolerance=SUCCESS_TOLERANCE_PLACEHOLDER,
        solver_params=solver_params,
        objective_params=ROT_AVG_OBJ_PARAMS
    )

    runner.run_trials_parallel()

    # We set the "true minimum" to NaN as it's unknown (noise-dependent)
    runner.true_minimum = np.nan

    return runner


if __name__ == '__main__':

    # --- 1. CHOOSE SOLVER AND BACKEND ---
    # SOLVER_TO_RUN = cbo_module.KimCBOSolver
    SOLVER_TO_RUN = cbo_module.CormacsCBOSolver

    ACCELERATION_BACKEND = 'eigen'

    # --- 2. EXECUTE SUITE ---

    rot_avg_runner = setup_and_run_rotation_averaging_experiment(SOLVER_TO_RUN, ACCELERATION_BACKEND)

    # --- 3. PLOT RESULTS ---
    time_pts, dynamics, asymptotics, obj_instance = rot_avg_runner.get_results_for_plotting()

    if time_pts is not None and obj_instance:
        print("\nPlotting results...")
        plot_cbo_results(
            "rotation_averaging",
            time_pts,
            dynamics,
            asymptotics,
            obj_instance,
            SOLVER_TO_RUN.__name__,
            n=D_DIMENSION,
            k=D_DIMENSION
        )

    print("\n--- Rotation Averaging experiment suite execution finished. ---")