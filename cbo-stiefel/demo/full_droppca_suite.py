# filename: full_droppca_suite.py

import numpy as np
import os
import multiprocessing
import sys

# --- Import C++ Bindings ---
# We assume cbo_module is compiled and in the Python path
# 1. Get the current directory (demo/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory (the project root, which contains cbo_module.so)
parent_dir = os.path.join(current_dir, '..')

# 3. Insert the parent directory path to Python's module search path
sys.path.insert(0, parent_dir)
import cbo_module

# --- Import Experiment Utilities ---
# We assume these .py files are in the same directory or Python path
import experiment_runner
import visualisation


def run_droppca_experiment():
    """
    Configures and runs a CBO experiment for the
    Distributionally Robust PCA (DRO-PCA) objective.
    """

    # --- Experiment Configuration ---

    # 1. Manifold & Objective Parameters
    n = 1000  # Dimension of the space (d in the paper)
    k = 50  # Number of principal components (r in the paper)

    # Create a nominal covariance matrix Sigma_0 (n x n, symmetric positive-semidefinite)
    # We can generate one randomly for this test.
    np.random.seed(42)
    A = np.random.rand(n, n)
    Sigma_0 = A.T @ A

    # Parameters from the paper's experiments
    rho = 1.0  # Wasserstein radius (rho)
    gamma = 0.05  # l1-norm regularization (gamma)

    objective_params = {
        "nominal_covariance": Sigma_0,
        "rho": rho,
        "gamma": gamma
    }

    # 2. Solver Configuration
    solver_class = cbo_module.CormacsCBOSolver
    solver_name = "CormacsCBOSolver"  # For plotting
    solver_params = {
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
        'annealing_rate': 3,
        'reheat_check_interval': 5000000,
        'reheat_window': 50000 * 2,
        'reheat_threshold': 1e-4,
        'reheat_sigma_boost': 0.002,
        'reheat_lambda_reset': True,
    }

    # 3. Runner Configuration
    T = 20.0  # Total simulation time
    dt = 0.5  # Time step
    num_particles = 30
    trials = 100  # Run 50 independent trials

    # The true minimum is 0.0 (as we specified in the C++ class)
    success_tolerance = 1e-5

    # 4. Output Configuration
    SAVE_DIR = "droppca"
    # Ensure the save directory exists
    os.makedirs(f"saves/{SAVE_DIR}", exist_ok=True)

    print("=" * 80)
    print(f"--- Starting Tessera DRO-PCA Experiment ---")
    print(f"Objective: DroPcaObjective on V({k},{n})")
    print(f"Solver:    {solver_name}")
    print(f"Params:    rho={rho}, gamma={gamma}")
    print(f"Trials:    {trials} @ T={T}, dt={dt}")
    print("=" * 80)

    # --- Instantiate and Run Experiment ---

    runner = experiment_runner.ExperimentRunner(
        solver_class=solver_class,
        objective_name="DroPca",  # Must match the string in the runner
        n=n, k=k,
        num_particles=num_particles,
        T=T, dt=dt, trials=trials,
        success_tolerance=success_tolerance,
        solver_params=solver_params,
        objective_params=objective_params
    )

    # Run all trials in parallel
    runner.run_trials_parallel()

    # --- Collect and Plot Results ---

    time_points, all_f_M_dynamics, asymptotic_values, obj_instance = \
        runner.get_results_for_plotting()

    print(f"\nExperiment complete. Generating plots in 'saves/{SAVE_DIR}'...")

    visualisation.plot_cbo_results(
        save_dir=SAVE_DIR,
        time_points=time_points,
        all_f_M_dynamics=all_f_M_dynamics,
        asymptotic_values=asymptotic_values,
        objective_instance=obj_instance,
        solver_name=solver_name,
        k=k, n=n
    )

    print("--- DRO-PCA Experiment Finished ---")


if __name__ == "__main__":
    # Set multiprocessing start method to "spawn"
    # This is crucial for cross-platform compatibility (especially macOS)
    # when passing complex C++ objects between processes.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        # Might already be set, or not the main process
        print(f"Note: Could not set multiprocessing start method: {e}")

    run_droppca_experiment()