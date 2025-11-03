# filename: run_asset_experiment.py

import numpy as np
import pandas as pd
import os
import multiprocessing
import sys

# --- Import C++ Bindings ---
# 1. Get the current directory (demo/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory (the project root, which contains cbo_module.so)
parent_dir = os.path.join(current_dir, '..')

# 3. Insert the parent directory path to Python's module search path
sys.path.insert(0, parent_dir)
import cbo_module

# --- Import Experiment Utilities ---
import experiment_runner
import visualisation


def calculate_heuristic_rho(csv_file_path: str, heuristic_rho: float = 2):
    """
    Calculates the heuristic for the robustness radius rho (ρ)
    based on the number of samples (n) in the data file.

    The heuristic is derived from the paper's experimental setup:
    ρ = 5 * n^(-1/2)
    """

    print("--- Calculating Heuristic Rho (ρ) ---")

    # --- 1. Load Data to find 'n' ---
    try:
        returns_df = pd.read_csv(csv_file_path)

        # 'n' is the number of samples (rows/observations)
        n = len(returns_df)

        if n == 0:
            print(f"Error: The file '{csv_file_path}' is empty.")
            return

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # --- 2. Calculate Rho ---

    # Apply the formula: rho = 5 / sqrt(n)
    rho_heuristic = heuristic_rho / np.sqrt(n)

    return rho_heuristic

def run_droppca_on_assets():
    """
    Configures and runs a CBO experiment for the
    Distributionally Robust PCA (DRO-PCA) objective
    using a nominal covariance matrix derived from 'asset_returns.csv'.
    """

    # --- Experiment Configuration ---

    # 1. Manifold & Objective Parameters
    print("--- Loading Data ---")
    try:
        returns_df = pd.read_csv("asset_returns.csv")
        numerical_returns_df = returns_df.iloc[:, 1:]  # Remove first (Date) column
        Sigma_0_df = numerical_returns_df.cov()
        Sigma_0 = Sigma_0_df.to_numpy()

        n = Sigma_0.shape[0]
        k = 10

        if n < k:
            print(f"Warning: Number of assets ({n}) is less than k ({k}). Adjusting k to {max(1, n - 1)}")
            k = max(1, n - 1)

        print(f"Successfully calculated {n}x{n} covariance matrix from {len(returns_df)} observations.")

    except FileNotFoundError:
        print(f"Error: 'asset_returns.csv' not found.")
        return
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return

    rho = calculate_heuristic_rho("asset_returns.csv")
    gamma = 0.005

    objective_params = {
        "nominal_covariance": Sigma_0,
        "rho": rho,
        "gamma": gamma
    }

    # 2. Solver Configuration
    solver_class = cbo_module.CormacsCBOSolver
    solver_name = "CormacsCBOSolver"
    solver_params = {
        'beta_val': 10000,
        'lambda_initial': 1,
        'lambda_min': 0.1,
        'lambda_max': 5.0,
        'lambda_increase_factor': 1.1,
        'lambda_decrease_factor': 0.98,
        'lambda_adapt_interval': 50,
        'lambda_stagnation_thresh': 0.005,
        'lambda_convergence_thresh': 0.3,
        'sigma_initial': 0.25,
        'sigma_final': 0.0,
        'sigma_max': 1.0,
        'annealing_rate': 4,
        'reheat_check_interval': 5000000,
        'reheat_window': 50000 * 2,
        'reheat_threshold': 1e-4,
        'reheat_sigma_boost': 0.002,
        'reheat_lambda_reset': True,
    }

    # 3. Runner Configuration
    T = 500.0
    dt = 0.05
    num_particles = 56
    trials = 56
    success_tolerance = 1e-5

    # 4. Output Configuration
    SAVE_DIR = "asset_returns_experiment"
    os.makedirs(f"saves/{SAVE_DIR}", exist_ok=True)

    print("=" * 80)
    print(f"--- Starting Tessera DRO-PCA Experiment (Asset Data) ---")
    print(f"Objective: DroPcaObjective on V({k},{n})")
    print(f"Solver:    {solver_name}")
    print(f"Params:    rho={rho}, gamma={gamma}")
    print(f"Trials:    {trials} @ T={T}, dt={dt}")
    print("=" * 80)

    # --- Instantiate and Run Experiment ---

    runner = experiment_runner.ExperimentRunner(
        solver_class=solver_class,
        objective_name="DroPca",
        n=n, k=k,
        num_particles=num_particles,
        T=T, dt=dt, trials=trials,
        success_tolerance=success_tolerance,
        solver_params=solver_params,
        objective_params=objective_params
    )

    runner.run_trials_parallel()

    # --- 🌟 NEW: Get and Save the Best Minimizer X 🌟 ---

    best_X = runner.get_best_minimizer()

    if best_X is not None:
        save_path = f"saves/{SAVE_DIR}/best_minimizer_X.npy"
        np.save(save_path, best_X)
        print(f"🏆 Optimal minimizer X (the 'U' value) saved to: {save_path}")
    else:
        print(f"⚠️ Could not save minimizer X (not returned by C++ solver).")

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

    print("--- Asset Data Experiment Finished ---")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        print(f"Note: Could not set multiprocessing start method: {e}")

    run_droppca_on_assets()