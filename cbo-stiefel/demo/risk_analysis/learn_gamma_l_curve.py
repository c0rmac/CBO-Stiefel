# filename: learn_gamma_l_curve.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
import sys
import time

# --- Import Tessera Project Modules ---
try:
    # 1. Get the current directory (demo/)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Get the parent directory (the project root, which contains cbo_module.so)
    parent_dir = os.path.join(current_dir, '..')

    # 3. Insert the parent directory path to Python's module search path
    sys.path.insert(0, parent_dir)
    import cbo_module
    import experiment_runner
except ImportError as e:
    print(f"Error: Could not import cbo_module or experiment_runner: {e}")
    print("Please ensure cbo_module is compiled and all .py files are accessible.")
    sys.exit(1)


def calculate_nominal_cost(X, Sigma_0):
    """
    Calculates the "Nominal Cost" g(X) = tr((I - XX^T) * Sigma_0)
    """
    if X is None:
        return np.nan
    n = Sigma_0.shape[0]
    I_n = np.eye(n)
    P = I_n - (X @ X.T)
    nominal_cost = np.trace(P @ Sigma_0)
    return nominal_cost


def calculate_robust_cost(X, Sigma_0, rho):
    """
    Calculates the Robust Cost (the objective function *without*
    the sparsity penalty): g(X) + r(X)
    """
    g_X = calculate_nominal_cost(X, Sigma_0)
    if np.isnan(g_X):
        return np.nan

    # Per the paper, r(X) = 2 * rho * sqrt(g(X))
    # We must ensure g(X) is non-negative for the sqrt
    g_X_non_negative = np.array([0.0, g_X]).max()
    r_X = 2 * rho * np.sqrt(g_X_non_negative)

    # The cost we are plotting is the fit, which is g(X) + r(X)
    return g_X + r_X


def run_l_curve_experiment_for_gamma():
    """
    Runs the full CBO experiment for a range of gamma values
    to generate an L-curve plot and find the optimal gamma.
    """

    print("--- Starting L-Curve Experiment for Sparsity Hyperparameter (γ) ---")

    # --- 1. Load Data and get Constants ---
    try:
        returns_df = pd.read_csv("asset_returns.csv")
        numerical_returns_df = returns_df.iloc[:, 1:]  # Drop Date column
        Sigma_0 = numerical_returns_df.cov().to_numpy()

        n_samples = len(numerical_returns_df)
        n_features = Sigma_0.shape[0]
        k_factors = 5  # Fix k=5

        # This is our *known optimal* proportional factor
        C_val = 1.2
        n_inv_sqrt = 1.0 / np.sqrt(n_samples)
        rho_heuristic = C_val * n_inv_sqrt

        print(f"Loaded {n_features}x{n_features} covariance matrix (n={n_samples} samples).")
        print(f"Using fixed optimal C = {C_val:.1f} (ρ = {rho_heuristic:.5f})")

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # --- 2. Define Experiment Parameters ---

    # We will search for gamma on a logarithmic scale
    # from 0.0001 (very dense) to 0.1 (very sparse)
    gamma_list = np.logspace(-4, -2, 10)
    # This list will be [0.0001, 0.0002, 0.0005, ..., 0.046, 0.1]

    l_curve_points = []  # To store (x, y) tuples for our plot

    # Fixed parameters for all runs
    base_solver_class = cbo_module.CormacsCBOSolver
    base_solver_params = {
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

    # --- 3. Run the Experiment Loop ---

    total_start_time = time.time()

    for i, gamma_val in enumerate(gamma_list):

        print("\n" + "=" * 80)
        print(f"STARTING RUN {i + 1}/{len(gamma_list)}:  Testing gamma = {gamma_val:.6f}")
        print("=" * 80)

        # Set the objective parameters for this specific run
        objective_params = {
            "nominal_covariance": Sigma_0,
            "rho": rho_heuristic,  # <-- This is now fixed
            "gamma": gamma_val  # <-- This is the knob we are turning
        }

        # Instantiate and run the full experiment
        runner = experiment_runner.ExperimentRunner(
            solver_class=base_solver_class,
            objective_name="DroPca",
            n=n_features, k=k_factors,
            num_particles=50,
            T=500.0, dt=0.05, trials=8,
            success_tolerance=1e-5,
            solver_params=base_solver_params,
            objective_params=objective_params
        )

        runner.run_trials_parallel()

        # Get the best matrix X found during this experiment
        best_X = runner.get_best_minimizer()

        if best_X is not None:
            # X-Axis: Sparsity (L1-Norm)
            x_val = np.sum(np.abs(best_X))

            # Y-Axis: Robust Cost (Fit)
            y_val = calculate_robust_cost(best_X, Sigma_0, rho_heuristic)

            print(f"--- Result for γ = {gamma_val:.6f}: L1-Norm={x_val:.4f}, RobustCost={y_val:.6f} ---")
            l_curve_points.append((x_val, y_val))
        else:
            print(f"--- FAILED RUN for γ = {gamma_val:.6f}: No minimizer found. ---")
            l_curve_points.append((np.nan, np.nan))

    total_end_time = time.time()
    print("\n" + "=" * 80)
    print(f"L-Curve Experiment Finished. Total time: {(total_end_time - total_start_time) / 60:.2f} minutes")
    print("=" * 80)

    # --- 4. Plot the L-Curve ---

    # Filter out any failed (NaN) runs
    valid_points = [(x, y) for x, y in l_curve_points if not (np.isnan(x) or np.isnan(y))]
    valid_gammas = [g for g, (x, y) in zip(gamma_list, l_curve_points) if not (np.isnan(x) or np.isnan(y))]

    if not valid_points:
        print("No valid data to plot. All runs may have failed.")
        return

    x_data = [p[0] for p in valid_points]
    y_data = [p[1] for p in valid_points]

    print(f"Plotting L-Curve with the following data:")
    print(f"Gamma values: {[f'{g:.1e}' for g in valid_gammas]}")
    print(f"L1-Norms (x-axis):   {[f'{x:.2f}' for x in x_data]}")
    print(f"Robust Costs (y-axis): {[f'{y:.5f}' for y in y_data]}")

    plt.figure(figsize=(12, 8))
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='b')

    # Annotate each point with its gamma value
    for i, gamma_val in enumerate(valid_gammas):
        plt.annotate(f"γ = {gamma_val:.1e}",
                     (x_data[i], y_data[i]),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha='left')

    # Highlight your previous "sweet spot" of 0.005
    # We find the gamma in our list that is *closest* to 0.005
    if 0.005 in gamma_list:
        idx = gamma_list.index(0.005)
        if not np.isnan(l_curve_points[idx][0]):
            plt.plot(l_curve_points[idx][0], l_curve_points[idx][1], marker='*', markersize=15,
                     color='red', label=f"Previous Sweet Spot (γ=0.005)")

    plt.title('Tessera: L-Curve for Sparsity Hyperparameter (γ)')
    plt.xlabel('Sparsity (L1-Norm of X) -- (Lower is Sparser)')
    plt.ylabel('Robust Cost g(X) + r(X) -- (Lower is Better Fit)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save the plot
    save_path = "l_curve_gamma_analysis.png"
    plt.savefig(save_path)
    print(f"\nL-Curve plot saved to: {save_path}")


# --- Main execution ---
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        print(f"Note: Could not set multiprocessing start method: {e}")

    run_l_curve_experiment_for_gamma()