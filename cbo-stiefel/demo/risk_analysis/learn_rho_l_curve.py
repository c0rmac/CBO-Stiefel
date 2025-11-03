# filename: learn_C_l_curve.py

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
    This is the "price" of robustness.
    """
    if X is None:
        return np.nan

    n = Sigma_0.shape[0]
    I_n = np.eye(n)
    P = I_n - (X @ X.T)
    nominal_cost = np.trace(P @ Sigma_0)

    return nominal_cost


def run_l_curve_experiment_for_C():
    """
    Runs the full CBO experiment for a range of scaling constants C
    to generate an L-curve plot and find the optimal C.
    rho = C * n^(-1/2)
    """

    print("--- Starting L-Curve Experiment for Robustness Constant (C) ---")

    # --- 1. Load Data and get Constants ---
    try:
        returns_df = pd.read_csv("asset_returns.csv")
        numerical_returns_df = returns_df.iloc[:, 1:]  # Drop Date column
        Sigma_0 = numerical_returns_df.cov().to_numpy()

        n_samples = len(numerical_returns_df)
        n_features = Sigma_0.shape[0]
        k_factors = 10  # Use k=5

        # This is our known proportional factor
        n_inv_sqrt = 1.0 / np.sqrt(n_samples)

        print(f"Loaded {n_features}x{n_features} covariance matrix.")
        print(f"Found n = {n_samples} samples.")
        print(f"Proportional factor n^(-1/2) = {n_inv_sqrt:.5f}")

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # --- 2. Define Experiment Parameters ---

    # The list of CONSTANTS C to test
    # We are testing C, not rho.
    C_min = 0.0
    C_max = 8.0
    num_steps = 21  # 21 points gives 20 intervals

    C_list = np.linspace(C_min, C_max, num_steps)

    nominal_costs = []  # To store the y-axis of our plot

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
    base_gamma = 3.2e-03  # Your "sweet spot" gamma

    # --- 3. Run the Experiment Loop ---

    total_start_time = time.time()

    for i, C_val in enumerate(C_list):

        # Calculate the actual rho for this run
        rho_val = C_val * n_inv_sqrt

        print("\n" + "=" * 80)
        print(f"STARTING RUN {i + 1}/{len(C_list)}:  Testing C = {C_val:.2f} (ρ = {rho_val:.4f})")
        print("=" * 80)

        # Set the objective parameters for this specific run
        objective_params = {
            "nominal_covariance": Sigma_0,
            "rho": rho_val,  # <-- Use the calculated rho
            "gamma": base_gamma
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

        # Calculate the "price" of this robustness
        cost = calculate_nominal_cost(best_X, Sigma_0)

        if np.isnan(cost):
            print(f"--- FAILED RUN for C = {C_val:.2f}: No minimizer found. ---")
        else:
            print(f"--- Result for C = {C_val:.2f}: Nominal Cost g(X) = {cost:.6f} ---")

        nominal_costs.append(cost)

    total_end_time = time.time()
    print("\n" + "=" * 80)
    print(f"L-Curve Experiment Finished. Total time: {(total_end_time - total_start_time) / 60:.2f} minutes")
    print("=" * 80)

    # --- 4. Plot the L-Curve ---

    # Filter out any failed (NaN) runs
    valid_C_values = [r for r, c in zip(C_list, nominal_costs) if not np.isnan(c)]
    valid_costs = [c for c in nominal_costs if not np.isnan(c)]

    if not valid_costs:
        print("No valid data to plot. All runs may have failed.")
        return

    print(f"Plotting L-Curve with the following data:")
    print(f"Constant C (x-axis):   {[f'{r:.2f}' for r in valid_C_values]}")
    print(f"Nominal Costs (y-axis): {[f'{c:.5f}' for c in valid_costs]}")

    plt.figure(figsize=(10, 6))
    plt.plot(valid_C_values, valid_costs, marker='o', linestyle='-', color='b')

    # Highlight the paper's heuristic
    paper_heuristic_C = 5.0
    if paper_heuristic_C in valid_C_values:
        idx = valid_C_values.index(paper_heuristic_C)
        plt.plot(valid_C_values[idx], valid_costs[idx], marker='*', markersize=15,
                 color='red', label=f"Paper's Heuristic (C=5.0)")

    plt.title('Tessera: L-Curve for Robustness Constant (C)')
    plt.xlabel('Robustness Constant C (where ρ = C * n⁻¹/²)')
    plt.ylabel('Nominal Cost g(X) = tr((I - XX^T)Σ_0)\n(Higher = Worse fit to historical data)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save the plot
    save_path = "l_curve_C_analysis.png"
    plt.savefig(save_path)
    print(f"\nL-Curve plot saved to: {save_path}")


# --- Main execution ---
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        print(f"Note: Could not set multiprocessing start method: {e}")

    run_l_curve_experiment_for_C()