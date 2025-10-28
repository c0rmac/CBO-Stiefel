# filename: wopp_experiment.py

import numpy as np
import sys
import time

# --- Set up Imports ---
sys.path.append('.')
try:
    from ackley_experiment_parallel import ExperimentRunner  # Import runner
    from wopp import WOPPObjective  # Import objective
    from kim_solver import KimCBOSolver  # Import fixed-parameter solver
    from visualisation import plot_cbo_results  # Import plotting
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure dependency files are correctly named and located.")
    sys.exit(1)


# ==============================================================================
# WOPP EXPERIMENT SETUP FUNCTION
# ==============================================================================

def run_wopp_experiment(solver_class):
    """
    Sets up and runs the high-dimensional WOPP experiment V(10, 20).
    Uses parameters from the paper's section 8.1.3.
    """

    # --- Experiment Configuration ---
    N_AMBIENT = 20  # n: Ambient dimension (rows)
    K_RANK = 10  # k: Rank (columns)
    NUM_PARTICLES = 50  # N: Standard N used in the paper's comparisons
    SIGMA_VAL = 0.1  # sigma: Noise coefficient specified [cite: 1500]
    T_VAL = 500.0  # T: Asymptotic time used [cite: 1501]
    DT_VAL = 0.01  # dt: Standard time step
    NUM_TRIALS = 100  # Standardizing to 100 trials

    # --- CBO Solver Parameters (Kim's fixed parameters used in the paper) ---
    BETA_VAL = 5000.0  # beta: Assumed high value for strong consensus
    LAMBDA_VAL = 1.0  # lambda: Assumed standard value

    # --- Success Tolerance ---
    # The paper approximated Max f(X) ~ 343.1688 and used a tolerance of 0.01 * Range
    # 0.01 * (343.1688 - 0) = 3.43. A simpler tight check is needed.
    # Since all trials attained f(M) < 1[cite: 1502], we set a tight tolerance.
    SUCCESS_TOLERANCE = 1.0  # SUCCESS if final f(M) < 1.0 [cite: 1502]

    # --- Solver Params Dictionary ---
    solver_params = {
        'beta': BETA_VAL,
        'lambda_': LAMBDA_VAL,
        'sigma': SIGMA_VAL,
    }

    print("=" * 60)
    print(f"--- Setting up WOPP Experiment: V({K_RANK}, {N_AMBIENT}) ---")
    print("NOTE: WOPP Objective requires regenerating A, B, C for each trial, which is handled inside the runner.")
    print("=" * 60)

    # The WOPPObjective __init__ generates random A, B, C *once* here.
    # We must ensure the ExperimentRunner re-initializes the objective for every worker.

    # Create the ExperimentRunner instance
    runner = ExperimentRunner(
        solver_class=solver_class,
        objective_class=WOPPObjective,
        n=N_AMBIENT,
        k=K_RANK,
        num_particles=NUM_PARTICLES,
        T=T_VAL,
        dt=DT_VAL,
        trials=NUM_TRIALS,
        success_tolerance=SUCCESS_TOLERANCE,
        solver_params=solver_params,
        objective_params={}  # No fixed params passed, uses defaults/random generation
    )

    # Run the parallel trials
    runner.run_trials_parallel()

    return runner


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # Use KimCBOSolver as it corresponds to the fixed-parameter method used in the paper
    SOLVER_TO_USE = KimCBOSolver

    wopp_runner = run_wopp_experiment(SOLVER_TO_USE)

    # Plot results
    time_pts, dyn_data, async_data = wopp_runner.get_results_for_plotting()
    plot_cbo_results(
        time_pts,
        dyn_data,
        async_data,
        wopp_runner.objective_instance,
        SOLVER_TO_USE.__name__
    )

    print("\n--- WOPP experiment complete. ---")