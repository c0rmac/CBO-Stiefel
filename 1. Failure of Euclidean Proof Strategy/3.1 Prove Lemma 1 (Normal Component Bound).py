import numpy as np
import time
# Assuming stiefel_utils.py contains project_stiefel and projection_tangent_space
from stiefel_utils import project_stiefel, projection_tangent_space

def validate_lemma1(n_dim, k_dim, num_runs_check, X_star):
    """
    Numerically validates Lemma 1 at t=0.

    Args:
        n_dim (int): Ambient dimension.
        k_dim (int): Number of columns/frames.
        num_runs_check (int): Number of particles for averaging.
        X_star (np.ndarray): Target minimizer on V(n, k).

    Returns:
        dict: Containing Avg_N2, Avg_V, and a boolean indicating if the check passed.
    """
    print(f"\n--- Numerical Check for Lemma 1 ---")
    print(f"Parameters: V({n_dim}, {k_dim})")
    print(f"Number of particles for averaging: {num_runs_check}")

    # Ensemble at time t=0
    particles_t0 = []
    sum_V = 0.0
    sum_N2 = 0.0
    start_time = time.time()

    for i in range(num_runs_check):
        # Initialize particle
        np.random.seed(i) # Use index for reproducibility
        X0_pre = np.random.randn(n_dim, k_dim)
        X_i_t0 = project_stiefel(X0_pre)
        particles_t0.append(X_i_t0)

        # Calculate error vector
        Y_i = X_i_t0 - X_star

        # Calculate V contribution
        V_i = 0.5 * np.linalg.norm(Y_i, 'fro')**2
        sum_V += V_i

        # Calculate Normal Component N_X(Y) = 0.5 * (X @ Y.T @ X + X @ X.T @ Y)
        term1_N = X_i_t0 @ Y_i.T @ X_i_t0
        term2_N = X_i_t0 @ X_i_t0.T @ Y_i
        N_i = 0.5 * (term1_N + term2_N)

        # Calculate squared norm of Normal Component
        N2_i = np.linalg.norm(N_i, 'fro')**2
        sum_N2 += N2_i

    end_time = time.time()
    np.random.seed(None) # Reset seed

    # Calculate Averages
    Avg_V = sum_V / num_runs_check
    Avg_N2 = sum_N2 / num_runs_check

    print(f"Calculation time: {end_time - start_time:.2f} seconds.")
    print(f"Average V           (Estimates V(rho_0))  : {Avg_V:.8f}")
    print(f"Average ||N(X-X*)||^2 (Estimates E[||N||^2]): {Avg_N2:.8f}")
    print(f"2 * Average V                             : {2.0 * Avg_V:.8f}")

    # Check Inequality
    check_passed = Avg_N2 <= (2.0 * Avg_V + 1e-9) # Add small tolerance for floating point
    print(f"\nCheck Passed (Avg ||N||^2 <= 2 * Avg V)? : {check_passed}")

    validation_result = {
        "Avg_V": Avg_V,
        "Avg_N2": Avg_N2,
        "check_passed": check_passed
    }

    if not check_passed:
        print(f"⚠️ Inequality FAILED: Avg ||N||^2 ({Avg_N2:.4f}) > 2 * Avg V ({2.0*Avg_V:.4f})")
    else:
        print("✅ Inequality holds numerically.")

    return validation_result

# --- Example Usage ---
if __name__ == "__main__":
    # Parameters for the validation
    n_dim_val = 5
    k_dim_val = 3
    num_particles_val = 50000 # Use many particles for good averaging

    # Set up target
    np.random.seed(42)
    X_star_val_pre = np.random.randn(n_dim_val, k_dim_val)
    X_star_val = project_stiefel(X_star_val_pre)
    np.random.seed(None)

    # Run validation
    results = validate_lemma1(n_dim_val, k_dim_val, num_particles_val, X_star_val)