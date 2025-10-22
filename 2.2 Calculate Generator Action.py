# File: validate_generator.py

import numpy as np
import time
# Assuming stiefel_utils.py contains project_stiefel and projection_tangent_space
from stiefel_utils import project_stiefel, projection_tangent_space

# --- SDE Step Function ---
def stiefel_cbo_step(X_current, X_alpha, lambda_param, sigma_param, dt, n_dim, k_dim):
  """Performs one projected Euler-Maruyama step for CBO on V(n, k)."""
  C_nk_prime = n_dim - (k_dim + 1) / 2.0
  dist_alpha_sq = np.linalg.norm(X_current - X_alpha, 'fro')**2
  dist_alpha = np.sqrt(dist_alpha_sq) if dist_alpha_sq > 1e-16 else 0.0

  # Drift
  drift1 = +lambda_param * projection_tangent_space(X_current, X_alpha) # Corrected Sign
  drift2 = -(sigma_param**2 * C_nk_prime / 2.0) * dist_alpha_sq * X_current # Corrected Sign
  total_drift = drift1 + drift2

  # Diffusion
  diffusion_term = np.zeros_like(X_current)
  if sigma_param > 1e-15 and dist_alpha > 1e-15:
    dW = np.random.normal(0.0, np.sqrt(dt), size=(n_dim, k_dim))
    projected_noise = projection_tangent_space(X_current, dW)
    diffusion_term = sigma_param * dist_alpha * projected_noise

  # Euler Step & Projection
  X_predict = X_current + total_drift * dt + diffusion_term
  X_next = project_stiefel(X_predict)
  return X_next

# --- Main Validation Function ---
def validate_generator_action(n_dim, k_dim, lambda_param, sigma_param, dt_check, num_runs_check, X_star, X_alpha):
    """
    Numerically validates the theoretical generator action (Task 2.2).
    Args/Returns are the same as before...
    """
    print(f"\n--- Numerical Check for Task 2.2 ---")
    print(f"Parameters: lambda={lambda_param}, sigma={sigma_param}, dt_check={dt_check}")
    print(f"Number of particles for averaging: {num_runs_check}")

    # Ensemble at time t=0
    particles_t0 = []
    V_t0_sum = 0.0
    for i in range(num_runs_check):
        np.random.seed(i)
        X0_pre = np.random.randn(n_dim, k_dim)
        X_i_t0 = project_stiefel(X0_pre)
        particles_t0.append(X_i_t0)
        V_t0_sum += 0.5 * np.linalg.norm(X_i_t0 - X_star, 'fro')**2
    V_numerical_t0 = V_t0_sum / num_runs_check
    np.random.seed(None)
    print(f"V_numerical(t=0) = {V_numerical_t0:.12f}") # Increased precision for checking

    # --- Calculate Theoretical RHS at t=0 ---
    theoretical_sum = 0.0
    start_time_th = time.time()

    is_pure_drift_ideal = (sigma_param == 0.0 and np.allclose(X_alpha, X_star))

    if is_pure_drift_ideal:
        print("  Calculating theoretical RHS for pure drift (+ drift sign), idealized consensus...")
        for X_i_t0 in particles_t0:
            # Calculate lambda * < P_X(X - X*), X - X* >_F = lambda * ||P_X(X - X*)||^2
            projected_error = projection_tangent_space(X_i_t0, X_i_t0 - X_star)
            norm_sq_projected_error = np.linalg.norm(projected_error, 'fro')**2
            # The drift term should lead to a DECREASE, so the generator action is negative
            theoretical_sum += -lambda_param * norm_sq_projected_error # Corrected based on dV/dt = -lambda ||P_X(X-X*)||^2
    else:
        # General case using the full formula derived from Task 2.2, with corrected drift sign
        print("  Calculating theoretical RHS for general case (+ drift sign)...")
        C_nk_prime = n_dim - (k_dim + 1) / 2.0
        k_const = k_dim
        for X_i_t0 in particles_t0:
            dist_alpha_sq = np.linalg.norm(X_i_t0 - X_alpha, 'fro')**2

            # Term A components (Drift contribution to dV/dt)
            # <X - X*, +lambda P_X(X_alpha) - sigma^2*C'/2 * ||X-X_alpha||^2 * X>
            term_drift_part1 = +lambda_param * np.trace( (X_i_t0 - X_star).T @ projection_tangent_space(X_i_t0, X_alpha) )
            term_drift_part2 = -(sigma_param**2 * C_nk_prime / 2.0) * dist_alpha_sq * np.trace( (X_i_t0 - X_star).T @ X_i_t0 )
            term_A = term_drift_part1 + term_drift_part2

            # Term B (Diffusion contribution to dV/dt)
            term_B = (sigma_param**2 * k_const * C_nk_prime / 2.0) * dist_alpha_sq

            theoretical_sum += (term_A + term_B)

    end_time_th = time.time()
    dV_dt_theoretical = theoretical_sum / num_runs_check
    print(f"dV/dt (Theoretical RHS) = {dV_dt_theoretical:.8f} (Calc time: {end_time_th - start_time_th:.2f}s)")

    # --- Simulate One Step for Numerical Derivative ---
    particles_t1 = []
    V_t1_sum = 0.0
    start_time_num = time.time()
    for i in range(num_runs_check):
        X_i_t0 = particles_t0[i]
        # if sigma_param > 1e-15: np.random.seed(i * 1000) # Optional seeding

        X_i_t1 = stiefel_cbo_step(X_i_t0, X_alpha, lambda_param, sigma_param, dt_check, n_dim, k_dim)
        particles_t1.append(X_i_t1)
        V_t1_sum += 0.5 * np.linalg.norm(X_i_t1 - X_star, 'fro')**2
    end_time_num = time.time()
    V_numerical_t1 = V_t1_sum / num_runs_check
    np.random.seed(None)

    # --- Calculate Numerical Derivative & Check Precision ---
    delta_V_numerical = V_numerical_t1 - V_numerical_t0
    dV_dt_numerical = delta_V_numerical / dt_check
    relative_change = abs(delta_V_numerical) / abs(V_numerical_t0) if abs(V_numerical_t0) > 1e-15 else abs(delta_V_numerical)

    print(f"V_numerical(t=dt) = {V_numerical_t1:.12f}") # Increased precision
    print(f"Delta V_numerical = {delta_V_numerical:.6e}")
    print(f"Relative change (Delta V / V) = {relative_change:.6e}")
    print(f"dV/dt (Numerical Approx) = {dV_dt_numerical:.8f} (Sim time: {end_time_num - start_time_num:.2f}s)")

    # --- Precision Warning ---
    machine_epsilon = np.finfo(float).eps # Approx 2.2e-16 for float64
    if relative_change < 100 * machine_epsilon: # Warning threshold (adjust if needed)
        print("\n *** Warning: Relative change in V is very small (close to machine epsilon). ***")
        print(" *** Numerical derivative might be dominated by floating-point errors. Consider increasing dt_check. ***\n")

    # --- Compare ---
    abs_error = abs(dV_dt_numerical - dV_dt_theoretical)
    rel_error = abs_error / abs(dV_dt_theoretical) if abs(dV_dt_theoretical) > 1e-10 else abs_error
    print(f"Absolute Error = {abs_error:.4e}")
    print(f"Relative Error = {rel_error:.4f}")

    validation_result = {
        "theoretical_deriv": dV_dt_theoretical,
        "numerical_deriv": dV_dt_numerical,
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "relative_change": relative_change
    }

    if rel_error < 0.1: # Threshold for agreement
        print("✅ Theoretical RHS and Numerical Derivative agree well.")
    else:
        print("⚠️ Discrepancy found. Check derivation/implementation/parameters (e.g., dt_check).")

    return validation_result

# --- Example Usage ---
if __name__ == "__main__":
    # Parameters for the validation
    n_dim_val = 5
    k_dim_val = 3
    lambda_val = 115.0
    sigma_val = 0.0   # Test pure drift case first
    dt_val = 1e-3     # Use a less extreme dt first
    num_particles_val = 50_000

    # Set up target and consensus
    np.random.seed(42)
    X_star_val_pre = np.random.randn(n_dim_val, k_dim_val)
    X_star_val = project_stiefel(X_star_val_pre)
    np.random.seed(None)
    X_alpha_val = X_star_val # Idealized

    # Run validation for pure drift
    results_drift = validate_generator_action(n_dim_val, k_dim_val, lambda_val, sigma_val, dt_val, num_particles_val, X_star_val, X_alpha_val)

    # --- Test pure drift with very small dt ---
    dt_val_small = 1e-8
    results_drift_small_dt = validate_generator_action(n_dim_val, k_dim_val, lambda_val, sigma_val, dt_val_small, num_particles_val, X_star_val, X_alpha_val)


    # --- Test with Noise ---
    print("\n" + "="*30 + "\n--- Testing with Noise ---")
    lambda_val_noise = 115.0
    sigma_val_noise = 0.5
    dt_val_noise = 1e-3 # Use the less extreme dt
    num_particles_val_noise = 50_000

    results_noise = validate_generator_action(n_dim_val, k_dim_val, lambda_val_noise, sigma_val_noise, dt_val_noise, num_particles_val_noise, X_star_val, X_alpha_val)

    lambda_val_noise = 115.0
    sigma_val_noise = 0.5
    dt_val_noise = 1e-8 # Use the more extreme dt
    num_particles_val_noise = 50_000

    results_noise_small_dt = validate_generator_action(n_dim_val, k_dim_val, lambda_val_noise, sigma_val_noise, dt_val_noise, num_particles_val_noise, X_star_val, X_alpha_val)