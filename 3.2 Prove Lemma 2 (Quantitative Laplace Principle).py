# File: validate_lemma2_relationship.py

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # For fitting lines
# Assuming stiefel_utils.py contains project_stiefel
from stiefel_utils import project_stiefel

# --- Helper Functions (calculate_consensus_point, objective_simple_distance_sq, calculate_V_functional) ---
# (Include the function definitions here - same as before)
def calculate_consensus_point(particles, alpha, objective_func):
    """Calculates the empirical consensus point X_alpha."""
    min_energy = np.inf
    # Find minimum energy for numerical stability
    for p in particles:
        energy = objective_func(p)
        min_energy = min(min_energy, energy)

    weighted_sum = np.zeros_like(particles[0])
    total_weight_denominator = 0.0
    for p in particles:
        energy = objective_func(p)
        weight = np.exp(-alpha * (energy - min_energy))
        if weight > 1e-100: # Avoid adding negligible weights if possible
            weighted_sum += p * weight
            total_weight_denominator += weight

    if total_weight_denominator < 1e-100:
        print(f"Warning: Total weight near zero for alpha={alpha}. Returning mean.")
        # Fallback to simple mean
        valid_particles = [p for p in particles if not np.isnan(p).any()]
        if not valid_particles:
             raise ValueError("No valid particles found in the ensemble.")
        return np.mean(valid_particles, axis=0)

    consensus = weighted_sum / total_weight_denominator
    return consensus

def objective_simple_distance_sq(X, X_target):
    """Simple objective: squared Frobenius distance to a target X_target."""
    return 0.5 * np.linalg.norm(X - X_target, 'fro')**2

def calculate_V_functional(particles, X_target):
    """Calculates the V functional for an ensemble."""
    sum_V = 0.0
    valid_count = 0
    for p in particles:
       if not np.isnan(p).any():
           sum_V += 0.5 * np.linalg.norm(p - X_target, 'fro')**2
           valid_count += 1
    if valid_count == 0:
        return np.nan # Or raise an error
    return sum_V / valid_count


# --- Main Validation Function ---

def validate_lemma2_relationship(n_dim, k_dim, alphas_to_test, num_ensembles, particles_per_ensemble, X_star):
    """
    Numerically investigates the relationship structure of Lemma 2.
    Args/Returns are the same as before...
    """
    print(f"\n--- Numerical Check for Lemma 2 (Relationship V vs Error) ---")
    print(f"Parameters: V({n_dim}, {k_dim})")
    print(f"Number of Ensembles (M): {num_ensembles}")
    print(f"Particles per Ensemble (N): {particles_per_ensemble}")
    print(f"Alphas to test: {alphas_to_test}")

    objective_func = lambda x: objective_simple_distance_sq(x, X_star)

    results_V = []
    results_ErrorSq = []
    results_Alpha = []

    total_start_time = time.time()
    print("Generating ensembles and calculating V and errors...")

    for m in range(num_ensembles):
        ensemble_m = []
        for i in range(particles_per_ensemble):
            np.random.seed(int(time.time() * 1000) % (2**32 -1) + m*particles_per_ensemble + i)
            X0_pre = np.random.randn(n_dim, k_dim)
            X_i = project_stiefel(X0_pre)
            ensemble_m.append(X_i)

        V_m = calculate_V_functional(ensemble_m, X_star)
        if np.isnan(V_m):
            print(f"Skipping ensemble {m+1}/{num_ensembles} due to calculation issues.")
            continue

        for alpha in alphas_to_test:
            try:
                X_alpha_m = calculate_consensus_point(ensemble_m, alpha, objective_func)
                ErrorSq_m_alpha = np.linalg.norm(X_alpha_m - X_star, 'fro')**2

                results_V.append(V_m)
                results_ErrorSq.append(ErrorSq_m_alpha)
                results_Alpha.append(alpha)
            except Exception as e:
                print(f"Error calculating for ensemble {m+1}, alpha={alpha}: {e}")
                continue

        if (m + 1) % (num_ensembles // 10 or 1) == 0:
            print(f"  Processed ensemble {m + 1}/{num_ensembles}")

    total_end_time = time.time()
    print(f"Calculations finished in {total_end_time - total_start_time:.2f} seconds.")

    results_V = np.array(results_V)
    results_ErrorSq = np.array(results_ErrorSq)
    results_Alpha = np.array(results_Alpha)

    # --- Validation Step: Fit Lines and Check Trends ---
    print("\n--- Validation Check ---")
    fitted_slopes = []
    fitted_intercepts = []
    unique_alphas = sorted(list(set(results_Alpha)))

    for i, alpha in enumerate(unique_alphas):
        mask = (results_Alpha == alpha)
        if np.sum(mask) > 1: # Need at least 2 points to fit a line
            V_subset = results_V[mask].reshape(-1, 1)
            Err_subset = results_ErrorSq[mask]

            # Fit linear model: ErrSq = slope * V + intercept
            model = LinearRegression().fit(V_subset, Err_subset)
            slope = model.coef_[0]
            intercept = model.intercept_

            fitted_slopes.append(slope)
            fitted_intercepts.append(intercept)
            print(f"  alpha = {alpha:.1e}: Fitted ErrSq ≈ {slope:.4e} * V + {intercept:.4e}")
        else:
            fitted_slopes.append(np.nan)
            fitted_intercepts.append(np.nan)
            print(f"  alpha = {alpha:.1e}: Not enough data points to fit line.")

    # Check if slopes and intercepts decrease with alpha (as expected from theory)
    slopes_decreasing = all(fitted_slopes[i] >= fitted_slopes[i+1] or np.isnan(fitted_slopes[i+1]) for i in range(len(fitted_slopes)-1))
    intercepts_decreasing = all(fitted_intercepts[i] >= fitted_intercepts[i+1] or np.isnan(fitted_intercepts[i+1]) for i in range(len(fitted_intercepts)-1))

    print("\nValidation Summary:")
    print(f"  Fitted slopes generally decreasing with alpha? {slopes_decreasing}")
    print(f"  Fitted intercepts generally decreasing with alpha? {intercepts_decreasing}")
    if slopes_decreasing and intercepts_decreasing:
        print("  ✅ The observed relationship structure is consistent with Lemma 2.")
    else:
        print("  ⚠️ The observed relationship structure may deviate from Lemma 2's prediction.")

    # --- Plotting ---
    plt.figure(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_alphas)))

    # Scatter plot
    for i, alpha in enumerate(unique_alphas):
        mask = (results_Alpha == alpha)
        if np.any(mask):
            plt.scatter(results_V[mask], results_ErrorSq[mask],
                        label=f'α = {alpha:.1e}', color=colors[i], alpha=0.6, s=15)

    # Plot fitted lines
    plot_V_range = np.logspace(np.log10(np.min(results_V[results_V > 1e-12]) if np.any(results_V > 1e-12) else 1e-12),
                               np.log10(np.max(results_V) if np.any(results_V > 1e-12) else 1.0), 50)
    for i, alpha in enumerate(unique_alphas):
         if not np.isnan(fitted_slopes[i]):
             plt.plot(plot_V_range, fitted_slopes[i] * plot_V_range + fitted_intercepts[i],
                      color=colors[i], linestyle='--', linewidth=1) #, label=f'Fit α={alpha:.1e}')

    plt.xlabel("Ensemble V = (1/N) Σ 0.5 ||Xi - X*||_F^2")
    plt.ylabel("Squared Consensus Error ||X_α - X*||_F^2")
    plt.title(f"Consensus Error vs. V for different Alpha (Fitted Lines)\n"
              f"V({n_dim},{k_dim}), N={particles_per_ensemble} particles/ensemble, M={num_ensembles} ensembles")
    plt.legend(title="Alpha Values", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--")
    plt.xscale('log')
    plt.yscale('log')
    if len(results_V)>0 and len(results_ErrorSq)>0 :
        plt.xlim(left=max(1e-12, np.min(results_V[results_V > 0])/10) if np.any(results_V > 0) else 1e-12,
                 right=np.max(results_V) * 1.5 if np.any(results_V > 0) else 1.0)
        plt.ylim(bottom=max(1e-12, np.min(results_ErrorSq[results_ErrorSq > 0])/10) if np.any(results_ErrorSq > 0) else 1e-12,
                 top=np.max(results_ErrorSq) * 1.5 if np.any(results_ErrorSq > 0) else 1.0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    validation_result = {"alphas": results_Alpha, "V_values": results_V, "ErrorSq_values": results_ErrorSq,
                         "fitted_slopes": fitted_slopes, "fitted_intercepts": fitted_intercepts,
                         "slopes_decreasing": slopes_decreasing, "intercepts_decreasing": intercepts_decreasing}
    return validation_result

# --- Example Usage ---
if __name__ == "__main__":
    n_dim_val = 5
    k_dim_val = 3
    num_ensembles_val = 100
    particles_per_ensemble_val = 1000
    alphas_val = np.logspace(0, 8, 9)

    np.random.seed(42)
    X_star_val_pre = np.random.randn(n_dim_val, k_dim_val)
    X_star_val = project_stiefel(X_star_val_pre)
    np.random.seed(None)

    results = validate_lemma2_relationship(n_dim_val, k_dim_val, alphas_val, num_ensembles_val, particles_per_ensemble_val, X_star_val)