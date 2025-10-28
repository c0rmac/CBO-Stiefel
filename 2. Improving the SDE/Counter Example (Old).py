import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import det  # Use scipy.linalg.det for potentially more stable determinant


# --- Geometric Helper Functions ---

def project_stiefel(X):
    """Projects a matrix X onto the Stiefel manifold V(n, k) via QR decomposition."""
    try:
        q, r = np.linalg.qr(X)
        # Project onto O(k)
        q = q[:, :X.shape[1]]

        # Optional sign correction (ensures uniqueness, not strictly necessary for V(n,k))
        # d = np.diag(np.sign(np.diag(r)))
        # q = q @ d
        return q
    except np.linalg.LinAlgError:
        print("Warning: QR decomposition failed during projection.")
        n, k = X.shape
        I_nk = np.zeros((n, k))
        I_nk[:k, :k] = np.eye(k)
        return project_stiefel(I_nk + 1e-6 * np.random.randn(n, k))  # Attempt recovery


def project_so_n(X):
    """Projects a matrix X onto the Special Orthogonal group SO(n)."""
    if X.shape[0] != X.shape[1]:
        raise ValueError("Input matrix must be square to project onto SO(n).")
    n = X.shape[0]

    # Project onto O(n) first
    Q = project_stiefel(X)  # Q is now in O(n)

    # Check determinant
    if det(Q) < 0:
        # Flip the sign of one column to make determinant +1
        # This is a standard way to project O(n) -> SO(n)
        Q[:, 0] = -Q[:, 0]

    return Q


def projection_tangent_space(X, Z):
    """Projects matrix Z onto the tangent space of V(n, k) at X."""
    # Using the form from Kim et al. / Thesis Approach
    term1 = X @ Z.T @ X
    term2 = X @ X.T @ Z
    return Z - 0.5 * (term1 + term2)


def sample_uniform_so_n(n):
    """Generates a uniformly random matrix from SO(n)."""
    # Generate a random matrix
    A = np.random.normal(size=(n, n))
    # Project onto O(n) via QR
    Q, _ = np.linalg.qr(A)
    # Ensure it's in SO(n)
    if det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# --- CBO Core Functions ---

def objective_func_trap(X, X_star, X_local):
    """
    Objective function with a global min at X_star and local min at X_local.
    E(X) = 0.5 * ||X - X_local||_F^2 - 5 * exp(-||X - X_star||_F^2)
    """
    dist_sq_local = np.linalg.norm(X - X_local, 'fro') ** 2
    dist_sq_global = np.linalg.norm(X - X_star, 'fro') ** 2

    energy = 0.5 * dist_sq_local - 5.0 * np.exp(-dist_sq_global)
    return energy


def calculate_consensus_point(particles, alpha, objective_func):
    """Calculates the empirical consensus point X_alpha."""
    min_energy = np.inf
    energies = []
    for p in particles:
        energy = objective_func(p)
        energies.append(energy)
        min_energy = min(min_energy, energy)

    weighted_sum = np.zeros_like(particles[0])
    total_weight_denominator = 0.0

    for i, p in enumerate(particles):
        energy = energies[i]
        # Numerically stable weight
        weight = np.exp(-alpha * (energy - min_energy))

        if weight > 1e-100:
            weighted_sum += p * weight
            total_weight_denominator += weight

    if total_weight_denominator < 1e-100:
        # Fallback if all weights are zero (e.g., extreme alpha)
        # Return the particle that had the minimum energy
        return particles[np.argmin(energies)]

    consensus = weighted_sum / total_weight_denominator
    return consensus


def stiefel_cbo_step(X_current, X_alpha, lambda_param, sigma_param, dt, n_dim, k_dim):
    """Performs one projected Euler-Maruyama step for CBO on V(n, k)."""
    C_nk_prime = n_dim - (k_dim + 1) / 2.0
    dist_alpha_sq = np.linalg.norm(X_current - X_alpha, 'fro') ** 2
    dist_alpha = np.sqrt(dist_alpha_sq) if dist_alpha_sq > 1e-16 else 0.0

    # Drift (Corrected Signs)
    drift1 = +lambda_param * projection_tangent_space(X_current, X_alpha)
    drift2 = -(sigma_param ** 2 * C_nk_prime / 2.0) * dist_alpha_sq * X_current
    total_drift = drift1 + drift2

    # Diffusion
    diffusion_term = np.zeros_like(X_current)
    if sigma_param > 1e-15 and dist_alpha > 1e-15:
        dW = np.random.normal(0.0, np.sqrt(dt), size=(n_dim, k_dim))
        projected_noise = projection_tangent_space(X_current, dW)
        diffusion_term = sigma_param * dist_alpha * projected_noise

    # Euler Step & Projection
    X_predict = X_current + total_drift * dt + diffusion_term

    # Project back onto SO(n) since n=k=3
    X_next = project_so_n(X_predict)

    return X_next


# --- Simulation Setup ---
n_dim = 3
k_dim = 3  # We are on SO(3)
N_particles = 100
lambda_param = 10.0
sigma_param = 1.0
alpha_param = 0.0  # Large alpha for strong consensus
dt = 0.01
K_steps = 500
M_runs = 100  # Number of independent simulations

# --- Minima Definition ---
X_star = np.eye(n_dim)  # Global minimizer (Identity)
X_local = np.diag([-1.0, -1.0, 1.0])  # Local minimizer (180 deg rot around z)

# Objective function with fixed minima
obj_func = lambda x: objective_func_trap(x, X_star, X_local)

# --- Results Storage ---
results = {"success": 0, "failure": 0}
final_d_global = []
final_d_local = []

print(f"--- Starting Counterexample Simulation ---")
print(f"Manifold: SO({n_dim})")
print(f"Objective: Global min E(I) = -1, Local min E(diag(-1,-1,1)) approx -0.0017")
print(
    f"Parameters: N={N_particles}, lambda={lambda_param}, sigma={sigma_param}, alpha={alpha_param}, K_steps={K_steps}, M_runs={M_runs}")

start_time_total = time.time()

# --- Main Simulation Loop ---
for m in range(M_runs):
    # Initialize N particles uniformly on SO(3)
    particles = [sample_uniform_so_n(n_dim) for _ in range(N_particles)]

    # Run simulation for K steps
    for k in range(K_steps):
        # Calculate consensus point *at each step*
        X_alpha_k = calculate_consensus_point(particles, alpha_param, obj_func)

        # Update each particle
        new_particles = []
        for i in range(N_particles):
            X_next = stiefel_cbo_step(particles[i], X_alpha_k, lambda_param, sigma_param, dt, n_dim, k_dim)
            new_particles.append(X_next)
        particles = new_particles

        # Optional: Print progress of a single run
        # if (k+1) % 100 == 0:
        #     print(f"  Run {m+1}/{M_runs}, Step {k+1}/{K_steps}")

    # --- End of run: Classify Result ---
    # Calculate final consensus point
    X_alpha_K = calculate_consensus_point(particles, alpha_param, obj_func)

    # Calculate distances to minima
    d_global = np.linalg.norm(X_alpha_K - X_star, 'fro') ** 2
    d_local = np.linalg.norm(X_alpha_K - X_local, 'fro') ** 2

    final_d_global.append(d_global)
    final_d_local.append(d_local)

    if d_global < d_local:
        results["success"] += 1
    else:
        results["failure"] += 1

    if (m + 1) % (M_runs // 10 or 1) == 0:
        print(f"Completed run {m + 1}/{M_runs}... (Current Failures: {results['failure']})")

end_time_total = time.time()
print(f"\n--- Simulation Finished in {end_time_total - start_time_total:.2f} seconds ---")

# --- Report Results ---
failure_rate = results["failure"] / M_runs
print(f"\nFailure Rate (Converged to Local Min): {failure_rate * 100:.1f}%")
print(f"Success Rate (Converged to Global Min): {results['success'] / M_runs * 100:.1f}%")

if failure_rate > 0.05:  # 5% threshold
    print("\n✅ Counterexample successful: Algorithm frequently trapped in local minimum.")
    print("This supports the hypothesis that mild assumptions (like uniform init) are insufficient")
    print("and stricter 'well-preparedness' conditions (used by variance proof) may be necessary.")
else:
    print("\n⚠️ Counterexample failed: Algorithm consistently found global minimum.")
    print("This might suggest the algorithm is more robust than expected, or the 'trap' was not effective.")

# --- Plotting Histogram ---
plt.figure(figsize=(10, 6))
plt.hist(np.log10(final_d_global), bins=30, alpha=0.7, label='log10(Distance to Global Min $X^*$)', color='green')
plt.hist(np.log10(final_d_local), bins=30, alpha=0.7, label='log10(Distance to Local Min $X_L$)', color='red')
plt.xlabel("Log10(Final Squared Frobenius Distance)")
plt.ylabel("Number of Runs")
plt.title(f"Histogram of Final Consensus Point Distances ({M_runs} runs)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()