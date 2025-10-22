import numpy as np
import matplotlib.pyplot as plt
import time

# --- Manifold & SDE Functions (same as before) ---

def project_stiefel(X):
  """Projects a matrix X onto the Stiefel manifold V(n, k) via QR decomposition."""
  try:
    q, r = np.linalg.qr(X)
    # Correct the signs based on the diagonal of r (optional but good practice)
    # d = np.diag(np.sign(np.diag(r)))
    # q = q @ d
    return q[:, :X.shape[1]] # Ensure we get nxk matrix
  except np.linalg.LinAlgError:
    print("Warning: QR decomposition failed during projection.")
    # Fallback or error handling
    n, k = X.shape
    I_nk = np.zeros((n,k))
    I_nk[:k, :k] = np.eye(k)
    # Return a point on the manifold, perhaps identity or re-randomized
    return project_stiefel(I_nk + 1e-6 * np.random.randn(n,k))

def projection_tangent_space(X, Z):
  """Projects matrix Z onto the tangent space of V(n, k) at X."""
  term1 = X @ Z.T @ X
  term2 = X @ X.T @ Z
  return Z - 0.5 * (term1 + term2)

def stiefel_cbo_step(X_current, X_alpha, lambda_param, sigma_param, dt, n_dim, k_dim):
  """Performs one projected Euler-Maruyama step for CBO on V(n, k)."""
  C_nk_prime = n_dim - (k_dim + 1) / 2.0
  dist_alpha_sq = np.linalg.norm(X_current - X_alpha, 'fro')**2
  dist_alpha = np.sqrt(dist_alpha_sq) if dist_alpha_sq > 1e-16 else 0.0

  # Drift
  drift1 = -lambda_param * projection_tangent_space(X_current, X_alpha)
  drift2 = (sigma_param**2 * C_nk_prime / 2.0) * dist_alpha_sq * X_current
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

# --- Simulation Setup ---
n_dim = 5
k_dim = 3
dt = 0.01
num_steps = 5000
num_runs = 50

# --- Parameter Sets to Test ---
parameter_sets = [
    # Previous problematic ones for reference
    {'lambda': 1.0, 'sigma': 0.5, 'label': 'λ=1, σ=0.5'},
    {'lambda': 2.0, 'sigma': 0.2, 'label': 'λ=2, σ=0.2'},
    # Pure drift cases
    {'lambda': 5.0, 'sigma': 0.0, 'label': 'λ=5, σ=0.0'},
    {'lambda': 115.0,'sigma': 0.0, 'label': 'λ=115, σ=0.0'},
    # Expected decrease
    {'lambda': 10.0, 'sigma': 0.2, 'label': 'λ=10, σ=0.2'},
    {'lambda': 5.0,  'sigma': 0.1, 'label': 'λ=5, σ=0.1'},
    # High drift, high noise
    {'lambda': 115.0,'sigma': 0.9, 'label': 'λ=115, σ=0.9'},
    # Borderline?
    {'lambda': 20.0, 'sigma': 0.5, 'label': 'λ=20, σ=0.5'},
]

# --- Initialization ---
np.random.seed(42) # Seed for X_star
X_star_pre = np.random.randn(n_dim, k_dim)
X_star = project_stiefel(X_star_pre)
np.random.seed(None) # Reset seed

X_alpha = X_star # Idealized consensus

results = {}

# --- Simulation Loop for Multiple Parameter Sets ---
start_time_total = time.time()
for params in parameter_sets:
    lambda_p = params['lambda']
    sigma_p = params['sigma']
    label = params['label']
    print(f"\n--- Running: {label} ---")

    V_history = np.zeros((num_runs, num_steps + 1))
    max_manifold_error_overall = 0.0
    start_time_set = time.time()

    for run in range(num_runs):
      X0_pre = np.random.randn(n_dim, k_dim)
      X_current = project_stiefel(X0_pre)
      V_history[run, 0] = 0.5 * np.linalg.norm(X_current - X_star, 'fro')**2
      max_manifold_error_run = 0.0

      for k in range(num_steps):
        X_next = stiefel_cbo_step(X_current, X_alpha, lambda_p, sigma_p, dt, n_dim, k_dim)
        identity_k = np.eye(k_dim)
        manifold_error = np.linalg.norm(X_next.T @ X_next - identity_k, 'fro')
        max_manifold_error_run = max(max_manifold_error_run, manifold_error)
        X_current = X_next
        V_history[run, k + 1] = 0.5 * np.linalg.norm(X_current - X_star, 'fro')**2

      max_manifold_error_overall = max(max_manifold_error_overall, max_manifold_error_run)
      # Optional: Print progress
      if (run + 1) % (num_runs // 5 or 1) == 0:
          print(f"  Run {run + 1}/{num_runs} completed.")

    end_time_set = time.time()
    results[label] = {
        'avg_V': np.mean(V_history, axis=0),
        'std_V': np.std(V_history, axis=0),
        'min_V': np.min(V_history, axis=0),
        'max_V': np.max(V_history, axis=0),
        'max_manifold_error': max_manifold_error_overall,
        'time': end_time_set - start_time_set,
        'params': params # Store params for title
    }

    # --- Print Summary for this Parameter Set ---
    print(f"  Finished in {results[label]['time']:.2f} seconds.")
    print(f"  Max Manifold Error: {results[label]['max_manifold_error']:.2e}")
    if results[label]['max_manifold_error'] < 1e-10:
      print("  ✅ Manifold constraint preserved.")
    else:
      print("  ⚠️ Manifold constraint error larger than expected.")
    print(f"  Initial Avg V(0) : {results[label]['avg_V'][0]:.6f}")
    print(f"  Final Avg V(T)   : {results[label]['avg_V'][-1]:.6f}")
    print(f"  Final Std Dev V(T): {results[label]['std_V'][-1]:.6f}")
    if results[label]['avg_V'][-1] < results[label]['avg_V'][0]:
        print("  ✅ Average V decreased.")
    else:
        print(f"  ❌ Average V did NOT decrease (Ratio Final/Initial: {results[label]['avg_V'][-1]/results[label]['avg_V'][0]:.2f}).")


end_time_total = time.time()
print(f"\nTotal simulation time: {end_time_total - start_time_total:.2f} seconds.")

# --- Plotting Results (Separate Plots) ---
time_points = np.arange(num_steps + 1) * dt

for label, res in results.items():
    plt.figure(figsize=(10, 6)) # Create a new figure for each plot
    avg_V = res['avg_V']
    std_V = res['std_V']
    params = res['params']

    # Find sensible y-axis limits for log scale
    min_plot_V = max(1e-12, np.min(avg_V[avg_V > 0]) / 10) if np.any(avg_V > 0) else 1e-12
    max_plot_V = np.max(avg_V) * 1.5 if np.any(avg_V > 0) else 1.0

    plt.plot(time_points, avg_V, label=f'Average V(t) over {num_runs} runs', color='blue', linewidth=2)
    # Shaded region for standard deviation
    plt.fill_between(time_points, avg_V - std_V, avg_V + std_V,
                     color='lightblue', alpha=0.5, label='Std Dev Range')

    plt.xlabel("Time (t)")
    plt.ylabel("V(t) = 0.5 * E[||X_t - X*||_F^2]")
    # Use the label directly in the title
    plt.title(f"Evolution of V(t) for {label}\n"
              f"V({n_dim},{k_dim}), dt={dt}, N_runs={num_runs}")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    # Use calculated limits, ensuring bottom is not zero or negative for log scale
    plt.ylim(bottom=min_plot_V, top=max_plot_V)
    # plt.show() # Display the current figure
    plt.savefig(f"2.1 Ito's Formula Results/{label.replace('λ','lambda').replace('σ', 'sigma').replace(', ','-')}.png")