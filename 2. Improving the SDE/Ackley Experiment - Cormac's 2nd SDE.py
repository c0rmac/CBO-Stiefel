import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from tqdm import tqdm
import warnings
import pandas as pd

# Suppress runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Parameters from Paper (Section V.C) ---

# Experiment parameters
N = 200  # Number of particles
n = 5  # Matrix rows
k = 3  # Matrix columns
A = 20.0
a = 0.2
b = 3.0

# SDE parameters
beta = 50.0  # Inverse temperature (controls consensus strength)
lambda_ = 1.0  # Drift coefficient

# --- NEW: Hybrid Noise Parameters ---
# Kim's original sigma, now for the adaptive part
sigma1 = 0.25

# Annealing schedule for the robust exploration part
sigma2_initial = 0.05  # Start with a modest "temperature"
sigma2_final = 0.003  # End with no exploration
# sigma2_initial = 0.0  # Start with a modest "temperature"
# sigma2_final = 0.0  # End with no exploration
T = 20.0  # Terminal time
h = 0.05  # Time step (dt)
num_steps = int(T / h)
num_trials = 100  # Number of trials to average


def get_sigma2(t):
    """Linear annealing schedule for the sigma2 noise component."""
    if T == 0:
        return sigma2_final
    return sigma2_initial * (1.0 - (t / T)) + sigma2_final * (t / T)


# Target minimizer X* = I_{n,k}
X_star = np.zeros((n, k))
X_star[:k, :k] = np.eye(k)

# Constant C_{n,k} = (2n - k - 1) / 2 [cite: 94-95]
C_nk = (2.0 * n - k - 1.0) / 2.0


# --- 2. Helper Functions (Manifold Operations & Objective) ---

def ackley_f(X, X_star, A, a, b):
    """Implements the Ackley function on V(n,k) [cite: 310-311]."""
    n_rows, k_cols = X.shape
    diff_norm_sq = np.linalg.norm(X - X_star, 'fro') ** 2

    term1_exp = -a * np.sqrt(b ** 2 * diff_norm_sq / (n_rows * k_cols))
    term1 = -A * np.exp(term1_exp)

    cos_term = np.cos(2 * np.pi * b * (X - X_star))
    term2 = -np.exp(np.sum(cos_term) / (n_rows * k_cols))

    return term1 + term2 + np.e + A


def project_tangent(Z, X):
    """Projects matrix Z onto the tangent space of V(n,k) at X."""
    return Z - 0.5 * (X @ (Z.T @ X) + X @ (X.T @ Z))


def project_manifold(Z):
    """Projects (retracts) matrix Z onto the Stiefel manifold V(n,k)."""
    U, _, Vh = svd(Z, full_matrices=False)
    return U @ Vh


def initialize_particles(N, n, k):
    """Uniformly samples N particles from V(n,k) [cite: 194-197]."""
    particles = np.zeros((N, n, k))
    for i in range(N):
        Z = np.random.randn(n, k)
        particles[i] = project_manifold(Z)
    return particles


# --- 3. NEW: Hybrid "Adaptive Annealing" CBO Loop ---

def run_hybrid_cbo_simulation():
    """
    Runs a single trial of the *Hybrid* CBO algorithm.
    """
    particles = initialize_particles(N, n, k)
    f_history = np.zeros(num_steps)

    for step in range(num_steps):
        t = step * h  # Current time

        # 1. Calculate f(X_i)
        f_values = np.array([ackley_f(p, X_star, A, a, b) for p in particles])

        # 2. Calculate weighted average X_bar_t* [cite: 206-210]
        f_min_val = np.min(f_values)
        weights = np.exp(-beta * (f_values - f_min_val))
        weights_sum = np.sum(weights)

        if weights_sum < 1e-10:
            weights = np.ones(N) / N
            weights_sum = 1.0

        X_bar_t = np.einsum('i,ijk->jk', weights, particles) / weights_sum
        f_history[step] = ackley_f(X_bar_t, X_star, A, a, b)

        # 3. Update each particle [Adapted from Alg. 1]
        new_particles = np.zeros_like(particles)
        for i in range(N):
            X_i = particles[i]

            # --- HYBRID PREDICTOR STEP ---

            # Get noise strengths
            dist_fro = np.linalg.norm(X_i - X_bar_t, 'fro')
            adaptive_noise_strength = sigma1 * dist_fro
            annealing_noise_strength = get_sigma2(t)
            S_total = adaptive_noise_strength + annealing_noise_strength

            # Drift term: (same as before)
            drift_term = lambda_ * project_tangent(X_bar_t, X_i)

            # Correction term: (uses S_total)
            correction_term = C_nk * (S_total ** 2) / 2.0 * X_i

            # Noise term: (uses S_total)
            dW = np.random.randn(n, k) * np.sqrt(h)
            noise_term = S_total * project_tangent(dW, X_i)

            # Predictor step
            X_pred = X_i + (drift_term - correction_term) * h + noise_term

            # Corrector step
            new_particles[i] = project_manifold(X_pred)

        particles = new_particles

    return f_history


# --- 4. Run Experiment and Plot Results ---

all_f_histories = []
print(f"Running {num_trials} trials for the *Hybrid CBO* Ackley experiment...")
for _ in tqdm(range(num_trials)):
    all_f_histories.append(run_hybrid_cbo_simulation())

# Process results for plotting
all_f_histories = np.array(all_f_histories)
f_avg = np.mean(all_f_histories, axis=0)
f_min = np.min(all_f_histories, axis=0)
f_max = np.max(all_f_histories, axis=0)
asymptotic_values = all_f_histories[:, -1]

time_axis = np.linspace(0, T, num_steps)
sigma2_axis = get_sigma2(time_axis)

# --- Plot 1: (Similar to Figure 4a) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.fill_between(time_axis, f_min, f_max, color='C0', alpha=0.2,
                 label='Min-Max Range (100 trials)')
ax1.plot(time_axis, f_avg, color='C0', lw=2,
         label='Average $f(\\bar{X}_t^*)$ (100 trials)')

ax1.set_xlabel('Time (t)')
ax1.set_ylabel('$f(\\bar{X}_t^*)$')
ax1.set_title('Hybrid CBO for Ackley Function on $V(5, 3)$')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(bottom=-0.1)  # Show zero line

# --- Plot 2: Annealing Schedule ---
ax2.plot(time_axis, sigma2_axis, lw=2, color='C1')
ax2.set_title("Annealing Noise $\sigma_2(t)$")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Noise 'Temperature' $\sigma_2(t)$")
ax2.grid(True)

plt.tight_layout()
plt.show()

# --- Plot 3: (Similar to Figure 4b) ---
plt.figure(figsize=(10, 6))
# Using a smaller, fixed range to zoom in on 0
bins = np.linspace(0, 0.5, 50)
plt.hist(asymptotic_values, bins=bins, density=True,
         edgecolor='k', alpha=0.7, color='C0')
plt.xlabel('Asymptotic value ($f(\\bar{X}_T^*)$)')
plt.ylabel('Probability density')
plt.title('Histogram of $f(\\bar{X}_T^*)$ at T=10 (Hybrid CBO)')
plt.grid(True)
plt.xlim(-0.01, 0.5)  # Zoom in to see the tail
plt.show()

print(f"\nExperiment complete.")
print(f"Mean asymptotic value f(X_T*): {np.mean(asymptotic_values):.4f}")
print(f"Min asymptotic value f(X_T*):  {np.min(asymptotic_values):.4f}")
print(f"Max asymptotic value f(X_T*):  {np.max(asymptotic_values):.4f}")