import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from tqdm import tqdm
import warnings

# Suppress runtime warnings (often from np.exp)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 1. Parameters from Paper (Section V.C) ---
N = 200  # Number of particles
n = 5  # Matrix rows
k = 3  # Matrix columns
A = 20.0
a = 0.2
b = 3.0

beta = 50.0  # Inverse temperature (alpha in paper's text)
lambda_ = 1.0  # Drift coefficient
sigma = 0.25  # Noise coefficient

T = 10.0  # Terminal time
h = 0.05  # Time step (dt)
num_steps = int(T / h)
num_trials = 100  # Number of trials to average

# Target global minimizer X* = I_{n,k}
X_star = np.zeros((n, k))
X_star[:k, :k] = np.eye(k)

# Constant C_{n,k} = (2n - k - 1) / 2 [cite: 94-95]
C_nk = (2.0 * n - k - 1.0) / 2.0


# --- 2. Helper Functions (Manifold Operations & Objective) ---

def ackley_f(X, X_star, A, a, b):
    """
    Implements the Ackley function on V(n,k) as defined in [cite: 310-311].
    Global minimum is 0 at X = X_star.
    """
    n_rows, k_cols = X.shape
    diff_norm_sq = np.linalg.norm(X - X_star, 'fro') ** 2

    term1_exp = -a * np.sqrt(b ** 2 * diff_norm_sq / (n_rows * k_cols))
    term1 = -A * np.exp(term1_exp)

    cos_term = np.cos(2 * np.pi * b * (X - X_star))
    term2 = -np.exp(np.sum(cos_term) / (n_rows * k_cols))

    return term1 + term2 + np.e + A


def project_tangent(Z, X):
    """
    Projects matrix Z onto the tangent space of V(n,k) at X.
    P_X(Z) = Z - 0.5 * (X @ Z.T @ X + X @ X.T @ Z) [cite: 77]
    """
    return Z - 0.5 * (X @ (Z.T @ X) + X @ (X.T @ Z))


def project_manifold(Z):
    """
    Projects (retracts) matrix Z onto the Stiefel manifold V(n,k).
    Pi(Z) = U @ Vh, where Z = U @ S @ Vh (SVD) [cite: 186-189]
    """
    U, _, Vh = svd(Z, full_matrices=False)
    return U @ Vh


def initialize_particles(N, n, k):
    """
    Uniformly samples N particles from V(n,k) [cite: 194-197].
    Method: Z(Z^T Z)^(-1/2), which is equivalent to project_manifold(Z)
    where Z is a standard normal matrix.
    """
    particles = np.zeros((N, n, k))
    for i in range(N):
        Z = np.random.randn(n, k)
        particles[i] = project_manifold(Z)
    return particles


# --- 3. Validation Helper Functions ---

def is_on_manifold(X, tol=1e-6):
    """Checks if X.T @ X is close to the identity matrix[cite: 16]."""
    _, k_dim = X.shape
    identity = np.eye(k_dim)
    return np.allclose(X.T @ X, identity, atol=tol)


def is_in_tangent_space(A, X, tol=1e-6):
    """Checks if A.T @ X is skew-symmetric[cite: 69]."""
    k_dim = X.shape[1]
    skew_check = A.T @ X + X.T @ A
    return np.allclose(skew_check, np.zeros((k_dim, k_dim)), atol=tol)


def run_validation_tests(N_test=10):
    """Runs a set of unit tests for the core manifold operations."""
    print("Running validation tests...")

    # Test 1: project_manifold (Corrector step)
    for _ in range(N_test):
        Z_rand = np.random.randn(n, k)
        X_proj = project_manifold(Z_rand)
        assert is_on_manifold(X_proj), "Test Failed: project_manifold"
    print("  ✓ project_manifold (Corrector) test passed.")

    # Test 2: project_tangent (SDE component)
    for _ in range(N_test):
        X = project_manifold(np.random.randn(n, k))
        Z_rand = np.random.randn(n, k)
        A_proj = project_tangent(Z_rand, X)
        assert is_in_tangent_space(A_proj, X), "Test Failed: project_tangent"
    print("  ✓ project_tangent (SDE) test passed.")

    # Test 3: initialize_particles
    particles_test = initialize_particles(N, n, k)
    for i in range(N):
        assert is_on_manifold(particles_test[i]), "Test Failed: initialize_particles"
    print("  ✓ initialize_particles test passed.")

    print("All validation tests passed.\n")


# --- 4. Main Simulation Loop (Algorithm 1) ---

def run_simulation():
    """
    Runs a single trial of the CBO algorithm.
    Returns:
        f_history (np.array): History of f(X_bar_t) at each step.
        final_X_bar (np.array): The final consensus point X_bar_T.
    """
    particles = initialize_particles(N, n, k)
    f_history = np.zeros(num_steps)
    X_bar_t = np.zeros((n, k))  # Initialize to avoid scope issues

    for step in range(num_steps):
        # 1. Calculate f(X_i)
        f_values = np.array([ackley_f(p, X_star, A, a, b) for p in particles])

        # 2. Calculate weighted average X_bar_t* [cite: 206-210]
        f_min_val = np.min(f_values)
        weights = np.exp(-beta * (f_values - f_min_val))
        weights_sum = np.sum(weights)

        if weights_sum == 0 or not np.isfinite(weights_sum):
            weights = np.ones(N) / N
            weights_sum = 1.0

        X_bar_t = np.einsum('i,ijk->jk', weights, particles) / weights_sum
        f_history[step] = ackley_f(X_bar_t, X_star, A, a, b)

        # 3. Update each particle [cite: 175-191]
        new_particles = np.zeros_like(particles)
        for i in range(N):
            X_i = particles[i]
            dist_fro = np.linalg.norm(X_i - X_bar_t, 'fro')

            # Predictor step [cite: 188]
            drift_term = lambda_ * project_tangent(X_bar_t, X_i)
            correction_term = C_nk * (sigma ** 2 * dist_fro ** 2) / 2.0 * X_i
            dW = np.random.randn(n, k) * np.sqrt(h)
            noise_term = sigma * dist_fro * project_tangent(dW, X_i)

            X_pred = X_i + (drift_term - correction_term) * h + noise_term

            # Corrector step [cite: 189]
            new_particles[i] = project_manifold(X_pred)

            # Runtime Validation: [cite: 115-117]
            assert is_on_manifold(new_particles[i]), \
                f"Particle {i} left the manifold at step {step}"

        particles = new_particles

    return f_history, X_bar_t  # Return final X_bar_T


# --- 5. Run Experiment and Plot Results ---

run_validation_tests()

all_f_histories = []
all_final_X_bars = []
print(f"Running {num_trials} trials for the Ackley experiment...")
for _ in tqdm(range(num_trials)):
    f_hist, final_X_bar = run_simulation()
    all_f_histories.append(f_hist)
    all_final_X_bars.append(final_X_bar)

# --- Process results for plotting ---
all_f_histories = np.array(all_f_histories)
f_avg = np.mean(all_f_histories, axis=0)
f_min = np.min(all_f_histories, axis=0)
f_max = np.max(all_f_histories, axis=0)
time_axis = np.linspace(0, T, num_steps)

# Validation 1: Global Minimum Value
asymptotic_values = all_f_histories[:, -1]  # f(X_T*) for all trials

# Validation 2: Global Minimizer Argument
errors_fro = [
    np.linalg.norm(X_bar - X_star, 'fro') for X_bar in all_final_X_bars
]

# --- Plot 1: Reproduction of Figure 4(a) (Time Evolution) ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.fill_between(time_axis, f_min, f_max, color='C0', alpha=0.2,
                 label='Min-Max Range (100 trials)')
plt.plot(time_axis, f_avg, color='C0', lw=2,
         label='Average $f(\\bar{X}_t^*)$ (100 trials)')
plt.xlabel('Time (t)')
plt.ylabel('$f(\\bar{X}_t^*)$')
plt.title('CBO for Ackley Function on $V(5, 3)$ (Fig. 4a)')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)

# --- Plot 2: Reproduction of Figure 4(b) (Value Convergence) ---
plt.subplot(1, 2, 2)
plt.hist(asymptotic_values, bins=30, density=True,
         edgecolor='k', alpha=0.7, color='C0')
plt.xlabel('Asymptotic value ($f(\\bar{X}_T^*)$)')
plt.ylabel('Probability density')
plt.title('Histogram of $f(\\bar{X}_T^*)$ at T=10 (Fig. 4b)')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 3: NEW (Argument Convergence Validation) ---
plt.figure(figsize=(8, 6))
plt.hist(errors_fro, bins=30, density=True,
         edgecolor='k', alpha=0.7, color='C1')
plt.xlabel('Final Error $|| \\bar{X}_T^* - X^* ||_F$')
plt.ylabel('Probability density')
plt.title('Validation: Histogram of Error to Global Minimizer $X^*$')
plt.grid(True)
plt.show()

print(f"\nExperiment complete.")
print(f"--- Global Minimum (Value) Validation ---")
print(f"Mean asymptotic value f(X_T*): {np.mean(asymptotic_values):.4f} (Target: 0)")
print(f"Min  asymptotic value f(X_T*): {np.min(asymptotic_values):.4f} (Target: 0)")
print(f"\n--- Global Minimizer (Argument) Validation ---")
print(f"Mean error ||X_bar_T - X*||_F:  {np.mean(errors_fro):.4f} (Target: 0)")
print(f"Min  error ||X_bar_T - X*||_F:  {np.min(errors_fro):.4f} (Target: 0)")

# --- 6. NEW: Final Pass/Fail Summary ---

print("\n--- Experiment Statistics ---")
mean_error_val = np.mean(errors_fro)
min_error_val = np.min(errors_fro)
mean_f_val = np.mean(asymptotic_values)
min_f_val = np.min(asymptotic_values)

print(f"Mean f(X_T*): {mean_f_val:.4f} (Target: 0)")
print(f"Min  f(X_T*): {min_f_val:.4f} (Target: 0)")
print(f"Mean ||X_T* - X*||_F:  {mean_error_val:.4f} (Target: 0)")
print(f"Min  ||X_T* - X*||_F:  {min_error_val:.4f} (Target: 0)")

print("\n--- Final Validation Summary ---")

# Define reasonable thresholds for this stochastic experiment
# We expect the *mean* error to be small, and the *best* error to be very small.
MEAN_ERROR_THRESHOLD = 0.2
MIN_ERROR_THRESHOLD = 0.05

# Check 1: Unit Tests (already passed if we got this far)
print("1. Unit Tests (Operators & Init):     ✅ PASS")

# Check 2: Runtime Constraints (passed if we got this far)
print("2. Runtime Manifold Constraint:     ✅ PASS")

# Check 3: Convergence to Global Minimizer
if mean_error_val < MEAN_ERROR_THRESHOLD and min_error_val < MIN_ERROR_THRESHOLD:
    print(f"3. Convergence to Global Minimizer: ✅ PASS")
    print(f"   (Mean Error: {mean_error_val:.4f} < {MEAN_ERROR_THRESHOLD})")
    print(f"   (Min Error:  {min_error_val:.4f} < {MIN_ERROR_THRESHOLD})")
else:
    print(f"3. Convergence to Global Minimizer: ❌ FAIL")
    if mean_error_val >= MEAN_ERROR_THRESHOLD:
        print(f"   (Mean Error: {mean_error_val:.4f} >= {MEAN_ERROR_THRESHOLD})")
    if min_error_val >= MIN_ERROR_THRESHOLD:
        print(f"   (Min Error:  {min_error_val:.4f} >= {MIN_ERROR_THRESHOLD})")