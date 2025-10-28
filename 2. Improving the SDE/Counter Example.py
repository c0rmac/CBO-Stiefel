import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. Parameters for V(2, 1) Counterexample ---

# Manifold parameters
n = 2
k = 1

# SDE parameters (from Kim paper for Ackley)
lambda_ = 1.0  # Drift coefficient
sigma = 0.25  # Noise coefficient
beta = 50.0  # Inverse temperature

# C_{n,k} = (2n - k - 1) / 2
C_nk = (2.0 * n - k - 1.0) / 2.0  # C_nk = (4 - 1 - 1) / 2 = 1.0

# Simulation parameters
N = 50  # Number of particles
T = 5.0  # Terminal time
h = 0.01  # Time step (dt)
num_steps = int(T / h)


# --- 2. Objective Function & Manifold Operations ---

def objective_f(X):
    """
    Objective function E(X) = 1 - x_1.
    Minimizer X* = [1, 0].T
    Maximizer X_a = [-1, 0].T
    """
    return 1.0 - X[0, 0]


def project_manifold(Z):
    """
    Projects a vector Z in R^{2x1} onto the manifold V(2, 1) (the unit circle).
    This is simply normalization for k=1.
    """
    norm = np.linalg.norm(Z)
    if norm == 0:
        # Handle zero vector case, return a random point
        Z_rand = np.random.randn(n, k)
        return Z_rand / np.linalg.norm(Z_rand)
    return Z / norm


def project_tangent(Z, X):
    """
    Projects ambient vector Z onto the tangent space T_X V(n,k) at X.
    [cite_start]From Kim paper[cite: 77]: P_X(Z) = Z - 0.5 * (X(Z.T @ X) + (X @ X.T) @ Z)

    For n=2, k=1:
    X is 2x1, Z is 2x1.
    Z.T @ X is a 1x1 scalar.
    X @ X.T is a 2x2 matrix.
    """
    ztx = Z.T @ X  # 1x1
    xxt = X @ X.T  # 2x2

    # P_X(Z) = Z - 0.5 * (X * ztx + (X @ X.T) @ Z)
    P_X_Z = Z - 0.5 * (X * ztx + xxt @ Z)
    return P_X_Z


# --- 3. Main CBO Simulation Loop ---

def run_cbo_simulation(initial_particles):
    """
    Runs the full CBO simulation from the Kim et al. paper.
    """
    N_sim = initial_particles.shape[0]
    # History stores [step, particle_idx, dimension, 1]
    particle_history = np.zeros((num_steps, N_sim, n, k))
    particle_history[0] = initial_particles

    particles = initial_particles.copy()

    for step in tqdm(range(1, num_steps), desc="Simulating..."):
        # 1. Calculate f(X_i) for all particles
        f_values = np.array([objective_f(p) for p in particles])

        # 2. Calculate weighted average X_bar_t*
        f_min = np.min(f_values)
        weights = np.exp(-beta * (f_values - f_min))
        weights_sum = np.sum(weights)

        # Ensure weights_sum is not zero
        if weights_sum < 1e-10:
            weights = np.ones(N_sim)
            weights_sum = N_sim

        # X_bar_t = sum(w_i * X_i) / sum(w_i)
        X_bar_t = np.einsum('i,ijk->jk', weights, particles) / weights_sum

        # 3. Update each particle using Algorithm 1
        new_particles = np.zeros_like(particles)
        for i in range(N_sim):
            X_i = particles[i]

            # Calculate distance |X_i - X_bar_t|_F
            dist_fro = np.linalg.norm(X_i - X_bar_t, 'fro')

            # SDE (5) terms
            # Drift term: lambda * P_Xi(X_bar_t)
            drift_term = lambda_ * project_tangent(X_bar_t, X_i)

            # Correction term: C_nk * (sigma^2 * |dist|^2) / 2 * X_i
            correction_term = C_nk * (sigma ** 2 * dist_fro ** 2) / 2.0 * X_i

            # Noise term: sigma * |dist| * P_Xi(dW_t)
            dW = np.random.randn(n, k) * np.sqrt(h)
            noise_term = sigma * dist_fro * project_tangent(dW, X_i)

            # [cite_start]Predictor step [cite: 188]
            X_pred = X_i + (drift_term - correction_term) * h + noise_term

            # [cite_start]Corrector step [cite: 189]
            new_particles[i] = project_manifold(X_pred)

        particles = new_particles
        particle_history[step] = particles

    return particle_history


# --- 4. Plotting Function ---

def plot_simulation(history, title, X_star, X_trap):
    """
    Plots the trajectories of all particles on the unit circle.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the unit circle (the manifold V(2, 1))
    circle = plt.Circle((0, 0), 1, color='gray', fill=False,
                        linestyle='--', label='Manifold $V(2, 1)$')
    ax.add_artist(circle)

    # Plot trajectories
    # Squeeze history to (steps, N, n)
    hist_sq = history.squeeze()
    for i in range(N):
        ax.plot(hist_sq[:, i, 0], hist_sq[:, i, 1], alpha=0.3)

    # Plot initial and final positions
    ax.plot(hist_sq[0, :, 0], hist_sq[0, :, 1], 'o',
            color='blue', label='Initial Positions', markersize=8)
    ax.plot(hist_sq[-1, :, 0], hist_sq[-1, :, 1], 'x',
            color='black', label='Final Positions', markersize=10, mew=2)

    # Plot key points
    ax.plot(X_star[0], X_star[1], '*', color='green',
            label='Global Min $X^*$', markersize=20, mec='black')
    ax.plot(X_trap[0], X_trap[1], 'X', color='red',
            label='Local Max $X_a$ (Trap)', markersize=15, mec='black')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    plt.grid(True)
    return fig


# --- 5. Run Experiments ---

# Define the key points
X_star = np.array([[1.0], [0.0]])  # Global minimizer
X_trap = np.array([[-1.0], [0.0]])  # Local maximizer (antipodal point)

# --- Experiment 1: The "Success Case" ---
# Initialize particles in a cluster near the global minimum X*
print("Running 'Success Case' simulation...")
init_noise_min = np.random.randn(N, n, k) * 0.1
particles_min = np.array([project_manifold(X_star + init_noise_min[i])
                          for i in range(N)])
history_min = run_cbo_simulation(particles_min)
fig1 = plot_simulation(history_min,
                       "Success Case: Particles Start Near Global Minimum",
                       X_star, X_trap)

# --- Experiment 2: The "Failure Case" (Counterexample) ---
# Initialize particles in a cluster near the local maximum X_a
print("\nRunning 'Failure Case' (Counterexample) simulation...")
init_noise_trap = np.random.randn(N, n, k) * 0.1
particles_trap = np.array([project_manifold(X_trap + init_noise_trap[i])
                           for i in range(N)])
history_trap = run_cbo_simulation(particles_trap)
fig2 = plot_simulation(history_trap,
                       "Failure Case: Particles Start Near Local Maximum (Trap)",
                       X_star, X_trap)

plt.show()