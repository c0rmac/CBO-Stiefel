# filename: kim_solver.py

import numpy as np
from .base_solver import BaseCBOSolver  # Relative import from base_solver.py


class KimCBOSolver(BaseCBOSolver):
    """
    Concrete implementation based on the Kim et al. paper (Algorithm 1).
    Uses fixed lambda and sigma parameters and SVD projection.
    """

    def __init__(self,
                 n: int,
                 k: int,
                 cost_function: callable,
                 N: int = 50,
                 beta: float = 5000.0,
                 lambda_: float = 1.0,
                 sigma: float = 0.5):

        super().__init__(n, k, cost_function, N, beta, "numpy")
        self.lambda_ = lambda_
        self.sigma = sigma

    def _run_single_step(self, h: float, current_particles):
        """Performs one predictor-corrector step for all particles."""

        M = self._calculate_consensus_stable(current_particles)
        new_particles = np.zeros_like(current_particles)
        sigma_sq = self.sigma * self.sigma

        for i in range(self.N):
            Xi = current_particles[i]
            dist_XM = np.linalg.norm(Xi - M)

            # --- Predictor Step (Euler-Maruyama) ---
            PX_M = self.project_tangent(M, Xi)
            drift_term = self.lambda_ * PX_M

            correction_term_coeff = self.C_nk * sigma_sq * (dist_XM ** 2) / 2.0
            correction_term = correction_term_coeff * Xi

            dW = np.random.randn(self.n, self.k) * np.sqrt(h)
            PX_dW = self.project_tangent(dW, Xi)
            noise_term = self.sigma * dist_XM * PX_dW

            X_pred = Xi + (drift_term - correction_term) * h + noise_term

            # --- Corrector Step (Project back onto Manifold using SVD) ---
            new_particles[i] = self.project_manifold(X_pred)

        return new_particles, M

    def solve(self, T: float, h: float):
        """
        Runs the simulation for time T with step size h.
        """
        num_steps = int(T / h)
        if num_steps <= 0: return {}

        self.particles = self._initialize_particles()

        f_consensus_history = np.full(num_steps, np.nan)

        M_current = self._calculate_consensus_stable(self.particles)
        f_initial = self.cost_function(M_current)
        if np.isfinite(f_initial): f_consensus_history[0] = f_initial

        for step in range(num_steps):
            # Run one step of the SDE update
            updated_particles, M_consensus = self._run_single_step(h, self.particles)
            self.particles = updated_particles

            # Store results
            f_val = self.cost_function(M_consensus)
            if step < num_steps - 1:  # Only store up to num_steps-1 (which corresponds to T-h)
                f_consensus_history[step + 1] = f_val if np.isfinite(f_val) else f_consensus_history[step]

        # Fill any initial NaNs (if any, though should only be f_history[0])
        first_valid_idx = np.where(~np.isnan(f_consensus_history))[0]
        if len(first_valid_idx) > 0 and first_valid_idx[0] > 0:
            f_consensus_history[:first_valid_idx[0]] = f_consensus_history[first_valid_idx[0]]

        f_consensus_history = np.nan_to_num(f_consensus_history,
                                            nan=f_consensus_history[~np.isnan(f_consensus_history)][-1] if np.any(
                                                ~np.isnan(f_consensus_history)) else 0)

        results = {
            "solver": "Kim",
            "f_history": f_consensus_history,
            "final_particles": self.particles,
            "final_M": self._calculate_consensus_stable(self.particles)
        }
        return results