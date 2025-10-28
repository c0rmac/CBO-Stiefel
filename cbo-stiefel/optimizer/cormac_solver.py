# filename: cormac_solver.py

import numpy as np
import warnings
from .base_solver import BaseCBOSolver  # Relative import from base_solver.py


class CormacsCBOSolver(BaseCBOSolver):
    """
    Concrete implementation featuring adaptive lambda and sigma (annealing + reheating).
    """

    def __init__(self,
                 n: int,
                 k: int,
                 cost_function: callable,
                 N: int = 200,
                 beta: float = 50.0,
                 lambda_initial: float = 1.0,
                 lambda_min: float = 0.1,
                 lambda_max: float = 100.0,
                 lambda_increase_factor: float = 1.1,
                 lambda_decrease_factor: float = 0.98,
                 lambda_adapt_interval: int = 20,
                 lambda_stagnation_thresh: float = 0.005,
                 lambda_convergence_thresh: float = 0.3,
                 sigma_initial: float = 0.5,
                 sigma_final: float = 0.001,
                 sigma_max: float = 1.0,
                 annealing_rate: float = 3.0,
                 reheat_check_interval: int = 50,
                 reheat_window: int = 100,
                 reheat_threshold: float = 1e-4,
                 reheat_sigma_boost: float = 0.2,
                 reheat_lambda_reset: bool = True,
                 suppress_warnings: bool = True):

        super().__init__(n, k, cost_function, N, beta, "numpy")

        # Store adaptive parameters
        self.lambda_initial = lambda_initial
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_increase_factor = lambda_increase_factor
        self.lambda_decrease_factor = lambda_decrease_factor
        self.lambda_adapt_interval = lambda_adapt_interval
        self.lambda_stagnation_thresh = lambda_stagnation_thresh
        self.lambda_convergence_thresh = lambda_convergence_thresh

        self.sigma_initial = sigma_initial
        self.sigma_final = sigma_final
        self.sigma_max = sigma_max
        self.annealing_rate = annealing_rate
        self.reheat_check_interval = reheat_check_interval
        self.reheat_window = reheat_window
        self.reheat_threshold = reheat_threshold
        self.reheat_sigma_boost = reheat_sigma_boost
        self.reheat_lambda_reset = reheat_lambda_reset

        self.suppress_warnings = suppress_warnings

    def _get_sigma_annealed(self, t, T):
        """Exponential annealing schedule."""
        if T <= 0: return self.sigma_final
        exponent = -min(700, self.annealing_rate * t / T)
        return max(self.sigma_final, self.sigma_initial * np.exp(exponent))

    def solve(self, T: float, h: float):
        """
        Runs the simulation with adaptive parameters.
        """
        if self.suppress_warnings:
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        num_steps = int(T / h)
        if num_steps <= 0: return {}

        self.particles = self._initialize_particles()

        # Initialize history arrays
        f_consensus_history = np.full(num_steps, np.nan)
        variance_history = np.full(num_steps, np.nan)
        lambda_history = np.zeros(num_steps)
        sigma_history = np.zeros(num_steps)

        lambda_current = self.lambda_initial
        sigma_current = self.sigma_initial
        variance_prev_check = 1.0
        f_consensus_min_window = float('inf')

        # --- Initial State Calculation ---
        X_bar_t_init = self._calculate_consensus_stable(self.particles)
        f_val_init = self.cost_function(X_bar_t_init)
        var_init = np.mean(np.linalg.norm(self.particles - X_bar_t_init[np.newaxis, :, :], axis=(1, 2)) ** 2)

        if np.isfinite(f_val_init): f_consensus_history[0] = f_val_init
        if np.isfinite(var_init): variance_history[0] = var_init
        variance_prev_check = variance_history[0] if np.isfinite(variance_history[0]) else 1.0

        # --- Main Time-Stepping Loop ---
        for step in range(num_steps):
            t = step * h

            # 1. Parameter Adaptation (occurs at the START of the step)
            sigma_annealed = self._get_sigma_annealed(t, T)
            sigma_current = sigma_annealed

            # Reheating
            if step > 0 and step % self.reheat_check_interval == 0:
                window_start = max(0, step - self.reheat_window)
                valid_f_in_window = f_consensus_history[window_start:step][
                    ~np.isnan(f_consensus_history[window_start:step])]
                f_min_now = np.min(valid_f_in_window) if len(valid_f_in_window) > 0 else float('inf')

                if np.isfinite(f_consensus_min_window) and np.isfinite(f_min_now):
                    improvement = f_consensus_min_window - f_min_now
                    if improvement < self.reheat_threshold and step >= self.reheat_window:
                        sigma_current = min(self.sigma_max, sigma_annealed + self.reheat_sigma_boost)
                        if self.reheat_lambda_reset:
                            lambda_current = self.lambda_initial
                    f_consensus_min_window = min(f_consensus_min_window, f_min_now)
                elif np.isfinite(f_min_now):
                    f_consensus_min_window = f_min_now

            # Lambda Adaptation
            if step > 0 and step % self.lambda_adapt_interval == 0:
                var_prev = variance_history[max(0, step - self.lambda_adapt_interval)]  # Var from start of interval
                var_current = variance_history[step - 1]  # Var from end of previous step

                if np.isfinite(var_prev) and np.isfinite(var_current) and variance_prev_check > 1e-12:
                    relative_decrease = (var_prev - var_current) / variance_prev_check
                    time_interval = h * self.lambda_adapt_interval
                    norm_stagnation_threshold = self.lambda_stagnation_thresh * time_interval
                    norm_convergence_threshold = self.lambda_convergence_thresh * time_interval

                    if relative_decrease < norm_stagnation_threshold:
                        lambda_current = min(lambda_current * self.lambda_increase_factor, self.lambda_max)
                    elif relative_decrease > norm_convergence_threshold:
                        lambda_current = max(lambda_current * self.lambda_decrease_factor, self.lambda_min)

                    variance_prev_check = var_current

            lambda_history[step] = lambda_current
            sigma_history[step] = sigma_current
            sigma_t_sq = sigma_current * sigma_current

            # 2. SDE Update
            X_bar_t = self._calculate_consensus_stable(self.particles)
            new_particles = np.zeros_like(self.particles)

            for i in range(self.N):
                X_i = self.particles[i]

                # Drift: P_X(X_bar - X_i) [NOTE: Differs from Kim's P_X(X_bar)]
                drift_consensus_part = self.project_tangent(X_bar_t - X_i, X_i)
                drift_term = lambda_current * drift_consensus_part

                correction_term = self.C_nk * sigma_t_sq / 2.0 * X_i

                dW = np.random.randn(self.n, self.k) * np.sqrt(h)
                noise_term = sigma_current * self.project_tangent(dW, X_i)

                X_pred = X_i + (drift_term - correction_term) * h + noise_term
                new_particles[i] = self.project_manifold(X_pred)

            self.particles = new_particles

            # 3. Store Metrics (based on state *after* the update)
            f_val = self.cost_function(X_bar_t)
            f_consensus_history[step] = f_val if np.isfinite(f_val) else f_consensus_history[max(0, step - 1)]
            current_variance = np.mean(np.linalg.norm(self.particles - X_bar_t[np.newaxis, :, :], axis=(1, 2)) ** 2)
            variance_history[step] = current_variance if np.isfinite(current_variance) else variance_history[
                max(0, step - 1)]

        # --- Final cleanup and packaging ---
        # Forward-fill and then replace any remaining NaNs with the last valid value
        def clean_history(history):
            if np.all(np.isnan(history)): return np.zeros_like(history)

            non_nan_indices = np.where(~np.isnan(history))[0]
            if len(non_nan_indices) == 0: return np.zeros_like(history)

            # Fill initial NaNs with the first valid value
            history[:non_nan_indices[0]] = history[non_nan_indices[0]]
            # Fill remaining NaNs (if any, with last valid value)
            history = np.nan_to_num(history, nan=history[non_nan_indices[-1]])
            return history

        results = {
            "solver": "Cormac",
            "f_history": clean_history(f_consensus_history),
            "var_history": clean_history(variance_history),
            "lambda_history": clean_history(lambda_history),
            "sigma_history": clean_history(sigma_history),
            "final_particles": self.particles,
            "final_M": self._calculate_consensus_stable(self.particles)
        }
        return results