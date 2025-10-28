# filename: accelerated_cormac_solver.py

import numpy as np
import torch
import warnings
import time
import sys
from abc import ABC, abstractmethod

# Assume base_solver.py contains BaseCBOSolver
from .base_solver import BaseCBOSolver


class AcceleratedCormacCBOSolver(BaseCBOSolver):
    """
    Concrete implementation of CBO using the Cayley Transformation scheme (Algorithm 2).
    Leverages PyTorch tensors and targets the MPS device for acceleration.
    Features adaptive lambda and sigma (annealing + reheating).
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
                 suppress_warnings: bool = True,
                 dtype: torch.dtype = torch.float32):

        super().__init__(n, k, cost_function, N, beta, "torch", dtype)

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

        print(f"[{self.__class__.__name__}] Using device: {self.device}")
        self.identity_n = torch.eye(self.n, dtype=self.dtype, device=self.device)

    # --- Adaptive Parameter Schedule ---

    def _get_sigma_annealed(self, t, T):
        """Exponential annealing schedule."""
        if T <= 0: return self.sigma_final
        exponent = -min(700, self.annealing_rate * t / T)
        # Use np.exp for scalar schedule calculation
        return max(self.sigma_final, self.sigma_initial * np.exp(exponent))

        # --- Main Solve Method (using PyTorch) ---

    def solve(self, T: float, h: float):
        """
        Runs the simulation with adaptive parameters using PyTorch tensors
        and the Cayley Transform update.
        """
        if self.suppress_warnings:
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        num_steps = int(T / h)
        if num_steps <= 0: return {}

        self.particles = self._initialize_particles()

        f_consensus_history = np.full(num_steps, np.nan)
        variance_history = np.full(num_steps, np.nan)
        lambda_history = np.zeros(num_steps)
        sigma_history = np.zeros(num_steps)

        lambda_current = self.lambda_initial
        sigma_current = self.sigma_initial
        variance_prev_check = 1.0
        f_consensus_min_window = float('inf')

        # --- Initial State Calculation ---
        with torch.no_grad():
            X_bar_t_init = self._calculate_consensus_stable(self.particles)
            f_val_init_np = self.cost_function(X_bar_t_init.cpu().numpy())
            var_init = torch.mean(torch.linalg.norm(self.particles - X_bar_t_init.unsqueeze(0), dim=(1, 2)) ** 2).item()

            if np.isfinite(f_val_init_np): f_consensus_history[0] = f_val_init_np
            if np.isfinite(var_init): variance_history[0] = var_init
            variance_prev_check = variance_history[0] if np.isfinite(variance_history[0]) else 1.0

        start_time = time.time()

        # --- Main Time-Stepping Loop ---
        for step in range(num_steps):
            t = step * h

            with torch.no_grad():

                # 1. Parameter Adaptation (occurs at the START of the step)
                sigma_annealed = self._get_sigma_annealed(t, T)
                sigma_current = sigma_annealed

                # Reheating (logic uses NumPy history list)
                if step > 0 and step % self.reheat_check_interval == 0:
                    window_start = max(0, step - self.reheat_window)
                    f_history_window = np.array(f_consensus_history[window_start:step])
                    valid_f_in_window = f_history_window[~np.isnan(f_history_window)]
                    f_min_now = np.min(valid_f_in_window) if len(valid_f_in_window) > 0 else float('inf')

                    if np.isfinite(f_consensus_min_window) and np.isfinite(f_min_now):
                        improvement = f_consensus_min_window - f_min_now
                        if improvement < self.reheat_threshold and step >= self.reheat_window:
                            sigma_current = min(self.sigma_max, sigma_annealed + self.reheat_sigma_boost)
                            if self.reheat_lambda_reset: lambda_current = self.lambda_initial
                        f_consensus_min_window = min(f_consensus_min_window, f_min_now)
                    elif np.isfinite(f_min_now):
                        f_consensus_min_window = f_min_now

                # Lambda Adaptation (logic uses NumPy history list)
                if step > 0 and step % self.lambda_adapt_interval == 0:
                    var_prev = variance_history[max(0, step - self.lambda_adapt_interval)]
                    var_current = variance_history[step - 1]

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

                # 2. SDE Update (Cayley Transform)
                X_bar_t = self._calculate_consensus_stable(self.particles)
                new_particles = torch.zeros_like(self.particles)

                for i in range(self.N):
                    Xi = self.particles[i]
                    dist_XM = torch.linalg.norm(Xi - X_bar_t)

                    # --- Predictor Step (Calculate Displacement Z_ni) ---
                    # Drift: P_X(X_bar - X_i) [Note: Cormac's SDE uses X_bar - X_i in P_X]
                    drift_consensus_part = self.project_tangent(X_bar_t - Xi, Xi)
                    drift_term = lambda_current * drift_consensus_part

                    correction_term = self.C_nk * sigma_t_sq / 2.0 * Xi

                    # Noise
                    dW = torch.randn(self.n, self.k, dtype=self.dtype, device=self.device) * np.sqrt(h)
                    noise_term = sigma_current * self.project_tangent(dW, Xi)

                    # Z_ni (Displacement vector)
                    Z_ni = (drift_term - correction_term) * h + noise_term

                    # --- Corrector Step (Cayley Transform) ---
                    A = Z_ni @ Xi.T
                    W_ni = A - A.T  # Skew-symmetric (n, n) matrix

                    left_matrix = self.identity_n - 0.5 * W_ni
                    right_vector_part = (self.identity_n + 0.5 * W_ni) @ Xi

                    try:
                        # Solve linear system A*X = B for X, using torch.linalg.solve
                        X_next_i = torch.linalg.solve(left_matrix, right_vector_part)
                        new_particles[i] = X_next_i

                    except Exception:  # Catch PyTorch LinAlgError/RuntimeError
                        new_particles[i] = Xi  # Failsafe

                self.particles = new_particles

                # 3. Store Metrics
                f_val_np = self.cost_function(X_bar_t.cpu().numpy())

                # Update history lists
                f_consensus_history[step] = f_val_np if np.isfinite(f_val_np) else f_consensus_history[max(0, step - 1)]
                current_variance = torch.mean(
                    torch.linalg.norm(self.particles - X_bar_t.unsqueeze(0), dim=(1, 2)) ** 2).item()
                variance_history[step] = current_variance if np.isfinite(current_variance) else variance_history[
                    max(0, step - 1)]

        end_time = time.time()

        # --- Final cleanup and packaging (using NumPy) ---
        f_history_np = np.array(f_consensus_history)
        var_history_np = np.array(variance_history)
        lambda_history_np = np.array(lambda_history)
        sigma_history_np = np.array(sigma_history)

        def clean_history_np(history):
            non_nan_mask = ~np.isnan(history)
            if not np.any(non_nan_mask): return np.zeros_like(history)
            history = np.copy(history)  # Work on copy
            first_valid_idx = np.where(non_nan_mask)[0][0]
            history[:first_valid_idx] = history[first_valid_idx]
            last_valid_val = history[first_valid_idx]
            for i in range(first_valid_idx + 1, len(history)):
                if np.isnan(history[i]):
                    history[i] = last_valid_val
                else:
                    last_valid_val = history[i]
            return history

        final_M_np = self._calculate_consensus_stable(self.particles).cpu().numpy()

        results = {
            "solver": "Cormac_Torch_MPS",
            "f_history": clean_history_np(f_history_np),
            "var_history": clean_history_np(var_history_np),
            "lambda_history": clean_history_np(lambda_history_np),
            "sigma_history": clean_history_np(sigma_history_np),
            "final_particles": self.particles.cpu().numpy(),
            "final_M": final_M_np
        }

        import multiprocessing
        if multiprocessing.current_process().name == 'MainProcess':
            print(f"Accelerated Cormac Solver (PyTorch MPS) Runtime: {end_time - start_time:.2f}s")

        return results