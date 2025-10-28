# filename: accelerated_kim_solver.py

import numpy as np
import torch
import warnings
import time
import sys

# Assume base_solver.py contains BaseCBOSolver
from .base_solver import BaseCBOSolver


class AcceleratedKimCBOSolver(BaseCBOSolver):
    """
    Concrete implementation of Kim's fixed-parameter CBO (Algorithm 1).
    Leverages PyTorch tensors and targets the MPS device for accelerated execution
    via a vectorized SDE update.
    """

    def __init__(self,
                 n: int,
                 k: int,
                 cost_function: callable,
                 N: int = 50,  # Default N from Ackley V(3,5) experiment
                 beta: float = 5000.0,  # Default beta from Ackley V(3,5) experiment
                 lambda_: float = 1.0,  # Fixed lambda
                 sigma: float = 0.5,  # Fixed sigma
                 dtype: torch.dtype = torch.float32,
                 suppress_warnings: bool = True):

        # BaseCBOSolver handles device, self.device, self.dtype setup
        super().__init__(n, k, cost_function, N, beta, "torch", dtype)
        self.lambda_ = lambda_
        self.sigma = sigma
        self.suppress_warnings = suppress_warnings

        # print(f"[{self.__class__.__name__}] Using device: {self.device}")

    def _run_single_step_vectorized(self, h: float, current_particles):
        with torch.no_grad():
            M = self._calculate_consensus_stable(current_particles)
            diff_XM = current_particles - M.unsqueeze(0)
            dist_XM = torch.linalg.norm(diff_XM, ord='fro', dim=(1, 2))
            dW = torch.randn(self.N, self.n, self.k, dtype=self.dtype, device=self.device) * np.sqrt(h)

            # --- Batch Tangent Projection (Corrected einsum strings - Final Attempt) ---
            def batch_project_tangent_torch(Z_batch, X_batch):
                # Ensure Z_batch is broadcastable
                if Z_batch.ndim == 2: Z_batch = Z_batch.unsqueeze(0)  # (1, n, k)

                N_b, n_b, k_b = X_batch.shape

                try:
                    # Term 1: X @ (Z.T @ X)
                    # Z.T @ X -> einsum: Z.mT (Nki), X (Nij) -> sum over i (n), output (Nkj)
                    ZT_X = torch.einsum('Nki,Nij->Nkj', Z_batch.mT, X_batch)  # (N, k, k)
                    # X @ (Z.T @ X) -> einsum: X (Nij), ZT_X (Njk) -> sum over j (k), output (Nik)
                    X_ZTX = torch.einsum('Nij,Njk->Nik', X_batch, ZT_X)  # (N, n, k)

                    # Term 2: (X @ X.T) @ Z
                    # X @ X.T -> einsum: X (Nik), X.mT (Nkj) -> sum over k, output (Nij)
                    # CORRECTED EINSUM STRING FOR XXT:
                    XXT = torch.einsum('Nik,Nkj->Nij', X_batch, X_batch.mT)  # Shape (N, n, n)
                    # (X @ X.T) @ Z -> einsum: XXT (Nij), Z (Njk) -> sum over j (n), output (Nik)
                    XXT_Z = torch.einsum('Nij,Njk->Nik', XXT, Z_batch)  # Shape (N, n, k)

                except ImportError as e:
                    print(f"Einsum error during projection: {e}")
                    print(f"Shapes: Z={Z_batch.shape}, X={X_batch.shape}")
                    return torch.zeros_like(Z_batch)

                projection_term = X_ZTX + XXT_Z  # Both should be (N, n, k) now

                result = Z_batch - 0.5 * projection_term

                # Handle broadcast result from M
                if result.shape[0] == 1 and X_batch.shape[0] > 1:
                    result = result.repeat(X_batch.shape[0], 1, 1)

                if not torch.all(torch.isfinite(result)): return torch.zeros_like(result)
                return result

            # --- Rest of the SDE step ---
            PX_M_batch = batch_project_tangent_torch(M, current_particles)
            PX_dW_batch = batch_project_tangent_torch(dW, current_particles)

            sigma_sq = self.sigma * self.sigma
            drift_term = self.lambda_ * PX_M_batch
            correction_term_coeff = self.C_nk * sigma_sq * (dist_XM ** 2) / 2.0
            correction_term = correction_term_coeff.unsqueeze(-1).unsqueeze(-1) * current_particles
            noise_term = self.sigma * dist_XM.unsqueeze(-1).unsqueeze(-1) * PX_dW_batch

            X_pred_batch = current_particles + (drift_term - correction_term) * h + noise_term

            # --- Corrector Step (Loop for SVD) ---
            new_particles = torch.zeros_like(current_particles)
            for i in range(self.N):
                new_particles[i] = self.project_manifold(X_pred_batch[i])

            return new_particles, M

    def solve(self, T: float, h: float):
        """
        Runs the simulation using the vectorized PyTorch step.
        """
        if self.suppress_warnings:
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        num_steps = int(T / h)
        if num_steps <= 0: return {}

        self.particles = self._initialize_particles()

        f_consensus_history_list = []

        # --- Initial State Calculation ---
        with torch.no_grad():
            M_current = self._calculate_consensus_stable(self.particles)
            # Cost function requires NumPy, so transfer M to CPU/NumPy
            f_val_init_np = self.cost_function(M_current.cpu().numpy())
            f_consensus_history_list.append(f_val_init_np if np.isfinite(f_val_init_np) else np.nan)

        start_time = time.time()

        # --- Main Time-Stepping Loop ---
        for step in range(num_steps - 1):
            updated_particles, M_consensus = self._run_single_step_vectorized(h, self.particles)
            self.particles = updated_particles

            # Store results (transfer M_consensus to CPU for cost evaluation)
            f_val_np = self.cost_function(M_consensus.cpu().numpy())
            f_consensus_history_list.append(f_val_np if np.isfinite(f_val_np) else f_consensus_history_list[step])

        end_time = time.time()

        # --- Final cleanup and packaging ---
        f_history_np = np.array(f_consensus_history_list)

        # Helper function for final cleanup (retains the logic from previous implementations)
        def clean_history_np(history):
            if np.all(np.isnan(history)): return np.zeros_like(history)
            non_nan_mask = ~np.isnan(history)
            if not np.any(non_nan_mask): return np.zeros_like(history)
            first_valid_idx = np.where(non_nan_mask)[0][0]
            history[:first_valid_idx] = history[first_valid_idx]
            last_valid_val = history[first_valid_idx]
            for i in range(first_valid_idx + 1, len(history)):
                if np.isnan(history[i]):
                    history[i] = last_valid_val
                else:
                    last_valid_val = history[i]
            return history

        results = {
            "solver": "Kim_Torch_MPS",
            "f_history": clean_history_np(f_history_np),
            "final_particles": self.particles.cpu().numpy(),
            "final_M": self._calculate_consensus_stable(self.particles).cpu().numpy()
        }

        import multiprocessing
        if multiprocessing.current_process().name == 'MainProcess':
            print(f"Kim Solver (PyTorch MPS) Runtime: {end_time - start_time:.2f}s")

        return results