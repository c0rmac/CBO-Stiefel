# filename: base_solver.py

import numpy as np
from scipy.linalg import svd as scipy_svd
from abc import ABC, abstractmethod
import torch
import time


class BaseCBOSolver(ABC):
    """
    Abstract Base Class for CBO solvers on V(n,k).

    This version is backend-agnostic, supporting NumPy/SciPy (default) or PyTorch
    for geometric and tensor operations based on the 'backend' parameter.
    """

    def __init__(self,
                 n: int,
                 k: int,
                 cost_function: callable,
                 N: int,
                 beta: float,
                 backend: str = 'numpy',  # New parameter to select backend
                 dtype: torch.dtype = torch.float32):
        """
        Initializes common parameters and sets up the device/backend.
        """
        if k > n:
            raise ValueError(f"k ({k}) must be less than or equal to n ({n}).")
        if backend not in ['numpy', 'torch']:
            raise ValueError("Backend must be 'numpy' or 'torch'.")

        self.n = n
        self.k = k
        self.cost_function = cost_function
        self.N = N
        self.beta = beta
        self.backend = backend
        self.dtype = dtype

        # --- Device/Backend Setup ---
        if self.backend == 'torch':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.device = torch.device("cpu")
            self.identity_k = torch.eye(k, dtype=dtype, device=self.device)
            self.identity_n = torch.eye(n, dtype=dtype, device=self.device)
        else:  # NumPy backend
            self.device = 'cpu'
            self.identity_k = np.eye(k)
            self.identity_n = np.eye(n)

        self.C_nk = (2.0 * self.n - self.k - 1.0) / 2.0
        self.particles = None

    # ======================================================================
    # 1. GEOMETRIC OPERATIONS (CONDITIONAL DISPATCH)
    # ======================================================================

    # NOTE: The public methods below dispatch to the appropriate static helper.

    def project_tangent(self, Z, X):
        """Dispatches to the correct tangent projection based on backend."""
        if self.backend == 'torch':
            return self._project_tangent_torch(Z, X)
        else:
            return self._project_tangent_numpy(Z, X)

    def project_manifold(self, Z):
        """Dispatches to the correct manifold projection (retraction)."""
        if self.backend == 'torch':
            return self._project_manifold_torch(Z)
        else:
            return self._project_manifold_numpy(Z)

    # ======================================================================
    # 1A. STATIC HELPERS (NumPy/SciPy)
    # ======================================================================

    @staticmethod
    def _project_tangent_numpy(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Projects Z onto tangent space at X using NumPy/SciPy."""
        if not np.all(np.isfinite(Z)): Z = np.zeros_like(Z)
        if not np.all(np.isfinite(X)): return np.zeros_like(Z)

        projection = Z - 0.5 * (X @ (Z.T @ X) + X @ (X.T @ Z))

        if not np.all(np.isfinite(projection)): return np.zeros_like(Z)
        return projection

    @staticmethod
    def _project_manifold_numpy(Z: np.ndarray) -> np.ndarray:
        """Projects Z onto V(n,k) via SVD using SciPy/NumPy."""
        if not np.all(np.isfinite(Z)): Z = np.random.randn(*Z.shape)
        try:
            U, _, Vh = scipy_svd(Z, full_matrices=False)
            proj = U @ Vh
            if not np.all(np.isfinite(proj)):
                raise np.linalg.LinAlgError("SVD resulted in non-finite values.")
            return proj
        except np.linalg.LinAlgError:
            Z_reinit = np.random.randn(*Z.shape)
            U_reinit, _, Vh_reinit = scipy_svd(Z_reinit, full_matrices=False)
            return U_reinit @ Vh_reinit

    # ======================================================================
    # 1B. STATIC HELPERS (PyTorch)
    # ======================================================================

    @staticmethod
    def _project_tangent_torch(Z: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Projects Z onto tangent space at X using PyTorch."""
        if not torch.all(torch.isfinite(Z)): Z = torch.zeros_like(Z)
        if not torch.all(torch.isfinite(X)): return torch.zeros_like(Z)
        try:
            Z_T = Z.T
            X_T = X.T
            term1 = X @ Z_T @ X
            term2 = X @ X_T @ Z
            projection = Z - 0.5 * (term1 + term2)
        except RuntimeError as e:
            return torch.zeros_like(Z)
        if not torch.all(torch.isfinite(projection)): return torch.zeros_like(Z)
        return projection

    @staticmethod
    def _project_manifold_torch(Z: torch.Tensor) -> torch.Tensor:
        """Projects Z onto V(n,k) via SVD using PyTorch."""
        if not torch.all(torch.isfinite(Z)): Z = torch.randn_like(Z)
        try:
            U, _, Vh = torch.linalg.svd(Z, full_matrices=False)
            proj = U @ Vh
            if not torch.all(torch.isfinite(proj)):
                raise RuntimeError("SVD resulted in non-finite values.")
            return proj
        except Exception:
            Z_reinit = torch.randn_like(Z)
            U_reinit, _, Vh_reinit = torch.linalg.svd(Z_reinit, full_matrices=False)
            return U_reinit @ Vh_reinit

    # ======================================================================
    # 2. CORE CBO LOGIC (Backend-Agnostic State Management)
    # ======================================================================

    def _initialize_particles(self):
        """Initializes N particles uniformly on V(n,k) using the selected backend."""

        if self.backend == 'torch':
            particles = torch.zeros((self.N, self.n, self.k), dtype=self.dtype, device=self.device)
            for i in range(self.N):
                Z = torch.randn(self.n, self.k, dtype=self.dtype, device=self.device)
                particles[i] = self.project_manifold(Z)
        else:  # NumPy
            particles = np.zeros((self.N, self.n, self.k))
            for i in range(self.N):
                Z = np.random.randn(self.n, self.k)
                particles[i] = self.project_manifold(Z)
        return particles

    def _calculate_consensus_stable(self, particles):
        """Calculates the Boltzmann-weighted consensus point X_bar using the selected backend."""

        # 1. Cost Evaluation (always uses NumPy/Python interface)
        costs = np.array([self.cost_function(self._to_numpy(p)) for p in particles])

        if self.backend == 'torch':
            # 2. Convert costs and particles to Tensors on device
            costs_tensor = torch.tensor(costs, dtype=self.dtype, device=self.device)
            particles_tensor = particles

            # Robust Consensus Calculation (on device)
            finite_costs = costs_tensor[torch.isfinite(costs_tensor)]
            min_cost = torch.min(finite_costs) if len(finite_costs) > 0 else torch.tensor(0.0, device=self.device)

            weights_unnormalized = torch.exp(-self.beta * (costs_tensor - min_cost))
            weights_unnormalized[~torch.isfinite(costs_tensor)] = 0.0
            sum_weights = torch.sum(weights_unnormalized)

            if sum_weights < 1e-100 or not torch.isfinite(sum_weights):
                valid_particles = particles_tensor[torch.isfinite(costs_tensor)]
                return torch.mean(valid_particles, dim=0) if len(valid_particles) > 0 else torch.mean(particles_tensor,
                                                                                                      dim=0)

            normalized_weights = weights_unnormalized / sum_weights
            consensus_point = torch.einsum('i,ijk->jk', normalized_weights, particles_tensor)

            return consensus_point

        else:  # NumPy Backend
            # 2. Consensus Calculation (NumPy)
            finite_costs = costs[np.isfinite(costs)]
            min_cost = np.min(finite_costs) if len(finite_costs) > 0 else 0.0

            weights_unnormalized = np.exp(-self.beta * (costs - min_cost))
            weights_unnormalized[~np.isfinite(costs)] = 0.0
            sum_weights = np.sum(weights_unnormalized)

            if sum_weights < 1e-100 or not np.isfinite(sum_weights):
                valid_particles = particles[np.isfinite(costs)]
                return np.mean(valid_particles, axis=0) if len(valid_particles) > 0 else np.mean(particles, axis=0)

            normalized_weights = weights_unnormalized / sum_weights
            consensus_point = np.einsum('i,ijk->jk', normalized_weights, particles)

            return consensus_point

    def _to_numpy(self, tensor_or_array):
        """Safely converts PyTorch tensor to NumPy array for cost evaluation, otherwise returns array."""
        if self.backend == 'torch':
            if isinstance(tensor_or_array, torch.Tensor):
                return tensor_or_array.cpu().numpy()
        return tensor_or_array  # Already NumPy/Python type

    def _to_tensor(self, array):
        """Converts NumPy array to PyTorch tensor on the correct device."""
        if self.backend == 'torch':
            return torch.tensor(array, dtype=self.dtype, device=self.device)
        return array

    @abstractmethod
    def solve(self, T: float, h: float):
        """
        Runs the CBO simulation for time T with step size h.
        Subclasses MUST implement this method.
        """
        pass