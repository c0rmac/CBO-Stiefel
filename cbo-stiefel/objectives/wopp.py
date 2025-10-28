# filename: wopp.py

import numpy as np


class WOPPObjective:
    """
    Implements the Weighted Orthogonal Procrustes Problem (WOPP) objective function.
    f(X) = 0.5 * ||A @ X @ C - B||_F^2

    The experiment in the paper used the high-dimensional case V(10, 20).
    """

    TRUE_MINIMUM = 0.0  # Guaranteed if B is constructed as A @ X* @ C [cite: 1499]

    def __init__(self, n, k, A_matrix=None, B_matrix=None, C_matrix=None):
        """
        Initializes the WOPP objective with problem dimensions and coefficient matrices.
        If A, B, C are None, they are generated randomly based on the paper's method.

        Args:
            n (int): Ambient dimension (rows of X, A).
            k (int): Rank (columns of X, C).
            A_matrix, B_matrix, C_matrix (np.ndarray): Coefficient matrices.
        """
        if k > n:
            raise ValueError(f"k ({k}) must be less than or equal to n ({n}).")

        self.n = n
        self.k = k

        if A_matrix is None or B_matrix is None or C_matrix is None:
            self._generate_random_matrices()
        else:
            self.A = A_matrix
            self.B = B_matrix
            self.C = C_matrix
            self._validate_shapes()

    def _validate_shapes(self):
        """Ensures input matrices have the correct dimensions."""
        if self.A.shape != (self.n, self.n):
            raise ValueError(f"A must be {self.n}x{self.n}, got {self.A.shape}")
        if self.B.shape != (self.n, self.k):
            raise ValueError(f"B must be {self.n}x{self.k}, got {self.B.shape}")
        if self.C.shape != (self.k, self.k):
            raise ValueError(f"C must be {self.k}x{self.k}, got {self.C.shape}")

    def _generate_random_matrices(self):
        """
        Generates matrices A and C based on the paper's method,
        and then generates B such that B = A @ X* @ C, ensuring TRUE_MINIMUM = 0.
        [cite: 1496-1499]
        """

        # 1. Generate A = P @ S @ R.T (P, R orthogonal; S diagonal U(0, 2))
        P = np.linalg.qr(np.random.randn(self.n, self.n))[0]
        R = np.linalg.qr(np.random.randn(self.n, self.n))[0]
        S = np.diag(np.random.uniform(0, 2, self.n))
        self.A = P @ S @ R.T

        # 2. Generate C = Q @ T @ Q.T (Q orthogonal; T diagonal U(0, 10))
        Q = np.linalg.qr(np.random.randn(self.k, self.k))[0]
        T = np.diag(np.random.uniform(0, 10, self.k))
        self.C = Q @ T @ Q.T

        # 3. Generate a random true minimizer X* on V(k, n)
        # Use QR decomposition of a random n x k matrix to get an orthonormal matrix
        X_star_rand = np.linalg.qr(np.random.randn(self.n, self.k))[0]

        # 4. Generate B = A @ X* @ C, ensuring the minimum cost is 0 at X=X* [cite: 1499]
        self.B = self.A @ X_star_rand @ self.C

    def __call__(self, X):
        """
        Computes f(X) = 0.5 * ||A @ X @ C - B||_F^2.

        Args:
            X (np.ndarray): The current n x k particle matrix.

        Returns:
            float: The value of f(X).
        """

        # Ensure X is 2D and has correct dimensions for calculation
        if X.ndim == 1 and self.k == 1:
            X = X.reshape(-1, 1)

        # The main WOPP calculation
        residual = self.A @ X @ self.C - self.B

        # Use np.linalg.norm with 'fro' for Frobenius norm squared
        f_X = 0.5 * np.linalg.norm(residual, 'fro') ** 2
        return f_X