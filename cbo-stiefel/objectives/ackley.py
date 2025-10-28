# filename: ackley.py
import numpy as np


class AckleyObjective:
    DEFAULT_A = 20.0
    DEFAULT_B = 0.2
    DEFAULT_C = 3.0
    TRUE_MINIMUM = 0.0

    def __init__(self, n, k, a=DEFAULT_A, b=DEFAULT_B, c=DEFAULT_C):
        if k > n:
            raise ValueError(f"k ({k}) must be less than or equal to n ({n}).")

        self.n = n
        self.k = k
        self.a = a
        self.b = b
        self.c = c
        self.nk = float(n * k)

        self.X_star = np.zeros((n, k))
        self.X_star[:k, :k] = np.eye(k)

    def __call__(self, X):

        # Fix for k=1 (single column) arrays, which NumPy might pass as 1D (n,)
        if X.ndim == 1 and self.k == 1:
            X = X.reshape(-1, 1)  # Reshape (n,) to (n, 1)

        # The check that is failing for you
        if X.shape != (self.n, self.k):
            raise ValueError(
                f"Shape mismatch: Input matrix X has shape {X.shape}, "
                f"but the objective was initialized for shape ({self.n}, {self.k})."
            )

        diff = X - self.X_star

        term1_inner = np.sqrt((self.c ** 2 / self.nk) * np.sum(diff ** 2))
        term1 = -self.a * np.exp(-self.b * term1_inner)

        term2_inner = np.sum(np.cos(2 * np.pi * self.c * diff)) / self.nk
        term2 = -np.exp(term2_inner)

        f_X = term1 + term2 + np.e + self.a
        return f_X