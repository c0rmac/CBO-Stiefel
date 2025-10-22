import numpy as np

def project_stiefel(X):
  """Projects a matrix X onto the Stiefel manifold V(n, k) via QR decomposition."""
  try:
    q, r = np.linalg.qr(X)
    # Optional sign correction
    # d = np.diag(np.sign(np.diag(r)))
    # q = q @ d
    return q[:, :X.shape[1]]
  except np.linalg.LinAlgError:
    print("Warning: QR decomposition failed during projection.")
    # Fallback or error handling
    n, k = X.shape
    I_nk = np.zeros((n,k))
    I_nk[:k, :k] = np.eye(k)
    return project_stiefel(I_nk + 1e-6 * np.random.randn(n,k)) # Attempt recovery

def projection_tangent_space(X, Z):
  """Projects matrix Z onto the tangent space of V(n, k) at X."""
  # Using the form from Kim et al. / Thesis Approach
  term1 = X @ Z.T @ X
  term2 = X @ X.T @ Z
  return Z - 0.5 * (term1 + term2)