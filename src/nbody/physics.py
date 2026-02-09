import numpy as np

def accelerations(r: np.ndarray, m: np.ndarray, G: float = 1.0, eps: float = 1e-2) -> np.ndarray:
    """
    Compute accelerations for an N-body system in 2D.

    Parameters
    ----------
    r : (N, 2) array
        Positions of bodies.
    m : (N,) array
        Masses of bodies.
    G : float
        Gravitational constant (scaled units).
    eps : float
        Softening parameter to avoid singularities.

    Returns
    -------
    a : (N, 2) array
        Accelerations on each body.
    """
    n = r.shape[0]
    a = np.zeros_like(r, dtype=float)

    for i in range(n):
        dr = r - r[i]                       # vector from i to every body j
        dist2 = np.sum(dr * dr, axis=1) + eps * eps
        dist2[i] = np.inf                   # ignore self-force
        inv_r3 = 1.0 / (dist2 * np.sqrt(dist2))
        a[i] = G * np.sum((m[:, None] * dr) * inv_r3[:, None], axis=0)

    return a
