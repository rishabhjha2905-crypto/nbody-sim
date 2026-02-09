import numpy as np
from nbody.physics import accelerations

def step_euler(r: np.ndarray, v: np.ndarray, m: np.ndarray, dt: float, G: float = 1.0, eps: float = 1e-2):
    """
    One Euler step:
      r_{new} = r + dt * v
      v_{new} = v + dt * a(r)
    """
    a = accelerations(r, m, G=G, eps=eps)
    r_new = r + dt * v
    v_new = v + dt * a
    return r_new, v_new
