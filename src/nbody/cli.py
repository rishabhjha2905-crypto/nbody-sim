import numpy as np
from nbody.integrators import step_euler
from nbody.physics import accelerations

def main():
    # Two-body setup
    r = np.array([[0.0, 0.0],
                  [1.0, 0.0]])
    v = np.array([[0.0, 0.0],
                  [0.0, 0.0]])
    m = np.array([1.0, 1.0])

    dt = 0.1

    print("Initial r:\n", r)
    print("Initial v:\n", v)
    print("Initial a:\n", accelerations(r, m))

    r2, v2 = step_euler(r, v, m, dt=dt)

    print("\nAfter 1 Euler step (dt=0.1):")
    print("r:\n", r2)
    print("v:\n", v2)

if __name__ == "__main__":
    main()
