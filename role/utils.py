from __future__ import division

import numpy as np

# power iteration to solve the eigen-trust
def eigen_trust(A, alpha=0.1, epsilon=0.00001, max_iter=1000):
    # init
    n = A.shape[0]  # size
    v = np.ones(n)  # all-one vector
    d = A * v       # degree vector

    ind = np.argsort(d)[::-1]   # sort descend
    p = np.zeros(n)
    p[ind[0:3]] = 1

    # Power iteration:
    for i in range(max_iter):
        v_old = v.copy()
        v = (1-alpha) * A * v + alpha * p
        v = v / np.linalg.norm(v)
        delta = v - v_old
        if np.linalg.norm(delta) < epsilon:
            break

    return v
