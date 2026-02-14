import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
#   TRANSFORMATIONS
# ---------------------------

# Sierpiński triangle IFS
m = [
    [0.5, 0,   0, 0.5, 0.25, np.sqrt(3)/4],
    [0.5, 0,   0, 0.5, 0.0,  0.0],
    [0.5, 0,   0, 0.5, 0.5,  0.0]
]

# Barnsley fern IFS
n = [
    [0.001, 0.0,   0.0, 0.16, 0.0, 0.0],
    [-0.15, 0.28,  0.26, 0.24, 0.0, 0.44],
    [0.20, -0.26,  0.23, 0.22, 0.0, 1.60],
    [0.85, 0.04,  -0.04, 0.85, 0.0, 1.60]
]

p_m = 1/3
p_n = np.array([0.02, 0.09, 0.10, 0.79])
p_n_cum = np.cumsum(p_n)


# ---------------------------
#   FUNCTIONS
# ---------------------------

def sym_m(v):
    """Random Sierpiński transform."""
    x = np.random.rand()
    if x < p_m:
        m0 = m[0]
    elif x < 2*p_m:
        m0 = m[1]
    else:
        m0 = m[2]

    M = np.array(m0[:4]).reshape(2, 2)
    w = np.array(m0[4:])
    return M @ v + w


def sym_n(v):
    """Random Barnsley fern transform with correct probabilities."""
    x = np.random.rand()
    if x < p_n_cum[0]:
        n0 = n[0]
    elif x < p_n_cum[1]:
        n0 = n[1]
    elif x < p_n_cum[2]:
        n0 = n[2]
    else:
        n0 = n[3]

    N = np.array(n0[:4]).reshape(2, 2)
    w = np.array(n0[4:])
    return N @ v + w


# ---------------------------
#   ITERATION
# ---------------------------

N = 200_000          # much nicer pictures
burn = 200           # skip initial iterations
v0 = np.random.rand(2)

a = np.empty((N, 2))
b = np.empty((N, 2))
a[0] = v0
b[0] = v0

for i in range(N - 1):
    a[i+1] = sym_m(a[i])
    b[i+1] = sym_n(b[i])


# ---------------------------
#   PLOTTING
# ---------------------------

plt.figure(figsize=(7, 12))

plt.subplot(211)
plt.scatter(a[burn:, 0], a[burn:, 1], s=0.1, color='red')
plt.title("Sierpiński Triangle (IFS)")

plt.subplot(212)
plt.scatter(b[burn:, 0], b[burn:, 1], s=0.1, color='green')
plt.title("Barnsley Fern (IFS)")

plt.tight_layout()
plt.show()