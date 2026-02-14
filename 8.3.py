import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import scipy

p = 0.5
L = 50
delta_T = 0.1
T0 = 1
Tk = 5
N = int((Tk - T0)/delta_T + 1)
T_ = np.linspace(1, 5, N)

def A_start(L):
    A0 = np.random.rand(L, L)
    A0 = A0 < p
    A0 = A0.astype(int)
    A0 = 2*(A0 - 1)
    return A0

t0 = 2000
t_m = 5000


@jit(nopython=True)
def Monte_Carlo(L, A0, T, E_matrix):
    E_matrix2 = E_matrix.copy()
    A = A0.copy()
    beta = 1/T
    for i in range(L**2):
        r = np.random.randint(0, L, 2)

        r_next = np.zeros((2, 2), dtype=np.int64)

        for i in range(2):
            if r[i] == 0:
                r_next[i, 0] = 1
                r_next[i, 1] = L - 1
            elif r[i] == L - 1:
                r_next[i, 0] = L - 2
                r_next[i, 1] = 0
            else:
                r_next[i, 0] = r[i] - 1
                r_next[i, 1] = r[i] + 1

        neighbours = np.array([
            [r[0], r_next[1, 0]],
            [r[0], r_next[1, 1]],
            [r_next[0, 0], r[1]],
            [r_next[0, 1], r[1]]
        ], dtype=np.int64)

        # calculating the spin of our element
        delta = 0
        for i in range(neighbours.shape[0]):
            cord = neighbours[i]
            spin = A[cord[0], cord[1]]
            delta += 2*spin

        x = np.random.rand()
        if x < 1.0 / (1.0 + np.exp(- beta * delta)):
            s = 1
        else:
            s = -1

        E =  - s * delta / 2
        E_matrix2[r[0], r[1]] = E

        A[r[0], r[1]] = s

    return A, E_matrix2

A_ = A_start(L)
E_m = np.zeros((L, L))
c = []


for T in T_:
    beta = 1/T
    E = np.empty(t_m)
    E_matrix = E_m.copy()
    A = A_.copy()
    for i in range(t0):
        A, E_matrix = Monte_Carlo(L, A, T, E_matrix)
    for i in range(t_m):
        A, E_matrix = Monte_Carlo(L, A, T, E_matrix)
        E[i] = np.sum(E_matrix) / 2
    E02 = np.mean(E)**2
    E20 = np.mean(E**2)
    c.append(beta**2 * (E20 - E02) / L**2)


c = np.array(c)

fig = plt.figure()
ax1 = fig.add_subplot(111)

T_prim = np.linspace(1, 5, 1000)

def f(beta):
    kap = 2 * np.tanh(2 * beta) / np.cosh(2*beta)
    kap_ = 2 * (np.tanh(2 * beta) ** 2) - 1
    K1 = scipy.special.ellipk(kap**2)
    E1 = scipy.special.ellipe(kap**2)
    c = (2 / np.pi * L**2) * (beta * 1 / np.tanh(2 * beta) ) * (2 * K1 - 2 * E1 - (1 - kap_) * (np.pi/2 + kap_ * K1))
    return c


ax1.plot(T_, c, label = "Simulation")
ax1.plot(T_, f(1 / T_) / L**2, label = "Analytical")
ax1.legend()


plt.show()