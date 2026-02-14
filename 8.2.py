import numpy as np
from numba import jit
import matplotlib.pyplot as plt

p = 0.5
L1 = 10
L2 = 20
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
def Monte_Carlo(L, A0, T):
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
                r_next[i, 0] = r[i] - 1  # Fixed indexing
                r_next[i, 1] = r[i] + 1  # Fixed indexing

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

        A[r[0], r[1]] = s

    return A

m_1 = []
m_2 = []

V_1 = []
V_2 = []

A_1 = A_start(L1)
A_2 = A_start(L2)

for T in T_:
    beta = 1/T
    m1 = np.empty(t_m)
    m2 = np.empty(t_m)
    A1 = A_1.copy()
    A2 = A_2.copy()
    for i in range(t0):
        A1 = Monte_Carlo(L1, A1, T)
        A2 = Monte_Carlo(L2, A2, T)
    for i in range(t_m):
        A1 = Monte_Carlo(L1, A1, T)
        A2 = Monte_Carlo(L2, A2, T)
        m1[i] = np.mean(A1)
        m2[i] = np.mean(A2)
    m_1.append(np.mean(np.abs(m1)))
    m_2.append(np.mean(np.abs(m2)))
    V_1.append(np.var(m1) * beta * L1**2)
    V_2.append(np.var(m2) * beta * L2**2)

m_1 = np.array(m_1)
m_2 = np.array(m_2)
V_1 = np.array(V_1)
V_2 = np.array(V_2)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

T_prim = np.linspace(1, 5, 1000)
f = (1 - 1/(np.sinh(2/T_prim))**4)**(1/8)


ax1.plot(T_, m_1, label = "L = 10")
ax1.plot(T_, m_2, label = "L = 20")
ax1.plot(T_, f, label = "Onsager")
ax1.legend()

ax2.plot(T_, V_1, label = "L = 10")
ax2.plot(T_, V_2, label = "L = 20")
ax2.legend()

plt.show()