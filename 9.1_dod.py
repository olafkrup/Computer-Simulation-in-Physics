import numpy as np
from numba import jit
import matplotlib.pyplot as plt

L = 200
T = 2

def A_start(L):
    A0 = np.random.randint(0, 2, (L, L))
    A0 = 2*(A0) - 1
    return A0

time = [10, 100, 1000, 10000, 50000]
time0 = time.copy()
for i in range(4):
    time0[i+1] = time0[i+1] - time0[i]


@jit(nopython=True)
def neighbours(r, L):
    r_next = np.zeros((2, 2), dtype=np.int64)

    for i in range(2):
        r_next[i, 0] = (r[i] - 1) % L
        r_next[i, 1] = (r[i] + 1) % L

    return np.array([
        [r[0], r_next[1, 0]],
        [r[0], r_next[1, 1]],
        [r_next[0, 0], r[1]],
        [r_next[0, 1], r[1]]
    ])

@jit(nopython=True)
def Monte_Carlo(L, A0, T):
    A = A0.copy()
    beta = 1/T
    for i in range(L**2):
        r1 = np.random.randint(0, L, 2)
        xy = np.random.randint(0, 2)
        delta_r2 = 2 * np.random.randint(0, 2) - 1
        if xy == 0:
            r2 = [r1[0], delta_r2]
        else:
            r2 = [delta_r2, r1[1]]

        n1 = neighbours(r1, L)
        n2 = neighbours(r2, L)

        # calculating the spin of our element
        s1 = A[r1[0], r1[1]]
        s2 = A[r2[0], r2[1]]
        delta1 = 0
        delta2 = 0

        for i in range(4):
            cord1 = n1[i]
            spin1 = A[cord1[0], cord1[1]]
            delta1 += 2*spin1

            cord2 = n2[i]
            spin2 = A[cord2[0], cord2[1]]
            delta2 += 2 * spin2

        delta1 = delta1 * s1
        delta2 = delta2 * s2
        delta = delta1 + delta2


        if delta < 0:
            s_ = s1
            s1 = s2
            s2 = s_
        else:
            x = np.random.rand()
            if x < np.exp(-beta * delta):
                s_ = s1
                s1 = s2
                s2 = s_

        A[r1[0], r1[1]] = s1
        A[r2[0], r2[1]] = s2

    return A

@jit(nopython=True)
def lots_Monte_carlo(L, A, T, t):
    for i in range(t):
        A = Monte_Carlo(L, A, T)
        print(i)

    return A

A = A_start(L)
A_list = []

for i in range(5):
    t = time0[i]
    A = lots_Monte_carlo(L, A, T, t)
    A_list.append(A)


fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(236)
axs = [ax1, ax2, ax3, ax4, ax5]

for i in range(5):
    axs[i].imshow(A_list[i])
    axs[i].set_title("Układ po " + str(time[i]) + " krokach")

plt.show()