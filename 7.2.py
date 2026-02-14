import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque


def matrix(p, L):
    A = np.random.rand(L, L)
    A = A < p

    lattice = np.ones((L, L)) * (-1)
    # setting the bounds
    lattice[0, :] = -2
    lattice[-1, :] = -2

    return A, lattice

def percolation(A, lattice, L):
    lattice0 = lattice.copy()
    # setting the queue
    queue = deque()
    cluster = []

    while np.sum(lattice0[1, :] == -1) > 0:
        for i in range(0, L):
            if lattice0[1, i] == -1:
                queue.appendleft([1, i])
                break
        cluster0 = []
        while queue:
            r = queue[0]
            y = r[0]
            x = r[1]
            queue.popleft()
            if lattice0[y][x] == -1:
                if A[y][x]:
                    if x == L-1:
                        x_next = [0, x-1]
                    elif x == 0:
                        x_next = [x + 1, L - 1]
                    else:
                        x_next = [x + 1, x - 1]

                    y_next = [y - 1, y + 1]

                    for y_ in y_next:
                        if lattice0[y_][x] == -1:
                            queue.appendleft([y_, x])
                    for x_ in x_next:
                        if lattice0[y][x_] == -1:
                            queue.appendleft([y, x_])

                    lattice0[y][x] = 1
                    cluster0.append(y)
                else:
                    lattice0[y][x] = 0
        cluster.append(cluster0)
    whether = np.sum(lattice0[-2, :] == 1) > 0
    S = np.sum(lattice0==1) / (L**2)
    if whether:
        for c in cluster:
            if L-2 in c:
                S = S - ( len(c) /(L**2) )
    return S, whether


L = [20, 50, 100]

p0 = 0.5
pk = 0.7
delta_p = 0.01
N = int((pk - p0)/delta_p + 2)

n = 100

data = []

x = 0

for p in np.linspace(p0, pk, N):
    prob = np.zeros((3, 1))
    S = np.zeros((3, 1))
    print(p)
    x+= 1
    for i in range(n):
        for j in range(3):
            l = L[j]
            A, lattice = matrix(p, l)
            S_, whether = percolation(A, lattice, l)
            S[j] += S_
            prob[j] += int(whether)

    S = S / n
    prob = prob / n

    data_ = []

    for j in range(3):
        data_.append([float(S[j]), float(p)])

    for j in range(3):
        data_.append([float(prob[j]), float(p)])

    data0 = np.array(data_).reshape(1, 6, 2)

    data.append(data0)

data_ = np.vstack(data)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

for j in range(3):
    y_data = data_[:, j, 0]
    x_data = data_[:, j, 1]
    ax1.plot(x_data, y_data, label = "S(p) for L = " + str(L[j]))

ax1.set_title("Average cluster size for different success probability")
ax1.set_xlabel("p")
ax1.set_ylabel("S")
ax1.legend()

for j in range(3, 6):
    y_data = data_[:, j, 0]
    x_data = data_[:, j, 1]
    ax2.plot(x_data, y_data, label = "P(p) for L = " + str(L[j-3]))

ax2.set_title("Probability of percolation for different success probability")
ax2.set_xlabel("p")
ax2.set_ylabel("P")
ax2.legend()

plt.show()

