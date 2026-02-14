import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit

L = 1001

A = np.zeros((L, L), dtype=np.int64)
r0 = np.array([int(L/2), int(L/2)], dtype=np.int64)
A[r0[0], r0[1]] = -10

R = 10


cmap = 'magma'

@jit(nopython = True)
def init_particle(r0, R):
    x = (np.random.rand() - 0.5) * 2*R
    y = (R ** 2 - x ** 2) ** 0.5
    if np.random.rand() < 0.5:
        y = -y
        y = int(y - 0.5)
    else:
        y = int(y + 0.5)

    x = int(x + 0.5)

    return np.array([r0[0] + x, r0[1] + y], dtype = np.int64)

@jit(nopython = True)
def distance(r1, r2):
    r = [0, 0]
    for i in (0, 1):
        r[i] = r1[i] - r2[i]

    return (r[0]**2 + r[1]**2)**0.5


@jit(nopython = True)
def step(r, L):
    delta = np.random.randint(0, 2)
    delta = delta * 2 - 1
    if np.random.rand() < 0.5:
        r_next = [r[0], (r[1] + delta) % L]
    else:
        r_next = [(r[0] + delta) % L, r[1]]
    r_next = np.array(r_next, dtype=np.int64)
    return r_next


fig, ax = plt.subplots()


@jit(nopython=True)
def sim(A, L, r0, M):
    A_ = A.copy()
    mtrx = []
    size = 0
    r_max = 0.0
    while size < 1000:
        print(size)
        if size % 10 == 0:
            mtrx.append(A_.copy())
        R = 10 + r_max
        R_kill = float(R + 150)
        r = init_particle(r0, R)
        dist = distance(r, r0)
        while(dist < R_kill):
            r_next = step(r, L)
            if A_[r_next[0], r_next[1]] < 0:
                if A_[r_next[0], r_next[1]] == -M:
                    A_[r[0], r[1]] = -1
                    size += 1
                    print(size)
                    d = distance(r, r0)
                    if d > r_max:
                        r_max = d
                    break
                else:
                    A_[r_next[0], r_next[1]] = A_[r_next[0], r_next[1]] - 1
                    break
            r = r_next
            dist = distance(r, r0)
    return mtrx


mtrx = sim(A, L, r0, 10)

ims = []

for i in range(len(mtrx)):
    A0 = mtrx[i]
    im = ax.imshow(A0[400:600, 400:600], interpolation='nearest', cmap=cmap, animated=True)
    ims.append([im])

anim0 = animation.ArtistAnimation(fig, ims, interval = 50, repeat = True, repeat_delay=1000)
anim0.save("103.gif")

plt.show()