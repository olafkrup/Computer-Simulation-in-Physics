import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap


colors = ['black', 'green', 'black', 'red']
cmap = ListedColormap(colors)

cmap = 'coolwarm'

p = 0.58  # probability of success
L = 50  # size of matrix
lattice = np.ones((L,L)) * (-1)
# setting the bounds
lattice[0, :] = -2
lattice[-1, :] = -2

# matrix of success
A = np.random.rand(L, L)
A = A < p

# setting the queue
queue = deque()
for i in range(0, L):
    queue.append([1, i])

fig, ax = plt.subplots()
ims = []
it = 0

while queue:
    r = queue[0]
    y = r[0]
    x = r[1]
    queue.popleft()
    if lattice[y][x] == -1:
        if A[y][x]:
            if x == L-1:
                x_next = [0, x-1]
            elif x == 0:
                x_next = [x + 1, L - 1]
            else:
                x_next = [x + 1, x - 1]
            y_next = [y - 1, y + 1]

            for y_ in y_next:
                if lattice[y_][x] == -1:
                    queue.append([y_, x])
            for x_ in x_next:
                if lattice[y][x_] == -1:
                    queue.append([y, x_])

            lattice[y][x] = 1
        else:
            lattice[y][x] = 0

        im = ax.imshow(lattice, interpolation='nearest', animated=True)
        if it % 5 == 0:
            ims.append([im])
        it += 1

im = ax.imshow(lattice, interpolation='nearest', animated=True)
ims.append([im])

animation0 = animation.ArtistAnimation(fig, ims, interval=20, repeat = True, repeat_delay=1000, blit=True)

plt.show()


