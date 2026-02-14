import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 50
grid0 = np.zeros((N, N))
grid0 = grid0 + 7
T = 5000
h = 4

def Add(grid, N):
    A = grid.copy()
    xy = np.random.randint(1, N, 2)
    A[xy[0], xy[1]] = A[xy[0], xy[1]] + 1
    return A

def Boundary(grid, N):
    A = grid.copy()
    A[0, :] = 0
    A[N-1, :] = 0
    A[:, 0] = 0
    A[:, N-1] = 0
    return A

def Avalanche(grid, N, h):
    A = grid.copy()
    A = Boundary(A, N)
    x_t, y_t = np.where(A > h)
    for i in range(len(x_t)):
        x = x_t[i]
        y = y_t[i]
        A[x, y] = 0
        A[x+1, y] = A[x+1, y] + 1
        A[x, y+1] = A[x, y+1] + 1
        A[x-1, y] = A[x-1, y] + 1
        A[x, y-1] = A[x, y-1] + 1
    A = Boundary(A, N)
    return A

def Evolve(grid, N, h, T):
    A = grid.copy()
    u = []
    for i in range(T):
        A = Avalanche(A, N, h)
        u.append(A)
        print(i)

    return u

u = Evolve(grid0, N, h, T)
fig, ax = plt.subplots()

ims = []
for i in range(0, T, 100):
    im = ax.imshow(u[i], animated=True, interpolation="none", cmap='rainbow', vmin=0, vmax=8)

    ims.append([im])

animation0 = animation.ArtistAnimation(fig, ims, interval=50, repeat = True, repeat_delay=1000, blit=True)
plt.show()