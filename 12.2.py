import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N = 32
grid0 = np.zeros((N, N))
T = 50000
h = 3

def Add(grid, N):
    A = grid.copy()
    xy = np.random.randint(1, N-1, 2)
    A[xy[0], xy[1]] = A[xy[0], xy[1]] + 1
    return A

def Boundary(grid):
    A = grid.copy()
    A[0, :] = 0
    A[-1, :] = 0
    A[:, 0] = 0
    A[:, -1] = 0
    return A

def Avalanche(grid, h):
    A = grid.copy()
    it = 0
    while np.max(A) > h:
        x_t, y_t = np.where(A > h)
        for i in range(len(x_t)):
            x = x_t[i]
            y = y_t[i]
            A[x, y] = A[x,y] - 4
            A[x+1, y] = A[x+1, y] + 1
            A[x, y+1] = A[x, y+1] + 1
            A[x-1, y] = A[x-1, y] + 1
            A[x, y-1] = A[x, y-1] + 1

            it += 1
        A = Boundary(A)
    return A, it

def Evolve(grid, N, h, T):
    A = grid.copy()
    it_list = []
    for i in range(T):
        while np.max(A) <= h:
            A = Add(A, N)
        A, it = Avalanche(A, h)
        if it == 0:
            print("?")
        it_list.append(it)
        print(i)

    return it_list

it_list = Evolve(grid0, N, h, T)

hist, bins = np.histogram(it_list, 100)

plt.scatter(bins[0:-1], hist)

def f(x, a):
    return a / x

popt, pcov = curve_fit(f, bins[0:-1], hist)

x_plot = np.linspace(bins[0], bins[-1], 1000)


plt.plot(x_plot, f(x_plot, popt[0]))
plt.xscale("log")
plt.yscale("log")
print(popt)


plt.show()