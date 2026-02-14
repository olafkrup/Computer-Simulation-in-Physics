import numpy as np
import matplotlib.pyplot as plt
from numba import jit

L = 2
N = 100
dx = 0.02
Line = np.linspace(dx, L, N)

u = np.ones(N)
v = np.zeros(N)
N1=N//4; N2=3*N//4
u[N1:N2+1] = 0.4 + np.random.rand(N2-N1+1) * 0.2
v[N1:N2+1] = 0.2 + np.random.rand(N2-N1+1) * 0.2

Du = 0.00002
Dv = 0.00001
F= 0.025
k = 0.055

dt = 1

@jit(nopython = True)
def Laplace(f, dx):
    f_next = np.roll(f, 1)
    f_prev = np.roll(f, -1)
    return (f_next + f_prev - 2*f) / dx**2

@jit(nopython = True)
def Evolve(u, v, dx, dt):
    u_next = u + (Du * Laplace(u, dx) - u*v**2 + F - F*u) * dt
    v_next = v + (Dv * Laplace(v, dx) + u*v**2 - (F + k)*v) * dt
    return u_next, v_next

@jit(nopython = True)
def ManyEvolve(u, v, dx, dt, n):
    u_ret = []
    v_ret = []
    u_ret.append(u)
    v_ret.append(v)
    for i in range(n):
        u, v = Evolve(u, v, dx, dt)
        u_ret.append(u)
        v_ret.append(v)
        print(i)

    return u_ret, v_ret


u_plot, v_plot = ManyEvolve(u, v, dx, dt, 10000)
plt.imshow(u_plot, aspect='auto')

plt.show()