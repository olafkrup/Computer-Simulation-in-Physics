import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import matplotlib.animation as animation

N = 100
h = 0.02

cmap = 'coolwarm'

u = np.ones((N,N))
print(u)
v = np.zeros((N,N))
N1=N//4; N2=3*N//4
u[N1:N2+1, N1:N2+1] = 0.4 + np.random.rand(N2-N1+1, N2-N1+1) * 0.2
v[N1:N2+1, N1:N2+1] = 0.2 + np.random.rand(N2-N1+1, N2-N1+1) * 0.2
Du = 0.00002
Dv = 0.00001
F= 0.025
k = 0.055

F2 = 0.03
k2 = 0.062

F3 = 0.01
k3 = 0.047

F4 = 0.04
k4 = 0.062

F5 = 0.052
k5 = 0.0615

F6 = 0.037
k6 = 0.06

F7 = 0.04
k7 = 0.08

dt = 1

#@jit(nopython = True)
def Laplace(f, h):
    fy_next = np.roll(f, 1, axis=0)
    fy_prev = np.roll(f, -1, axis=0)
    fx_next = np.roll(f, 1, axis=1)
    fx_prev = np.roll(f, -1, axis=1)
    return (fx_next + fx_prev + fy_next + fy_prev - 4*f) / h**2

#@jit(nopython = True)
def Evolve(u, v, h, dt, F, k):
    u_next = u + (Du * Laplace(u, h) - u*v**2 + F - F*u) * dt
    v_next = v + (Dv * Laplace(v, h) + u*v**2 - (F + k)*v) * dt
    return u_next, v_next

#@jit(nopython = True)
def ManyEvolve(u, v, dx, dt, n, F, k):
    u_ret = []
    v_ret = []
    u_ret.append(u)
    v_ret.append(v)
    for i in range(n):
        u, v = Evolve(u, v, dx, dt, F, k)
        u_ret.append(u)
        v_ret.append(v)
        print(i)

    return u_ret, v_ret

T = 10000

u_ev, v_ev = ManyEvolve(u, v, h, dt, T, F, k)
ims = []
fig, ax = plt.subplots()


for i in range(0, T, 100):
    im = ax.imshow(u_ev[i], animated=True, cmap=cmap, interpolation='nearest')
    ims.append([im])

animation0 = animation.ArtistAnimation(fig, ims, interval=50, repeat = True, repeat_delay=1000, blit=True)
plt.show()