import numpy as np
import matplotlib.pyplot as plt
from numba import jit

N = 100
h = 0.02


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
def ManyEvolve(u, v, h, dt, n, F, k):
    for i in range(n):
        u, v = Evolve(u, v, h, dt, F, k)
        print(i)

    return u, v

t = [0, 500, 1500, 2500, 5000, 10000]
fig = plt.figure()
ax1 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(233)
ax4 = plt.subplot(234)
ax5 = plt.subplot(235)
ax6 = plt.subplot(236)
axs = [ax1, ax2, ax3, ax4, ax5, ax6]

u_ev = []
v_ev = []
u_ev.append(u.copy())
v_ev.append(v.copy())

for i in range(5):
    time = t[i + 1] - t[i]
    u, v = ManyEvolve(u, v, h, dt, time, F2, k2)
    u_ev.append(u)
    v_ev.append(v)


for i in range(len(axs)):
    u_plot = u_ev[i]
    v_plot = v_ev[i]
    ax = axs[i]
    ax.imshow(u_plot)
    ax.set_title("Układ po " + str(t[i]))

plt.show()
