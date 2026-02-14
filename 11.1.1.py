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
    for i in range(n):
        u, v = Evolve(u, v, dx, dt)
        print(i)

    return u, v

t = [0, 500, 1500, 2500, 10000]
fig = plt.figure()
ax1 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(233)
ax4 = plt.subplot(234)
ax5 = plt.subplot(236)
axs = [ax1, ax2, ax3, ax4, ax5]

u_ev = []
v_ev = []
u_ev.append(u.copy())
v_ev.append(v.copy())

for i in range(4):
    time = t[i + 1] - t[i]
    u, v = ManyEvolve(u, v, dx, dt, time)
    u_ev.append(u)
    v_ev.append(v)


for i in range(len(axs)):
    u_plot = u_ev[i]
    v_plot = v_ev[i]
    ax = axs[i]
    ax.plot(Line, u_plot, label="Aktywator")
    ax.plot(Line, v_plot, label="Inhibitor")
    ax.set_xlabel("x")
    ax.legend()
    ax.set_title("Wykres po t = " + str(t[i]))

plt.show()
