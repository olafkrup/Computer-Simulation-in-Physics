import numpy as np
import matplotlib.pyplot as plt

# r = [x,y] ; dr/dt = [dx/dt, dy/dt]
dt = 0.0001
t_max = 10
N = int(t_max/dt) + 1
G = 0.01
M = 500
m = 0.1
t = np.linspace(0, t_max, N)

r0 = np.array([2, 0])
v0 = np.array([0, 1])

# d2r/dt2 = -GMm/r^2

def F(r):
    return -G*M*r/((np.dot(r, r))**1.5)

def Euler(r, v, dt, N):
    r_e = r.copy()
    v_e = v.copy()
    for i in range(N-1):
        r_e[i+1] = r_e[i] + v_e[i]*dt + (dt**2)*F(r_e[i])/2
        v_e[i+1] = v_e[i] + dt*F(r_e[i])
    return r_e, v_e

def Verlet(r, v, dt, N):
    r_v = r.copy()
    v_v = v.copy()
    r_v[1] = r_v[0] + v_v[0]*dt
    for i in range(1, N-1):
        r_v[i+1] = 2*r_v[i] - r_v[i-1] + F(r_v[i])*(dt**2)
    return r_v, v_v

def Frog(r, v, dt, N):
    r_f = r.copy()
    v_f_half = v.copy()
    v_f_half[0] = v_f_half[0] - F(r_f[0])*dt/2
    for i in range(N-1):
        v_f_half[i+1] = v_f_half[i] + F(r_f[i])*dt
        r_f[i+1] = r_f[i] + v_f_half[i+1]*dt
    return r_f, v_f_half

r, v = np.empty((N, 2)), np.empty((N, 2))
r[0] = r0
v[0] = v0

r_e, v_e = Euler(r, v, dt, N)
r_v, v_v = Verlet(r, v, dt, N)
r_f, v_f_half = Frog(r, v, dt, N)

v_f = np.zeros_like(v_f_half)

for i in range(N-2):
    v_v[i+1] = (r_v[i+2] - r_v[i])/(2*dt)
v_v[-1] = v_v[-2] + F(r_v[-1])*dt

for i in range(N):
    v_f[i] = v_f_half[i] + 0.5*F(r_v[i])*dt

Ek_e = np.zeros(N)
Ek_v = np.zeros(N)
Ek_f = np.zeros(N)
Ep_e = np.zeros(N)
Ep_v = np.zeros(N)
Ep_f = np.zeros(N)


for i in range(N):
    Ek_e[i] = 0.5*m*np.dot(v_e[i], v_e[i])
    Ek_v[i] = 0.5*m*np.dot(v_v[i], v_v[i])
    Ek_f[i] = 0.5*m*np.dot(v_f[i], v_f[i])

    Ep_e[i] = -G*M*m/(np.dot(r_e[i], r_e[i])**0.5)
    Ep_v[i] = -G*M*m/(np.dot(r_v[i], r_v[i])**0.5)
    Ep_f[i] = -G*M*m/(np.dot(r_f[i], r_f[i])**0.5)

E_e = Ep_e + Ek_e
E_v = Ep_v + Ek_v
E_f = Ep_f + Ek_f

# plotting
plt.subplot(221)
plt.title("Trajectories")
plt.plot(r_e[:, 0], r_e[:, 1], label="Euler")
plt.plot(r_v[:, 0], r_v[:, 1], label="Verlet")
plt.plot(r_f[:, 0], r_f[:, 1], label="Frog")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.scatter(0, 0, c='red')
plt.legend()

plt.subplot(222)
plt.title("Energy")
plt.plot(t, E_e, label='Euler')
plt.plot(t, E_v, label='Verlet')
plt.plot(t, E_f, label='Frog')
plt.xlabel("$t$")
plt.ylabel("$E$")
plt.legend()

plt.subplot(223)
plt.title("Kinetic Energy")
plt.plot(t, Ek_e, label='Euler')
plt.plot(t, Ek_v, label='Verlet')
plt.plot(t, Ek_f, label='Frog')
plt.xlabel("$t$")
plt.ylabel("$Ek$")
plt.legend()

plt.subplot(224)
plt.title("Potential Energy")
plt.plot(t, Ep_e, label='Euler')
plt.plot(t, Ep_v, label='Verlet')
plt.plot(t, Ep_f, label='Frog')
plt.xlabel("$t$")
plt.ylabel("$Ep$")
plt.legend()

plt.show()