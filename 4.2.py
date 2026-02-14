import numpy as np
import matplotlib.pyplot as plt

# r = [x,y] ; dr/dt = [dx/dt, dy/dt]
dt = 0.00001
t_max = 1.5
N = int(t_max/dt) + 1
G = 0.01
M = 1
t = np.linspace(0, t_max, N)

r10 = np.array(0.97000436, -0.24308753)
r20 = - r10.copy()
r30 = np.array([0, 0])

v30 = np.array([0.93240737, 0.86473146])
v10 = -2*v30.copy()
v20 = -2*v30.copy()


# d2r/dt2 = -GMm/r^2

def F(r):
    return -G*M*r/((np.dot(r, r))**1.5)

def Verlet(r, v, dt, N):
    r_v = r.copy()
    v_v = v.copy()
    r_v[1] = r_v[0] + v_v[0]*dt
    for i in range(1, N-1):
        r_v[i+1] = 2*r_v[i] - r_v[i-1] + F(r_v[i])*(dt**2)
    return r_v, v_v


r, v = np.empty((N, 2)), np.empty((N, 2))
r[0] = r0
v[0] = v0

r_v, v_v = Verlet(r, v, dt, N)
