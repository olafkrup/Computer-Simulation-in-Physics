import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def f(x, a):
    return a*x

m1 = [0.5, 0, 0, 0.5, 0.25, ((3)**0.5) / 4]
m2 = [0.5, 0, 0, 0.5, 0.0, 0]
m3 = [0.5, 0, 0, 0.5, 0.5, 0]

n1 = [0.001, 0.0, 0.0, 0.16, 0.0, 0.0]
n2 = [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44]
n3 = [0.2,-0.26, 0.23, 0.22, 0.0, 1.6]
n4 = [0.85, 0.04,-0.04, 0.85, 0.0, 1.6]

n = [n1, n2, n3, n4]

m = [m1, m2, m3]

p_m = 1/3

p_n = [0.02, 0.09, 0.1, 0.79]

v = np.array([0, 0])

def sym_m(v):
    x = np.random.rand()
    if x < p_m:
        m0 = m[0]
    elif x < 2*p_m:
        m0 = m[1]
    else:
        m0 = m[2]
    M = np.array(m0[:4]).reshape(2, 2)
    w = np.array([m0[4], m0[5]])
    return np.matmul(M, v) + w


def sym_n(v):
    x = np.random.rand()
    if x < p_n[0]:
        n0 = n[0]
    elif x < p_n[1]:
        n0 = n[1]
    elif x < p_n[2]:
        n0 = n[2]
    else:
        n0 = n[3]

    N = np.array(n0[:4]).reshape(2, 2)
    w = np.array([n0[4], n0[5]])
    return np.matmul(N, v) + w


N = 10000
a = np.empty((N, 2))
b = np.empty((N, 2))
b[0] = v
a[0] = v
for i in range(N-1):
    a[i+1] = sym_m(a[i])
    b[i+1] = sym_n(b[i])

n = 13

N_r = np.zeros(n)
z = np.arange(n)

for r in range(n):
    H_a, xedges, yedges = np.histogram2d(a[:, 0], a[:, 1], bins = 2**r)
    for x in H_a:
        for y in x:
            if y > 0:
                N_r[r] += 1

plt.scatter(z, np.log(N_r))
plt.plot(z, np.log(N_r))

popt, pcov = curve_fit(f, z, N_r)
print(popt)

plt.show()