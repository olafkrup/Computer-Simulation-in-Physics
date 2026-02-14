import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# x = [x, dx/dt]

x0 = np.array([0, 0.15])
a = 1
b = 1
c = 0.2
f = 0.2
omega = 2*np.pi * 0.213

def func(t, x):
    return [x[1], b*x[0] - a*x[0]**3 - c*x[1] + f * np.cos(omega*t)]

t = np.linspace(0, 100, 1000)
t_span = [np.min(t), np.max(t)]

f = 0
sol0 = solve_ivp(func, t_span, x0, t_eval=t)
x_0 = sol0.y[0]
x_eq = np.mean(x_0[-10:])

f = 0.2
sol = solve_ivp(func, t_span, x0, t_eval=t)
x = sol.y


plt.subplot(311)
plt.title("f = " + str(f))
plt.plot(t, x[0])
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.hlines(x_eq, t_span[0], t_span[1], color='red')

plt.subplot(312)
plt.plot(t, x[1])
plt.xlabel("$t$")
plt.ylabel("$v(t)$")

plt.subplot(313)
plt.plot(x[0], x[1])
plt.xlabel("$x(t)$")
plt.ylabel("$v(t)$")
plt.scatter(x_eq, 0, color='red', label="Punkt Równowagi")
plt.legend()


plt.show()


