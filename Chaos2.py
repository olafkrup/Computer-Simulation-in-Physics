import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# x = [x, dx/dt]

# initial condition
x0 = np.array([1, 0.15])
a = 1
b = 1
c = 0.2
f = 0.15
omega = 2*np.pi * 0.213

# ODE
def func(t, x):
    return [x[1], b*x[0] - a*x[0]**3 - c*x[1] + f * np.cos(omega*t)]


t = np.linspace(0, 700, 10000)
t_span = [np.min(t), np.max(t)]

f = 0
sol0 = solve_ivp(func, t_span, x0, t_eval=t)
x_0 = sol0.y[0]
x_eq = np.mean(x_0[-10:])

plt.subplot(311)
f = 0.15
sol = solve_ivp(func, t_span, x0, t_eval=t)
x = sol.y
plt.plot(x[0][-200:], x[1][-200:])
plt.title("f = " + str(f))
plt.ylabel("$v(t)$")
plt.scatter(x_eq, 0, color='red', label="Punkt Równowagi")
plt.legend()

plt.subplot(312)
f = 0.3
sol = solve_ivp(func, t_span, x0, t_eval=t)
x = sol.y
plt.plot(x[0][-200:], x[1][-200:])
plt.title("f = " + str(f))
plt.ylabel("$v(t)$")
plt.scatter(x_eq, 0, color='red', label="Punkt Równowagi")
plt.legend()

#f = 0.39 - chaos

plt.subplot(313)
f = 2
sol = solve_ivp(func, t_span, x0, t_eval=t)
x = sol.y
plt.plot(x[0],  x[1])
plt.title("f = " + str(f))
plt.xlabel("$x(t)$")
plt.ylabel("$v(t)$")
plt.scatter(x_eq, 0, color='red', label="Punkt Równowagi")
plt.legend()



plt.show()
