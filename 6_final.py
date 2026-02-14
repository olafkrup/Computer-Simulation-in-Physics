import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

N = 16
box_size = 8.0
eps = 1.0
sigma = 1.0
size = sigma / 2
dt = 0.0025
T0 = 0.1
kB = 1
m = 1
v_mean = np.sqrt(kB * T0 / m) #mean velocity in one direction
variance = v_mean * (3 - 8/np.pi)

def min_img(r = np.array([0, 0])):
    for i in (0, 1):
        x_i = r[i]
        if x_i > 0.5 * box_size:
            x_i -= box_size
        if x_i < -0.5 * box_size:
            x_i += box_size
        r[i] = x_i
    return r

class Particle:
    all_particles = [] # keeps all the instances of the class

    def __init__(self, size, r, v):
        self.r = r
        self.v = v
        self.F = np.array([0, 0])
        self.size = size
        self.v_half = None  #for leapfrog algorithm
        self.v_mu = None
        Particle.all_particles.append(self)

    def inside_check(self, box_size): # checks boundary conditions
        for i in (0, 1):
            x = self.r[i]
            if x < 0:
                x += box_size
            if x > box_size:
                x -=  box_size
            self.r[i] = x

    def Force(self, other): # returns the force between two particles
        if self == other:
            return np.array([0, 0])
        else:
            r = np.array(other.r - self.r)
            r = min_img(r)
            dist = np.linalg.norm(r)
            s_dist = sigma / dist
            if dist > 2.2 * sigma:
                return np.array([0, 0])
            else:
                return (-48 * eps / sigma**2) * (s_dist**14 - 0.5*(s_dist**8)) * r

    def Potential(self, other): # calculates potential energy between two particles
        if self == other:
            return 0
        else:
            r = np.array(other.r - self.r)
            r = min_img(r)
            dist = np.linalg.norm(r)
            s_dist = sigma / dist
            return 4 * eps * (s_dist**12 - s_dist**6)

    def Force_update(self): # calculates net force of one particle
        self.F = np.array([0, 0])
        for p in Particle.all_particles:
            self.F = self.F + self.Force(p)

    def Half_step(self, dt):
        if self.v_half is None:
            self.v_half = self.v - 0.5* self.F * dt
        self.v_mu = self.v_half + self.F * dt / 2

    def Update_position(self, eta, dt):
        if self.v_half is None:
            self.v_half = self.v - 0.5* self.F * dt
        v = self.v_half
        self.v_half = (2*eta - 1) * self.v_half + eta * self.F * dt
        self.r = self.r + self.v_half * dt
        self.v = (v + self.v_half)/2

    def __str__(self):
        return "x = " + str(self.r[0]) + " ; y = " + str(self.r[1]) +  " ; v = " + str(self.v)

def Temp_calc():
    E_k = 0
    for p in Particle.all_particles:
        E_k += m * (np.linalg.norm(p.v) ** 2) / 2
    return E_k / (N*kB)

def initialise(N, v_mean, sigma, size, box_size, T0):
    v = np.random.normal(v_mean, sigma,(N, 2))
    v_mean0 = v.mean(axis=0)
    v = (v - v_mean0)
    T_ = (np.sum(v**2, axis=1)/(2* m)).mean()
    v = v*((T0/T_)**0.5)
    r = np.array([[i, j] for i in range(4) for j in range(4)] )
    r = r * box_size/4 + 1
    for i in range(N):
        Particle(size, r[i], v[i])

def Update(dt):
    E_k = 0
    for p in Particle.all_particles:
        p.Force_update()
        p.Half_step(dt)
        E_k += m * (np.linalg.norm(p.v_mu) ** 2) / 2

    T = E_k / (N*kB)
    eta = (T0/T)**0.5

    for p in Particle.all_particles:
        p.Update_position(eta, dt)
        p.inside_check(box_size)

def Parameter_calc(): # counting energy, Temperature and Pressure
    E_p = 0
    E_k = 0
    P = 0
    V = box_size**2
    for p1 in Particle.all_particles:
        E_k += m * (np.linalg.norm(p1.v) **2)/2
        for p2 in Particle.all_particles:
            E_p += p1.Potential(p2) # double counting
            r = p1.r - p2.r
            r = min_img(r)
            P = P + np.dot(r, p1.Force(p2))
    T = E_k / (N * kB)
    P = P / (4 * V)
    P += N*kB*T/V
    E_p = E_p/2
    return (E_k, E_p, E_k + E_p, T, P)


initialise(N, v_mean, variance, size, box_size, T0=T0)

t = [0]

colors = np.random.rand(N, 3) # random colors

Parameters = []
E_k, E_p, E, T, P = Parameter_calc()
Parameters.append([E_k, E_p, E, T, P])

def init():
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 2, 1)
    return fig,ax1,ax2

def image_update(i):
    global Parameters
    global t
    ax1.cla()
    for j in range(50):
        Update(dt)
    t_ = t[-1]
    t.append(t_ + 50*dt)
    for p in Particle.all_particles:
        cir = Circle((p.r[0], p.r[1]), radius = p.size, color=(colors[Particle.all_particles.index(p)]))
        ax1.add_patch(cir)
        ax1.set_xlim((0, box_size))
        ax1.set_ylim((0, box_size))

    E_k, E_p, E, T, P = Parameter_calc()
    Parameters.append([E_k, E_p, E, T, P])

    k = np.array(Parameters)
    textstr = '\n'.join((
        r'$E_k=%.2f$' % (E_k,),
        r'$E_p=%.2f$' % (E_p,),
        r'$E=%.2f$' % (E,),
        r'$T=%.2f$' % (T,),
        r'$P=%.2f$' % (P,)))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='pink', alpha=0.5)

    # place a text box in upper left in axes coords
    ax1.text(-1, 9, textstr, fontsize=14, verticalalignment='top', bbox=props)

    ax2.cla()
    ax2.plot(t, k[:, 0], label='E_k')
    ax2.plot(t, k[:, 1], label='E_p')
    ax2.plot(t, k[:, 2], label='E')
    ax2.plot(t, k[:, 3], label='T')
    ax2.plot(t, k[:, 4], label='P')
    ax2.set_xlabel(r'$t$')
    ax2.legend()

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

anim = animation.FuncAnimation(fig, image_update, init_func=init, interval=20, frames=101, blit=False)
plt.show()