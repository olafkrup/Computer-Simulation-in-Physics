import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation

colors = ['deepskyblue', 'deeppink', 'black']
cmap = ListedColormap(colors)

bounds = [-0.5, 0.5, 1.5, 2.5]
norm = BoundaryNorm(bounds, cmap.N)

N = 40
A = 5
B = 5
E = 3

F_size = 300
S_size = 10

T = 300

grid = np.zeros((N, N))

class Fish:
    items = []
    def __init__(self, x, y, A):
        self.x = x
        self.y = y
        self.A = A
        self.t = 0
        Fish.items.append(self)

    def move(self, grid):
        M = grid.copy()

        if self.dead(grid):
            return M

        self.t = self.t + 1
        moves = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        pos = np.array([self.x, self.y])
        moves = pos + moves
        moves = moves % N
        np.random.shuffle(moves)
        for xy in moves:
            if M[xy[0], xy[1]] == 0:
                # Breeding
                if self.t % self.A == 0:
                    Fish(self.x, self.y, self.A)
                else:
                    M[self.x, self.y] = 0

                self.x = xy[0]
                self.y = xy[1]
                M[xy[0], xy[1]] = 1
                break
        return M

    def dead(self, grid):
        if grid[self.x, self.y] != 1:
            Fish.items.remove(self)
            return True
        return False

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        return False



class Shark:
    items = []
    def __init__(self, x, y, B, E):
        self.x = x
        self.y = y
        self.B = B
        self.E = E
        self.E0 = E
        self.t = 0
        Shark.items.append(self)

    def move(self, grid):
        M = grid.copy()
        self.t = self.t + 1
        moves = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        pos = np.array([self.x, self.y])
        moves = pos + moves
        moves = moves % N
        np.random.shuffle(moves)

        for xy in moves:
            if M[xy[0], xy[1]] == 1:
                self.E = self.E0

                # Breeding
                if self.t % self.B == 0:
                    Shark(self.x, self.y, self.B, self.E)
                else:
                    M[self.x, self.y] = 0

                self.x = xy[0]
                self.y = xy[1]
                M[xy[0], xy[1]] = 2
                return M

        for xy in moves:
            if M[xy[0], xy[1]] == 0:
                if self.E <= 0:
                    Shark.items.remove(self)
                    M[self.x, self.y] = 0
                    return M

                # Breeding
                if self.t % self.B == 0:
                    Shark(self.x, self.y, self.B, self.E)
                else:
                    M[self.x, self.y] = 0

                self.x = xy[0]
                self.y = xy[1]
                M[xy[0], xy[1]] = 2
                break
        self.E = self.E - 1
        return M



def init(N, grid, F_size, S_size, A, B, E):
    M = grid.copy()
    xy = [[i, j] for i in range(N) for j in range(N) if i != j]

    np.random.shuffle(xy)

    for i in range(F_size + S_size):
        if i < F_size:
            M[xy[i][0], xy[i][1]] = 1
            Fish(xy[i][0], xy[i][1], A)
        else:
            M[xy[i][0], xy[i][1]] = 2
            Shark(xy[i][0], xy[i][1], B, E)

    return M

def F_evolve(grid):
    M = grid.copy()

    for fish in Fish.items:
        M = fish.move(M)
    return M

def S_evolve(grid):
    M = grid.copy()
    for shark in Shark.items:
        M = shark.move(M)
    return M

def ManyEvolve(grid, T):
    M = grid.copy()
    grid_list = [M]
    for i in range(T):
        M = F_evolve(M)
        grid_list.append(M)
        M = S_evolve(M)
        grid_list.append(M)

    return grid_list


grid = init(N, grid, F_size, S_size, A, B, E)

grid_list = ManyEvolve(grid, T)

fig = plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ims = []

sharks = []
fishes = []

for M in grid_list:
    im1 = ax1.imshow(M, cmap=cmap, norm=norm)
    ax1.set_title("A = " + str(A) + " ; B = " + str(B))

    shark_numb = np.sum(M == 2)
    fish_numb = np.sum(M == 1)
    sharks.append(shark_numb)
    fishes.append(fish_numb)

    #ax2.cla()
    ax2.set_xlabel("Sharks")
    ax2.set_ylabel("Fishes")
    im2, = ax2.plot(sharks, fishes, color="red")


    ims.append([im1, im2])


animation0 = animation.ArtistAnimation(fig, ims, interval=200, repeat = True, repeat_delay=1000, blit=True)
animation0.save("Wa-Tor2.gif")
plt.show()
