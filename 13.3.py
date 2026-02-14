import numpy as np
import matplotlib.pyplot as plt


N = 200
A = 3
B = 5
E = 3

F_size = 300
S_size = 20

T = 500

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
    ret = [[len(Fish.items), len(Shark.items)]]
    for i in range(T):
        M = F_evolve(M)
        M = S_evolve(M)
        ret.append([len(Fish.items), len(Shark.items)])
        print(i)

    return np.array(ret)


grid = init(N, grid, F_size, S_size, A, B, E)

amount_list = ManyEvolve(grid, T)

fig = plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

fishes = amount_list[:, 0]
sharks = amount_list[:, 1]


T_space = np.linspace(0, T, T+1)

ax1.plot(T_space, fishes, label="fishes")
ax1.plot(T_space, sharks, label="sharks")
ax1.set_xlabel("T")
ax1.set_ylabel("N")
ax1.set_title("Time Evolution for: " + "A = " + str(A) + " B = " + str(B))
ax1.legend()

ax2.plot(fishes, sharks)
ax2.set_xlabel("fishes")
ax2.set_ylabel("sharks")
ax2.set_title("Phase Space Evolution")

plt.show()
