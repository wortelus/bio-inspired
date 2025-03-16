import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from b_core.func import Function


class DifferentialEvolution:
    def __init__(self,
                 func: Function,
                 n_population,
                 n_generations,
                 bounds,
                 f=0.5,
                 cr=0.9
                 ):
        self.func = func
        self.n_population = n_population
        self.n_generations = n_generations
        self.bounds = bounds
        self.f = f
        self.cr = cr
        
        self.cost = np.empty(self.n_population)

        self.best_cost = np.empty(dtype=np.float64, shape=(0,))
        self.best_params = np.empty(shape=(0, len(self.bounds))).reshape(-1, len(self.bounds))
        self.best_gen = np.empty(dtype=np.int32, shape=(0,))

        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7), dpi=100)

    def run(self):
        pop = np.random.rand(self.n_population, len(self.bounds))
        g = 0
        
        self.cost = np.array([self.func.func(p) for p in pop])
        
        while g < self.n_generations:
            print(f'Generation: {g}')
            # Kopie populace
            new_pop = pop[:]
            for i, p in enumerate(pop):
                print(f'Generation: {g}, Individual: {i}')
                
                # Výběr r1, r2, r3, aby r1 != r2 != r3 != i
                r_selection = [idx for idx in range(self.n_population) if idx != i]
                r1, r2, r3 = np.random.choice(r_selection, 3, replace=False)
                x_r1, x_r2, x_r3 = pop[r1], pop[r2], pop[r3]

                # Mutace
                mutant = (x_r1 - x_r2) * self.f + x_r3
                # Clipping do hranic (bounds)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Náhodná hodnota od 0 do 1 pro každý prvek s šancí Crossover
                j_rnd = np.random.rand(len(p))
                # Minimálně 1 Crossover, 
                # To ověříme tak, že si vygenerujeme náhodný index pro který Crossover provedeme
                id_rnd = np.random.randint(0, len(p))
                # Výsledný vektor
                u = np.zeros_like(p)

                # Crossover
                for j in range(len(p)):
                    if (j == id_rnd  # at least one element is from v
                            or j_rnd[j] < self.cr):  # crossover chance
                        u[j] = mutant[j]
                    else:
                        u[j] = p[j]

                # Evaluace
                f_u = self.func.func(u)
                f_p_iter = self.func.func(p)
                if f_u <= f_p_iter:
                    new_pop[i] = u
                    self.cost[i] = f_u

            pop = new_pop
            g += 1
            
            # Nejlepší cost a parametry a generace
            best_idx = np.argmin(self.cost)
            if (len(self.best_cost) == 0 or
                    self.cost[best_idx] < self.best_cost[-1]):
                self.best_cost = np.append(self.best_cost, self.cost[best_idx])
                self.best_params = np.vstack([self.best_params, pop[best_idx]])
                self.best_gen = np.append(self.best_gen, g)
                print(f'Better:\t{g}, Z={self.best_cost[-1]}')
            else:
                print(f'Worse:\t{g}')
            
        return pop, self.cost

    #
    # Vzato z předchozích cvičení a upraveno
    #
    def plot(self, i: int):
        self.ax.clear()
        
        lb = self.bounds[0]
        ub = self.bounds[1]

        #
        # surface
        #
        x = np.linspace(lb, ub, 100)
        y = np.linspace(lb, ub, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                params = [X[j, k], Y[j, k]]
                Z[j, k] = np.array(self.func.func(params))

        #
        # best points
        #
        x = self.best_params[i, 0]
        y = self.best_params[i, 1]
        z = self.best_cost[i]
        generation = self.best_gen[i]

        # surface
        surf = self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)

        # best points
        scatter = self.ax.scatter(x, y, z, c='r', marker='o')

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        plt.title(f'Generace: {generation}, Z={z}')

        return surf, scatter

    def animate(self):
        # Creating the animation
        anim = FuncAnimation(self.fig, self.plot, frames=self.best_cost.shape[0], repeat=False)
        # anim.save(f"plots/{self.func.name}.gif", writer='imagemagick', fps=10)
        anim.save(f"plots/{self.func.name}.mp4", writer='ffmpeg', fps=10, dpi=200)

        # Save or show the animation
        # plt.show()

    def animate_best(self):
        def rotate(i):
            self.ax.view_init(elev=30., azim=i)
            return self.plot(len(self.best_cost) - 1)  # Plot the last frame (best solution)

        anim = FuncAnimation(self.fig, rotate, frames=np.arange(0, 360, 2), repeat=False)
        anim.save(f'plots/{self.func.name}_rotate.gif', writer='imagemagick', fps=30)