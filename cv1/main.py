from collections import defaultdict

from matplotlib.animation import FuncAnimation

from func import Function, f_eval, f_names
import matplotlib.pyplot as plt

import numpy as np
class Solution:
    def __init__(self, dimension: int,
                 lower_bound: float,
                 upper_bound: float,
                 func: Function):
        self.dimension = dimension
        self.lB = lower_bound
        self.uB = upper_bound
        self.func = func

        self.best_fit = np.empty(dtype=np.float64, shape=(0,))
        self.best_params = np.empty(shape=(0, self.dimension)).reshape(-1, self.dimension)

        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})

    def search(self, iterations, population):
        for i in range(iterations):
            iter_fit = np.inf
            iter_params = np.zeros(self.dimension)
            for j in range(population):
                params = np.random.uniform(self.lB, self.uB, self.dimension)
                fit = self.func.func(params)
                if fit < iter_fit:
                    iter_fit = fit
                    iter_params = params


            if (len(self.best_fit) == 0 or
                    iter_fit < self.best_fit[-1]):
                self.best_fit = np.append(self.best_fit, iter_fit)
                self.best_params = np.vstack([self.best_params, iter_params])
                # self.plot(self.best_fit.shape[0] - 1)
                print(f'Better:\t{i}, Z={iter_fit}')
            else:
                print(f'Worse:\t{i}')

        return


    def plot(self, i: int):
        self.ax.clear()

        #
        # surface
        #
        x = np.linspace(self.lB, self.uB, 100)
        y = np.linspace(self.lB, self.uB, 100)
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
        z = self.best_fit[i]

        # surface
        surf = self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)

        # best points
        scatter = self.ax.scatter(x, y, z, c='r', marker='o')

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        plt.title(f'Iter: {i}, Z={self.best_fit[i]}')

        return surf, scatter


    def animate(self):
        # Creating the animation
        anim = FuncAnimation(self.fig, self.plot, frames=self.best_fit.shape[0], repeat=False)
        anim.save(f'plots/{self.func.name}.gif', writer='imagemagick', fps=3)

        # Save or show the animation
        # plt.show()
        
        
    def animate_best(self):
        def rotate(i):
            self.ax.view_init(elev=30., azim=i)
            return self.plot(len(self.best_fit) - 1)  # Plot the last frame (best solution)

        anim = FuncAnimation(self.fig, rotate, frames=np.arange(0, 360, 2), repeat=False)
        anim.save(f'plots/{self.func.name}_rotate.gif', writer='imagemagick', fps=30)


def main():
    # func = Solution(2, -4, 4, Function.get_func("Michalewicz"))
    # func.search(1000, 10)
    # func.animate()

    # return
    for name in f_names:
        func = Solution(2, -4, 4, Function.get_func(name))
        func.search(1000, 10)
        # func.animate()
        func.animate_best()

if __name__ == '__main__':
    main()